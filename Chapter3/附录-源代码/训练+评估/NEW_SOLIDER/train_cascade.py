import argparse
import datetime
import os.path as osp
import time
import sys
import os
import torch
import torch.utils.data

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
# from engine import evaluate_performance, train_one_epoch
from engine_cascade import evaluate_performance, train_one_epoch
# from models.seqnet import SeqNet
# from models.seqnet_fpn import SeqNet
# from models.seqnet_import_fpn import SeqNet
from models.seqnet_cascade import SeqNet
from models.softmax_loss import SoftmaxLoss
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
# log日志记录
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")
    #  在终端和日志文件中记录相同的消息
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    #  空方法
    def flush(self):
        pass
def main(args):
    # 将配置文件和命令行选项中的配置与默认配置进行合并，然后冻结配置
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)      # 命令行更新配置文件
    cfg.merge_from_list(args.opts)              # 合并参数
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    # 随机种子
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = SeqNet(cfg)     # 跳入模型
    model.to(device)
    # print(model)         # 输出模型

    print("Loading data")       # 导入训练集、验证集
    train_loader = build_train_loader(cfg)
    # print("cfg", cfg)
    gallery_loader, query_loader = build_test_loader(cfg)

    if cfg.MODEL.LOSS.USE_SOFTMAX:
        softmax_criterion_s2 = SoftmaxLoss(cfg)
        softmax_criterion_s3 = SoftmaxLoss(cfg)
        softmax_criterion_s2.to(device)
        softmax_criterion_s3.to(device)
    # 评估模型pth
    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)
    # SGD优化器
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.MODEL.LOSS.USE_SOFTMAX:
        params_softmax_s2 = [p for p in softmax_criterion_s2.parameters() if p.requires_grad]
        params_softmax_s3 = [p for p in softmax_criterion_s3.parameters() if p.requires_grad]
        params.extend(params_softmax_s2)
        params.extend(params_softmax_s3)

    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,                  # 学习率基准值
        momentum=cfg.SOLVER.SGD_MOMENTUM,       # 动量参数0.9
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,   # 权重衰减参数0.0005
    )
    # 学习率调度策略:学习率 milestones设为16，当前学习率与 "GAMMA" 相乘0.1减小后的学习率
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler)

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:  # 写入模式 "w"，分配给变量 f，cfg.dump() 将配置对象 cfg 转换为字符串格式
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    # 配置tensorboard，记录模型训练过程中的指标、损失函数、梯度
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        # 读模型
        # init_img = torch.zeros((1, 3, 864, 1504), device=device) #  生成假的图片作为输入
        # tfboard.add_graph(model, (init_img,))  # 模型及模型输入数据

        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, tfboard, softmax_criterion_s2, softmax_criterion_s3, output_dir)
        # update the learning rate
        lr_scheduler.step()
        # 仅保存最后5次权重
        # if epoch >= cfg.SOLVER.MAX_EPOCHS - 5:
        save_on_master(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            osp.join(output_dir, f"epoch_{epoch}.pth"),
        )

        #  测试最后5次
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
            # gallery_size=cfg.EVAL_GALLERY_SIZE,
        )

        # if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
        #     evaluate_performance(
        #         model,
        #         gallery_loader,
        #         query_loader,
        #         device,
        #         use_gt=cfg.EVAL_USE_GT,
        #         use_cache=cfg.EVAL_USE_CACHE,
        #         use_cbgm=cfg.EVAL_USE_CBGM,
        #     )
        #
        # if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
        #     save_on_master(
        #         {
        #             "model": model.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #             "lr_scheduler": lr_scheduler.state_dict(),
        #             "epoch": epoch,
        #         },
        #         osp.join(output_dir, f"epoch_{epoch}.pth"),
        #     )


    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    # 自定义目录存放日志文件
    log_path = './Logs/PRW/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
