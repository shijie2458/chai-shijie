import torch
import sys
import yaml
from torch import nn

from engine_cascade import evaluate_performance
from datasets import build_test_loader
from defaults import get_default_cfg
from models.seqnet_cascade import SeqNet

# 载入模型
model_path = '/public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/results/prw/a+b-best/epoch_24.pth'
model = torch.load(model_path, map_location='cuda')

# 导出模型权重
with open('output_epoch_24.txt', 'w') as f:
    sys.stdout = f
    print(model)
    sys.stdout = sys.__stdout__  # 恢复标准输出

# 打印模型结构
# print(model)
# 评估模式
# cfg_path = '/public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/results/prw/a+b-best/config.yaml'
# # 读取配置文件
# with open(cfg_path, 'r') as f:
#     cfg = yaml.safe_load(f)
#     config = cfg['INPUT']

# cfg = get_default_cfg()
# device = torch.device('cuda')
# model = SeqNet(cfg)
# model.to(device)
# # 加载模型权重
# model_pth_path = '/public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/results/prw/a+b-best/epoch_24.pth'
# print("model_pth_path",model_pth_path)
# model.load_state_dict(torch.load(model_pth_path, map_location=device))

# state_dict = torch.load(model_pth_path)
# 从state_dict中移除不需要的键
# state_dict.pop('model')
# state_dict.pop('optimizer')
# state_dict.pop('lr_scheduler')
# state_dict.pop('epoch')
# 加载模型参数
# model.load_state_dict(state_dict)


# state_dict = torch.load(model_pth_path)
# model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
# model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_pth_path).items()})

# checkpoint = torch.load(model_pth_path)
# model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})

# state_dict = torch.load(model_pth_path)
# run_logger.info('{} start evaluation!!!'.format(model_pth_path))
# model.load_state_dict(state_dict['state_dict'])

# model = nn.DataParallel(model)
# checkpoint = torch.load(model_pth_path)
# model.load_state_dict(checkpoint['state_dict'])

# model.load_state_dict(torch.load(model_pth_path))
# model = torch.load(model_pth_path)

# print("Loading test data")
# gallery_loader, query_loader = build_test_loader(cfg)
# evaluate_performance(
#                 model,
#                 gallery_loader,
#                 query_loader,
#                 device,
#             )
