import random
from mmdet.core.utils import cluster_hooks

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         OptimizerHook, build_optimizer)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook, ExtractFeatureHook, ClusterHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # 准备数据加载器
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    # 对比学习
    contrastive_learning = cfg.get('constrastive_learning', False)
    data_loaders = [
        # 创建每个数据集的加载器
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            constrastive_learning=contrastive_learning) for ds in dataset
    ]

    # 模型分配到 GPU
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        print(f"Find unused parameters: {find_unused_parameters}")
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # 构建优化器
    optimizer = build_optimizer(model, cfg.optimizer)
    # 构建一个基于 epoch 的训练器
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # FP16 设置（半精度训练）
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # 注册训练钩子： 注册学习率调度器、优化器钩子、检查点保存、日志记录和动量配置。
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # 注册评估钩子： 如果配置了验证（validate=True），构建验证数据集和数据加载器，并根据是否分布式选择注册 DistEvalHook（分布式评估）或 EvalHook（普通评估）。
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # 用户自定义钩子
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
    
    # unsupervised hooks:无监督学习钩子（SPCL）如果配置了 SPCL（自监督学习），则构建一个集群数据集和数据加载器，并注册提取特征和聚类钩子。
    if cfg.get('SPCL', None):
        train_cluster_dataset = build_dataset(cfg.data.train_cluster)
        train_cluster_dataloader = build_dataloader(
            train_cluster_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        pretrained_feature_file = cfg.get('pretrained_feature_file', None)
        extract_feature_hook = ExtractFeatureHook
        runner.register_hook(extract_feature_hook(train_cluster_dataloader, logger=logger, pretrained_feature_file=pretrained_feature_file))
        epoch_interval = cfg.get('cluster_epoch_interval', 1)
        cluster_hook = ClusterHook
        runner.register_hook(cluster_hook(data_loaders, logger=logger, cfg=cfg, epoch_interval=epoch_interval))
    # print(cfg.get('SPCL', None))
    # exit()
    #end
    # 恢复和加载模型
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    # 启动训练
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
