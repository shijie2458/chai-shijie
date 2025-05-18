#!/usr/bin/env bash
# 环境变量，用来控制每个进程在使用 OpenMP 时使用多少个线程
# export OMP_NUM_THREADS=2
#指定训练配置文件
# CONFIG=$1
CONFIG=/public/home/G19940018/Group/csj/projects/CGPS-main/configs/cgps/cuhk.py
#指定使用的 GPU 数量
GPUS=$2
#通信端口，默认值为 29500
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

PYTHONPATH="/public/home/G19940018/Group/csj/projects/CGPS-main":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    /public/home/G19940018/Group/csj/projects/CGPS-main/tools/train.py $CONFIG --launcher pytorch
