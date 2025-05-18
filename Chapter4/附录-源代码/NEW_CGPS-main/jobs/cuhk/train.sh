#!/bin/bash

# config_name="cuhk"
# config_path="../../configs/cgps/${config_name}.py" 

# python -u ../../tools/train.py ${config_path} >train_log.txt 2>&1 

CUDA_VISIBLE_DEVICES=1 python tools/train.py /public/home/G19940018/Group/csj/projects/CGPS-main/configs/cgps/cuhk.py 2>&1 | tee /public/home/G19940018/Group/csj/projects/CGPS-main/logs/sysu/train_b4_log.txt
