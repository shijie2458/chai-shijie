#python check_and_choose_GPU.py --ratio 0.9 --code 'sh run.sh'
# Swin Small(原)
#PRW GPU=1 device=0 csj3 SwinSmall_tea batch=3 lr=0.0003（9：30）out of memory
#CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/prw.yaml --resume --ckpt csj1/public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_small_tea.pth OUTPUT_DIR './results/prw/swin_small_tea' SOLVER.BASE_LR 0.0003 EVAL_PERIOD 5 MODEL.BONE 'swin_small' INPUT.BATCH_SIZE_TRAIN 3 MODEL.SEMANTIC_WEIGHT 0.6
#SYSU GPU=1 device=1 csj2  SwinSmall_tea batch=2 lr=0.0002（9：30）out of memory
#CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_small_tea.pth OUTPUT_DIR './results/cuhk_sysu/swin_small_tea' SOLVER.BASE_LR 0.0002 EVAL_PERIOD 5 MODEL.BONE 'swin_small' INPUT.BATCH_SIZE_TRAIN 2 MODEL.SEMANTIC_WEIGHT 0.6

# SwinTiny_tea(原)
#SYSU GPU=1 device=1 csj2 SwinTiny_tea batch=2 lr=0.0002（13：50）
#CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_tiny_tea.pth OUTPUT_DIR './results/cuhk_sysu/swin_tiny_tea' SOLVER.BASE_LR 0.0002 EVAL_PERIOD 5 MODEL.BONE 'swin_tiny' INPUT.BATCH_SIZE_TRAIN 2 MODEL.SEMANTIC_WEIGHT 0.6
#PRW GPU=2 device=2 csj1 SwinTiny_tea batch=3 lr=0.0003（13：47）
#CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/prw.yaml --resume --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_tiny_tea.pth OUTPUT_DIR './results/prw/swin_tiny_tea' SOLVER.BASE_LR 0.0003 EVAL_PERIOD 5 MODEL.BONE 'swin_tiny' INPUT.BATCH_SIZE_TRAIN 3 MODEL.SEMANTIC_WEIGHT 0.6

#test/PRW_cascade
#PRW-0.4/0.4 GPU=3 CUDA_VISIBLE_DEVICES=0
#PRW-0.5/0.5 GPU=2 CUDA_VISIBLE_DEVICES=2
#PRW-0.6/0.6 GPU=3 CUDA_VISIBLE_DEVICES=3
#PRW-0.5/0.6/0.7 GPU=4 CUDA_VISIBLE_DEVICES=0
#PRW-0.4/0.5/0.6 GPU=4 CUDA_VISIBLE_DEVICES=0
#PRW-swinCAS GPU=3 CUDA_VISIBLE_DEVICES=0无损失
#PRW-resnetCAS GPU=3 CUDA_VISIBLE_DEVICES=3
#PRW-CAS_CBGM GPU=2 CUDA_VISIBLE_DEVICES=1
#PRW-CAS_USE_GT GPU=4 CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=2 python train_cascade.py --cfg configs/prw.yaml --resume --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_tiny_tea.pth OUTPUT_DIR './results/prw/swin_tiny_3.1' SOLVER.BASE_LR 0.0002 EVAL_PERIOD 5 MODEL.BONE 'swin_tiny' INPUT.BATCH_SIZE_TRAIN 3 MODEL.SEMANTIC_WEIGHT 0.6
#test/SYSU
#SYSU-0.5/0.6/0.7 GPU=4 CUDA_VISIBLE_DEVICES=1X
#SYSU-0.3/0.3/0.3 GPU=2 CUDA_VISIBLE_DEVICES=1X
#SYSU-0.5/0.5/0.5 GPU=3 CUDA_VISIBLE_DEVICES=2X
#SYSU-0.4/0.4/0.4 GPU=3 CUDA_VISIBLE_DEVICES=3X
#SYSU-0.4/0.5/0.6 GPU=3 CUDA_VISIBLE_DEVICES=0
#SYSU-0.6/0.6/0.6 GPU=4 CUDA_VISIBLE_DEVICES=0
#SYSU-0.5/0.6/0.6 GPU=4 CUDA_VISIBLE_DEVICES=1
#SYSU-0.3/0.4/0.5 GPU=2 CUDA_VISIBLE_DEVICES=1
#SYSU-0.6/0.7/0.8 GPU=3 CUDA_VISIBLE_DEVICES=3X
#SYSU-0.5/0.6/0.7_0.04 GPU=3 CUDA_VISIBLE_DEVICES=2X
#SYSU-resnetCAS GPU=1 CUDA_VISIBLE_DEVICES=2 c10X
#SYSU-CAS_CBGM GPU=4 CUDA_VISIBLE_DEVICES=1 c40
#CUDA_VISIBLE_DEVICES=1 python train_cascade.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_tiny_tea.pth OUTPUT_DIR './results/cuhk_sysu/swin_tiny_CAS_CBGM' SOLVER.BASE_LR 0.0002 EVAL_PERIOD 5 MODEL.BONE 'swin_tiny' INPUT.BATCH_SIZE_TRAIN 2 MODEL.SEMANTIC_WEIGHT 0.6
#test/cascade
#CUDA_VISIBLE_DEVICES=1 python train_cascade.py --cfg configs/prw.yaml OUTPUT_DIR './results/prw/swin_tiny_without' SOLVER.BASE_LR 0.0002 EVAL_PERIOD 5 MODEL.BONE 'swin_tiny' INPUT.BATCH_SIZE_TRAIN 3 MODEL.SEMANTIC_WEIGHT 0.6
#epoch=20/18,保留5次
#PRW-arcfaceloss0.02+1 GPU=1 device=0 csj1 SwinTiny_tea batch=3 lr=0.0003（10：00）
#PRW-arcfaceloss0.03+1 GPU=1 device=1 csj2 SwinTiny_tea batch=3 lr=0.0003（10：10）
#PRW-arcfaceloss0.04+1 GPU=1 device=3 csj4 SwinTiny_tea batch=3 lr=0.0003（10：10）
#PRW-arcfaceloss0.009+1 GPU=2 device=3 csj1 SwinTiny_tea batch=3 lr=0.0003（10：10）
#CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/prw.yaml --resume --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_tiny_tea.pth OUTPUT_DIR './results/prw/oim_0.66' SOLVER.BASE_LR 0.0003 EVAL_PERIOD 5 MODEL.BONE 'swin_tiny' INPUT.BATCH_SIZE_TRAIN 3 MODEL.SEMANTIC_WEIGHT 0.66
#SYSU-arcfaceloss0.02+1 GPU=1 device=0 csj1 SwinTiny_tea batch=2 lr=0.0002（10：00）
#SYSU-arcfaceloss0.03+1 GPU=1 device=1 csj2 SwinTiny_tea batch=2 lr=0.0002（10：00）
#SYSU-arcfaceloss0.04+1 GPU=1 device=2 csj3 SwinTiny_tea batch=2 lr=0.0002（10：00）
#SYSU-arcfaceloss0.06+1 GPU=1 device=3 csj0 SwinTiny_tea batch=2 lr=0.0002（10：00）
#SYSU-arcfaceloss0.07+1 GPU=2 device=0 csj1 SwinTiny_tea batch=2 lr=0.0002（10：00）
#SYSU-arcfaceloss0.09+1 GPU=2 device=3 csj2 SwinTiny_tea batch=2 lr=0.0002（10：00）
#CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/cuhk_sysu.yaml --resume --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/configs/swin_tiny_tea.pth OUTPUT_DIR './results/cuhk_sysu/swin_tiny_0.3' SOLVER.BASE_LR 0.0002 EVAL_PERIOD 5 MODEL.BONE 'swin_tiny' INPUT.BATCH_SIZE_TRAIN 2 MODEL.SEMANTIC_WEIGHT 0.3
#demo
#CUDA_VISIBLE_DEVICES=3 python visual1.py --cfg /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/results/prw/swin_tiny_CAS_0.5_0.6_0.7/config.yaml --ckpt /public/home/G19830015/Group/CSJ/projects/NEW_SOLIDER/results/prw/swin_tiny_CAS_0.5_0.6_0.7/epoch_18.pth