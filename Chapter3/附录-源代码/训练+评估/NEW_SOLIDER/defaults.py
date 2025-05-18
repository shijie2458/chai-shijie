from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
_C.INPUT = CN()
# _C.INPUT.DATASET = "CUHK-SYSU"
# _C.INPUT.DATA_ROOT = "data/CUHK-SYSU"
_C.INPUT.DATASET = "PRW"
_C.INPUT.DATA_ROOT = "/public/home/G19830015/Group/CSJ/data/PRW"
# Size of the smallest side of the image
_C.INPUT.MIN_SIZE = 900
# Maximum size of the side of the image
_C.INPUT.MAX_SIZE = 1500

# TODO: support aspect ratio grouping
# Whether to use aspect ratio grouping for saving GPU memory
# _C.INPUT.ASPECT_RATIO_GROUPING_TRAIN = False

# Number of images per batch
_C.INPUT.BATCH_SIZE_TRAIN = 2
_C.INPUT.BATCH_SIZE_TEST = 1

# Number of data loading threads
_C.INPUT.NUM_WORKERS_TRAIN = 5
_C.INPUT.NUM_WORKERS_TEST = 1

# Image augmentation
_C.INPUT.IMAGE_CUTOUT = False
_C.INPUT.IMAGE_ERASE = False
_C.INPUT.IMAGE_MIXUP = False

# -------------------------------------------------------- #
#                          Solver求解器                     #
# -------------------------------------------------------- #
_C.SOLVER = CN()
# 默认18
_C.SOLVER.MAX_EPOCHS = 30

# Learning rate settings
_C.SOLVER.BASE_LR = 0.003

# TODO: add config option WARMUP_EPOCHS
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
# _C.SOLVER.WARMUP_EPOCHS = 1

# The epoch milestones to decrease the learning rate by GAMMA
_C.SOLVER.LR_DECAY_MILESTONES = [16]
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.SGD_MOMENTUM = 0.9

# Loss weight of RPN regression
_C.SOLVER.LW_RPN_REG = 1
# Loss weight of RPN classification
_C.SOLVER.LW_RPN_CLS = 1

# Loss weight of proposal regression
_C.SOLVER.LW_PROPOSAL_REG = 10
# Loss weight of proposal classification
_C.SOLVER.LW_PROPOSAL_CLS = 1
# Loss weight of box regression
_C.SOLVER.LW_BOX_REG = 1
# Loss weight of box classification
_C.SOLVER.LW_BOX_CLS = 1
# Loss weight of box OIM (i.e. Online Instance Matching)
_C.SOLVER.LW_BOX_REID = 0.5         # 1或0.5
# ADD
_C.SOLVER.LW_RCNN_REG_3RD = 1
_C.SOLVER.LW_RCNN_CLS_3RD = 1
_C.SOLVER.LW_RCNN_REID_3RD = 0.5  # 1或0.5
# Loss weight of box reid, softmax loss
_C.SOLVER.LW_RCNN_SOFTMAX_2ND = 0.5
_C.SOLVER.LW_RCNN_SOFTMAX_3RD = 0.5

# Set to negative value to disable gradient clipping渐变剪裁
_C.SOLVER.CLIP_GRADIENTS = 10.0
# -------------------------------------------------------- #
#                            FPN                           #
# -------------------------------------------------------- #
# _C.IN_CHANNELS_LIST = [96, 96 * 2, 96 * 2 * 2, 96 * 2 * 2 * 2]
# _C.OUT_CHANNELS = 256

# -------------------------------------------------------- #
#                            RPN                           #
# -------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.BONE = "swin_tiny"
_C.MODEL.SEMANTIC_WEIGHT = 1.0

_C.MODEL.RPN = CN()
# NMS threshold used on RoIs
_C.MODEL.RPN.NMS_THRESH = 0.7
# Number of anchors per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_TRAIN = 256
# Target fraction of foreground examples per RPN minibatch
_C.MODEL.RPN.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.RPN.POS_THRESH_TRAIN = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.RPN.NEG_THRESH_TRAIN = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.MODEL.RPN.PRE_NMS_TOPN_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPN_TEST = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOPN_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPN_TEST = 300

# -------------------------------------------------------- #
#                         RoI head                         #
# -------------------------------------------------------- #
_C.MODEL.ROI_HEAD = CN()
# Whether to use bn neck (i.e. batch normalization after linear)
_C.MODEL.ROI_HEAD.BN_NECK = True
# Number of RoIs per image used to train RoI head
_C.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN = 128
# Target fraction of foreground examples per RoI minibatch
_C.MODEL.ROI_HEAD.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN = 0.5
# Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN = 0.5
# Minimum score threshold
_C.MODEL.ROI_HEAD.SCORE_THRESH_TEST = 0.5
# NMS threshold used on boxes
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST = 0.4
# Maximum number of detected objects
_C.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST = 300
# -------------------------------------------------------- #
#                           阈值                           #
# -------------------------------------------------------- #
_C.MODEL.ROI_HEAD.USE_DIFF_THRESH = True
# Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN = 0.5
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND = 0.6
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN_3RD = 0.7
# Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN = 0.5
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND = 0.6
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_3RD = 0.7
# Minimum score threshold
_C.MODEL.ROI_HEAD.SCORE_THRESH_TEST = 0.5
# NMS threshold used on boxes
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST = 0.4
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST_1ST = 0.4
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND = 0.4
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST_3RD = 0.5

####
_C.MODEL.EMBEDDING_DIM = 256
# -------------------------------------------------------- #
#                           Loss                           #
# -------------------------------------------------------- #
_C.MODEL.LOSS = CN()
# Size of the lookup table in OIM
_C.MODEL.LOSS.LUT_SIZE = 5532
# Size of the circular queue in OIM
_C.MODEL.LOSS.CQ_SIZE = 5000
_C.MODEL.LOSS.OIM_MOMENTUM = 0.5
_C.MODEL.LOSS.OIM_SCALAR = 30.0
_C.MODEL.LOSS.USE_SOFTMAX = True
# Arcface 调参
_C.MODEL.LOSS.ARCFACE_LOSS_WEIGHT = 0.02                # PRW调参
# _C.MODEL.LOSS.ARCFACE_LOSS_WEIGHT = 0.04                # SYSU调参
_C.MODEL.LOSS.COSINE = 0.5                              # 调参默认0.5
# -------------------------------------------------------- #
#                        Evaluation                        #
# -------------------------------------------------------- #
# The period to evaluate the model during training 每1个训练周期结束后都会进行一次评估
_C.EVAL_PERIOD = 1
# Evaluation with GT boxes to verify the upper bound of person search performance
_C.EVAL_USE_GT = False
# Fast evaluation with cached features 使用缓存特征进行快速评估
_C.EVAL_USE_CACHE = False
# Evaluation with Context Bipartite Graph Matching (CBGM) algorithm 使用 CBGM 算法进行性能评估
_C.EVAL_USE_CBGM = False
# Feature used for evaluation
_C.EVAL_FEATURE = 'concat' # 'stage2', 'stage3'

# -------------------------------------------------------- #
#                           Miscs                          #
# -------------------------------------------------------- #
# Save a checkpoint after every this number of epochs
_C.CKPT_PERIOD = 1
# The period (in terms of iterations) to display training losses
_C.DISP_PERIOD = 10
# Whether to use tensorboard for visualization
_C.TF_BOARD = True
# The device loading the model
_C.DEVICE = "cuda"
# _C.DEVICE = "cpu"
# Set seed to negative to fully randomize everything
_C.SEED = 1
# Directory where output files are written
_C.OUTPUT_DIR = "./output"


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()  # 获取默认配置的副本
