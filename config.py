import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.MODEL = edict()

# Initial learning rate
__C.MODEL.LEARNING_RATE = 0.005

# Momentum
__C.MODEL.MOMENTUM = 0.9

# Weight decay, for regularization
__C.MODEL.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.MODEL.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.MODEL.STEPSIZE = 3

# # Iteration intervals for showing the loss during training, on command line interface
# __C.TRAIN.DISPLAY = 10

__C.MODEL.MIN_SIZE = 800
__C.MODEL.MAX_SIZE = 1333
__C.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
__C.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]

# Box parameter
#during inference, only return proposals with a classification score greater than box_score_thresh
__C.MODEL.BOX_SCORE_THRESH = 0.05
# NMS threshold for the prediction head
__C.MODEL.BOX_NMS_THRESH = 0.5
#maximum number of detections per image, for all classes.
__C.MODEL.BOX_DETECTIONS_PER_IMG = 100
#minimum IoU between the proposals and the GT box so that they can be considered as positive during training of the classification head
__C.MODEL.BOX_FG_IOU_THRESH = 0.5
#maximum IoU between the proposals and the GT box so that they can be considered as negative during training of the classification head
__C.MODEL.BOX_BG_IOU_THRESH = 0.5
#number of proposals that are sampled during training of the classification head
__C.MODEL.BOX_BATCH_SIZE_PER_IMAGE = 512
#proportion of positive proposals in a mini-batch during training of the classification head
__C.MODEL.BOX_POSITIVE_FRACTION = 0.25

# __C.TRAIN.BBOX_REG_WEIGHTS =

# Use RPN to detect objects
# IOU >= thresh: positive example. minimum IoU between the anchor and the GT box so that they can be
# considered as positive during training of the RPN.
__C.MODEL.RPN_FG_IOU_THRESH = 0.7
# IOU < thresh: negative example. maximum IoU between the anchor and the GT box so that they can be
#considered as negative during training of the RPN.
__C.MODEL.RPN_BG_IOU_THRESH = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
# __C.TRAIN.RPN_CLOBBER_POSITIVES = False
# proportion of positive anchors in a mini-batch during training of the RPN
__C.MODEL.RPN_POSITIVE_FRACTION = 0.5
# number of anchors that are sampled during training of the RPN for computing the loss
__C.TRAIN.RPN_BATCH_SIZE_PER_IMAGE = 256
# NMS threshold used on RPN proposals
__C.MODEL.RPN_NMS_THRESH = 0.5
# Number of top scoring boxes to keep before apply NMS to RPN proposals 
#number of proposals to keep before applying NMS during training
__C.MODEL.RPN_PRE_NMS_TOP_N_TRAIN = 2000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
#number of proposals to keep after applying NMS during training
__C.MODEL.RPN_POST_NMS_TOP_N_TRAIN = 2000
# number of proposals to keep before applying NMS during testing
__C.MODEL.RPN_PRE_NMS_TOP_N_TEST = 1000
#number of proposals to keep after applying NMS during testing
__C.MODEL.RPN_POST_NMS_TOP_N_TEST = 1000



# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Anchor scales for RPN      #[8,16,32]
__C.ANCHOR_SCALES = (32, 64, 128, 256, 512)     

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = (0.5, 1.0, 2.0) #[0.5,1,2]