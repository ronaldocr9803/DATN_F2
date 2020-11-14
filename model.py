import torchvision
from torch import nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from config import cfg 
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=2, pretrained_backbone=True,
                             trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet101', pretrained_backbone)
    model = ModelResnet101FasterRCNN(backbone, num_classes, **kwargs)
    return model

class ModelResnet101FasterRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes):

        anchor_ratios = cfg.ANCHOR_RATIOS
        anchor_sizes = cfg.ANCHOR_SCALES

        print("anchor_ratios = " + str(anchor_ratios))
        print("anchor_sizes = " + str(anchor_sizes))

        anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                           aspect_ratios=anchor_ratios)
        # if your backbone returns a Tensor, featmap_names is expected to be [0]. 

        rpn_pooling_size = cfg.POOLING_SIZE #7

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                        output_size=rpn_pooling_size,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        super().__init__(
                         backbone = backbone,
                         num_classes = num_classes
                         ,image_mean = cfg.MODEL.IMAGE_MEAN,
                         image_std = cfg.MODEL.IMAGE_STD,
                         rpn_anchor_generator = anchor_generator,
                         box_roi_pool = roi_pooler,
                         rpn_pre_nms_top_n_train = cfg.MODEL.RPN_PRE_NMS_TOP_N_TRAIN,
                         rpn_post_nms_top_n_train = cfg.MODEL.RPN_POST_NMS_TOP_N_TRAIN,
                         rpn_nms_thresh = cfg.MODEL.RPN_NMS_THRESH,
                         min_size = cfg.MODEL.MIN_SIZE,
                         max_size = cfg.MODEL.MAX_SIZE
                         )