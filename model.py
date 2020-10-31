import torchvision
from torch import nn

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from config import cfg 


class ModelResnet101FasterRCNN(FasterRCNN):
    def __init__(self, data_conf, model_conf):

        print("Creating model backbone with " + str(model_conf["hyperParameters"]["net"]))
        backbone_nn = torchvision.models.__dict__[model_conf["hyperParameters"]["net"]](pretrained=True)

        if model_conf["hyperParameters"]["freeze_pretrained_gradients"]:
            print("Using backbone as fixed feature extractor")
            modules = list(backbone_nn.children())[:-1]  # delete the last fc layer.
            backbone_nn = nn.Sequential(*modules)

            # FasterRCNN needs to know the number of
            # output channels in a backbone. For resnet101, it's 2048
            for param in backbone_nn.parameters():
                param.requires_grad = False
            backbone_nn.out_channels = model_conf["hyperParameters"]["net_out_channels"]
        else:
            print("Using fine-tuning of the model")
            modules = list(backbone_nn.children())[:-1]  # delete the last fc layer.
            backbone_nn = nn.Sequential(*modules)

            # FasterRCNN needs to know the number of
            # output channels in a backbone. For resnet101, it's 2048
            for param in backbone_nn.parameters():
                param.requires_grad = True
            backbone_nn.out_channels = model_conf["hyperParameters"]["net_out_channels"]
        #

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios

        anchor_ratios = cfg.ANCHOR_RATIOS
        anchor_sizes = cfg.ANCHOR_SCALES

        print("anchor_ratios = " + str(anchor_ratios))
        print("anchor_sizes = " + str(anchor_sizes))

        anchor_generator = AnchorGenerator(sizes=(anchor_sizes,),
                                           aspect_ratios=(anchor_ratios,))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.

        rpn_pooling_size = cfg.POOLING_SIZE

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"],
                                                        output_size=rpn_pooling_size,
                                                        sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        super().__init__(backbone=backbone_nn,
                         num_classes = 3,
                         image_mean = cfg.MODEL.IMAGE_MEAN,
                         image_std = cfg.MODEL.IMAGE_STD,
                         rpn_anchor_generator = anchor_generator,
                         box_roi_pool = roi_pooler,
                         rpn_pre_nms_top_n_train = cfg.MODEL.RPN_PRE_NMS_TOP_N_TRAIN,
                         rpn_post_nms_top_n_train = cfg.MODEL.RPN_POST_NMS_TOP_N_TRAIN,
                         rpn_nms_thresh = cfg.MODEL.RPN_NMS_THRESH,
                         min_size = cfg.MODEL.MIN_SIZE,
                         max_size = cfg.MODEL.MIN_SIZE)