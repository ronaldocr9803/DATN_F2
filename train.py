from dataset import RasterDataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from utils import save_checkpoint
from model import fasterrcnn_resnet101_fpn
from config import cfg

def build_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    tb = SummaryWriter()
    # import ipdb; ipdb.set_trace()
    # model = fasterrcnn_resnet101_fpn(pretrained = True)
    # use our dataset and defined transformations
    dataset = RasterDataset('data/training_data/', get_transform(train=True))
    dataset_test = RasterDataset('data/validating_data/', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2

    # model = fasterrcnn_resnet101_fpn(pretrained = False)
    model =  build_model(num_classes)

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.MODEL.LEARNING_RATE,
                                momentum = cfg.MODEL.MOMENTUM
                                ,weight_decay = cfg.MODEL.WEIGHT_DECAY)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.MODEL.STEPSIZE,
                                                   gamma=cfg.MODEL.GAMMA)
    
    # number of epochs
    num_epochs = 15
    #start training
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # tb.add_scalar('Loss', total_loss, epoch)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, epoch, device=device)
        # save_checkpoint(epoch, model, optimizer)
        
    # save trained model for inference    
    # torch.save(model, './output/faster-rcnn-Satellite.pt')