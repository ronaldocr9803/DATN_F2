import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import random
import csv

class SatelliteDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.base_path = os.getcwd()
        self.data_dir = data_dir
        self.transforms = transforms
        # load the annotations file, it also contain information of image names
        # load annotations
        self.lst_images = os.listdir(os.path.join(self.base_path , self.data_dir)) #'/training_data/training_data/images': data_dir
        # import ipdb; ipdb.set_trace()
        # annotations1 = json.load(open(os.path.join(data_dir, "via_region_data.json")))
        # print(annotations1)
        # self.annotations = list(annotations1.values())  # don't need the dict keys
        # self.annotations = [a for a in annotations if a['regions']]
        

    def __getitem__(self, idx):

        img_name = self.lst_images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        # import ipdb; ipdb.set_trace()
        img = Image.open(img_path).convert("RGB")

        anno_dir = self.base_path + '/data/training_data/labels'
        anno_path = os.path.join(anno_dir, img_name.split('.')[0] + '.xml')
        annotations = []
        tree = ET.parse(anno_path)  
        root = tree.getroot()
        boxes = []
        labels = []
        for element in root:
            if element.tag == 'object':
                obj_name = None
                coords = []
                for subelem in element:
                    if subelem.tag == 'name':
                        obj_name = subelem.text
                        labels.append(int(obj_name))
                    if subelem.tag == 'bndbox':
                        for subsubelem in subelem:
                            coords.append(float(subsubelem.text))
                        boxes.append(coords)
                        # convert everything into a torch.Tensor
                        # coords = torch.as_tensor(coords, dtype=torch.float32)
                        # import ipdb; ipdb.set_trace()
                        xMin = coords[0]
                        yMin = coords[1]
                        xMax = coords[2]
                        yMax = coords[3]
                        # coords = subelem.text
                                    # truncate any bounding box corrdinates that fall outside
                        # image boundaries
                xMin = max(0, xMin)
                yMin = max(0, yMin)
                xMax = max(0, xMax)
                yMax = max(0, yMax)


                # ignore bounding boxes where minimum values are larger than
                # max values and vice-versa (annotation errors)
                if xMin >= xMax or yMin >= yMax:
                    continue
                elif xMax <= xMin or yMax <= yMin:
                    continue
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.tensor(labels, dtype=torch.int64)
        # labels = np.asarray(labels)
        # labels = torch.from_numpy(labels.astype('long'))
        labels = torch.from_numpy(np.asarray(labels))
        image_id = torch.tensor([idx])
        # import ipdb; ipdb.set_trace()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # import ipdb; ipdb.set_trace()
        return img, target

    def __len__(self):
        return len(self.lst_images)

if __name__ == "__main__":
    a= SatelliteDataset('data/training_data/images')
    a.__getitem__(20)
    # import ipdb; ipdb.set_trace()
 