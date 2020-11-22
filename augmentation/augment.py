import random

import cv2
from matplotlib import pyplot as plt
import glob
import albumentations as A
import os
import csv
from tqdm import tqdm
def csvread(fn):
    with open(fn, 'r') as csvfile:
        list_arr = []
        reader = csv.reader(csvfile, delimiter=' ')

        for row in reader:
            list_arr.append([int(i) for i in row])
    return list_arr

DATA_TRANSFORMS = [
    A.Compose(
        [A.HorizontalFlip(p=0.5)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    ), 
    A.Compose(
        [A.VerticalFlip(p=0.5)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    ),
    A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    ],
    bbox_params=A.BboxParams(format='pascal_voc',min_area=1000, label_fields=['category_ids']),
    ),
    A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    ),
    A.Compose([
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
    ],
    bbox_params=A.BboxParams(format='pascal_voc',min_area=1000, label_fields=['category_ids']),
    ),
    A.Compose(
        [A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.RGBShift(r_shift_limit=50, g_shift_limit=80, b_shift_limit=30, p=0.3)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    ),
    A.Compose(
        [A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.RGBShift(r_shift_limit=50, g_shift_limit=80, b_shift_limit=30, p=0.3)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    ),
    A.Compose(
        [A.Transpose(p=0.5),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.4)],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
    )
]

def augment_one_image(imageFile, txtFile, save_folder_imgs, save_folder_labels):
    img_name = imageFile.split("/")[-1][:-4]
    image = cv2.imread(imageFile)
    bboxes = csvread(txtFile)
    category_ids = [0] * len(bboxes)
    for i in range(len(DATA_TRANSFORMS)):
        transform = DATA_TRANSFORMS[i]
        try:
            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            # print(transformed['bboxes'])
            cv2.imwrite("{}/{}_augmented_{}.png".format(save_folder_imgs, img_name, i), transformed['image'])
            f= open("{}/{}_augmented_{}.txt".format(save_folder_labels, img_name, i),"w+")
            for bbox in transformed['bboxes']:
                f.write(" ".join([str(i) for i in bbox])+"\n")
            f.close()
        except:
            print("error in {} of transform {}".format(imageFile, i))
    # print("Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augmentating Image ....")
    parser.add_argument("--save_folder_imgs", type=str, default="save_folder_imgs", help="Path to dataset consist of 'train' and 'val'")
    parser.add_argument("--save_folder_labels", type=str, default="save_folder_labels", help="Model name")
    # parser.add_argument("--widen_factor", type=int, default=1, help="Factor of model size")
    # parser.add_argument("--logs", type = str, required=True, help="Path saved model")
    # parser.add_argument("--num_epochs", type=int, default=30, help="num epoch")
    # parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    # parser.add_argument("--init_lr", type=float, default=0.002, help="Starting learning rate")
    # parser.add_argument("--num_age_classes", type=int, default=100, help="Number of age classes")
    # parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
    # parser.add_argument("--num_workers", type=int, default=8, help="Number of worker process data")
    args = parser.parse_args()


    save_aug_imgs_path = "./augmentation/{}".format(args.save_folder_imgs)
    save_aug_labels_path = "./augmentation/{}".format(args.save_folder_labels)
    if not os.path.exists(save_aug_imgs_path):
        os.makedirs(save_aug_imgs_path)
    if not os.path.exists(save_aug_labels_path):
        os.makedirs(save_aug_labels_path)
    img_lst = glob.glob(os.path.join(".", "data","images","*"))
    # print(csvread("./data/labels/Img_RSKA003603_5_r1280_c3712.txt"))
    for imageFile in tqdm(img_lst):
        # print("process {}".format(imageFile))
        txtFile = "./data/labels/{}.txt".format(imageFile.split("/")[-1][:-4])
        augment_one_image(imageFile, txtFile, save_aug_imgs_path, save_aug_labels_path)

    # import ipdb; ipdb.set_trace()