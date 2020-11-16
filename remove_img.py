from PIL import Image
import csv
import os
from tqdm import tqdm
import cv2
import glob
from shutil import copyfile, move

lst_empty_txt = [a for a in glob.glob(os.path.join(".", "data", "labels","*")) if os.stat(a).st_size==0]
des_labels_empty = "./data/no_bb_labels"
src_img = "./data/images"
des_img_empty = "./data/no_bb_images"

for i in tqdm(range(len(lst_empty_txt))):
   img_name = lst_empty_txt[i].split("/")[-1][:-4]
   des_path = os.path.join(".","data","training_data",img_name)
   #move image 
   move(os.path.join(src_img, "{}.png".format(img_name)), os.path.join(des_img_empty,"{}.png".format(img_name)))
   #move labels a
   move(lst_empty_txt[i], os.path.join(des_labels_empty,"{}.txt".format(img_name)))