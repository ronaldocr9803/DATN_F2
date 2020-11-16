#Img_RSKA003603_0_r000_c128.txt
import os
import glob
from tqdm import tqdm
import cv2 

txt_path_lst = glob.glob(os.path.join(".", "data","labels","*"))
for i in tqdm(range(len(txt_path_lst))):

    f = open(txt_path_lst[i] ,"r")
    # print(len(f.readlines()))
    lst = f.readlines()
    # import ipdb; ipdb.set_trace()
    # for j in range(len(lst)):
    #     fa = lst[j].split(" ")
    #     if fa[1] == fa[3]:
    #         print(txt_path_lst[i])

    if len(lst) == 0:
        continue
    img_name = txt_path_lst[i].split("/")[-1][:-4]
    img = cv2.imread(os.path.join(".","data","images","{}.png".format(img_name)))
    img_height, img_width, _ = img.shape
    # import ipdb; ipdb.set_trace()
    for j in range(len(lst)):
        x1, y1, x2, y2 = [int(k) for k in lst[j].strip().split(" ")]
        x_center = (float(x1+x2)/2)/img_width
        y_center = (float(y1+y2)/2)/img_height
        width = float(x2-x1)/img_width
        height = float(y2-y1)/img_height
        # fa = " ".join(lst[j][1:].split(" "))
        lst[j]= "0 {} {} {} {}\n".format(x_center, y_center, width, height)
    a_file = open(txt_path_lst[i], "w")
    a_file.writelines(lst)
    a_file.close()


