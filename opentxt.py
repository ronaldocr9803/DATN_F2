#Img_RSKA003603_0_r000_c128.txt
import os
import glob
from tqdm import tqdm

txt_path_lst = glob.glob(os.path.join(".", "data","labels","*"))
for i in tqdm(range(len(txt_path_lst))):

    f = open(txt_path_lst[i] ,"r")
    # print(len(f.readlines()))
    lst = f.readlines()
    # import ipdb; ipdb.set_trace()
    for j in range(len(lst)):
        fa = lst[j].split(" ")
        if fa[1] == fa[3]:
            print(txt_path_lst[i])

    # if len(lst) == 0:
    #     continue
    # for j in range(len(lst)):
    #     fa = " ".join(lst[j].split(" "))
    #     lst[j]= "0 {}".format(fa)
    # a_file = open(txt_path_lst[i], "w")
    # a_file.writelines(lst)
    # a_file.close()


