# select subsample of N for initial training
import os
import random
from shutil import copyfile, move
from tqdm import tqdm
import glob
import shutil

# import ipdb; ipdb.set_trace()
if not os.path.exists('./data/training_data'):
	os.makedirs('./data/training_data')
if not os.path.exists('./data/validating_data'):
	os.makedirs('./data/validating_data')
if not os.path.exists('./data/testing_data'):
	os.makedirs('./data/testing_data')
# determine number for the sample
NUM = 150
# create directory for the sample
base_dir = os.getcwd()
img_path_lst = glob.glob(os.path.join(".","data","images","*"))
num_train_sample = int(round(len(img_path_lst) * 0.7))
img_train_path_lst = random.sample(img_path_lst, num_train_sample) 
# img_train_path = base_dir + '/data/training_data/images/'
val_test_path_lst = [x for x in img_path_lst if x not in img_train_path_lst]
#val folder
img_val_path_lst = random.sample(val_test_path_lst, 1500)
#test folder
img_test_path_lst = [x for x in val_test_path_lst if x not in img_val_path_lst] 

for i in tqdm(range(len(img_train_path_lst))):
   img_name = img_train_path_lst[i].split("/")[-1]
   des_path = os.path.join(".","data","training_data",img_name)
   shutil.copy(img_train_path_lst[i], des_path)

for i in tqdm(range(len(img_val_path_lst))):
   img_name = img_val_path_lst[i].split("/")[-1]
   des_path = os.path.join(".","data","validating_data",img_name)
   shutil.copy(img_val_path_lst[i], des_path)

for i in tqdm(range(len(img_test_path_lst))):
   img_name = img_test_path_lst[i].split("/")[-1]
   des_path = os.path.join(".","data","testing_data",img_name)
   shutil.copy(img_test_path_lst[i], des_path)



# sub_dir = base_dir + '/data/training_data/images/'
# # labels = base_dir + '/data/training_data/labels/'
# image_dir = base_dir + '/data/training_data/'
# image_paths = os.listdir(image_dir)
# image_paths = [a for a in image_paths if a[-4:] == ".jpg"]
# # import ipdb; ipdb.set_trace()
# for i in tqdm(image_paths):
#    move(image_dir + i, sub_dir + i )
# import ipdb; ipdb.set_trace()
# # randomly select subsample
# num_val_sample = int(round(len(image_paths) * 0.3))
# # import ipdb; ipdb.set_trace()
# random_NUM = random.sample(image_paths, 300) 
# # copy subsample into subsample directory
# from tqdm import tqdm
# for i in tqdm(random_NUM):
#    move(image_dir + i, sub_dir + i )
# from tqdm import tqdm
# for i in tqdm(os.listdir(sub_dir)):
#    move(sub_dir + i, image_dir + i )