import os 
import glob
from tqdm import tqdm
import shutil

# if not os.path.exists('./raster'):
# 	os.makedirs('./raster')
# os.makedirs('./images')
# os.makedirs('./labels')
# os.makedirs('./raster')
# os.makedirs('./raster')
def make_data_folder():
    images = [
        'train',
        'val'
    ]

    labels = [
        'train',
        'val'
    ]
    main_dir = [images, labels]			# Loading the list of sub-directories
    root_dir = 'raster'
    main_dir_names = ['images', 'labels']
    for i in range(0, len(main_dir)):
        for j in range(0,len(main_dir[i])):
            dirName = str(root_dir) + '/' + str(main_dir_names[i]) +'/' + str(main_dir[i][j])
            
            try:
                # Create target Directory
                os.makedirs(dirName)
                print("Directory " , dirName ,  " Created ") 
            except FileExistsError:
                print("Directory " , dirName ,  " already exists")        
            
            # Create target Directory if don't exist
            if not os.path.exists(dirName):
                os.makedirs(dirName)
                print("Directory " , dirName ,  " Created ")
            else:    
                print("Directory " , dirName ,  " already exists")

if __name__ == "__main__":
    if os.path.exists('./raster'):
        shutil.rmtree('./raster')
    make_data_folder()
    img_train_path_lst = glob.glob(os.path.join(".","data","training_data","*"))
    img_val_path_lst = glob.glob(os.path.join(".","data","validating_data","*"))

    # images
    for i in tqdm(range(len(img_train_path_lst))):
    #    import ipdb; ipdb.set_trace()
       img_name = img_train_path_lst[i].split("/")[-1]
       des_path_train_img = os.path.join(".","raster","images","train",img_name)
       shutil.copy(img_train_path_lst[i], des_path_train_img)
       src_path_txt = os.path.join(".","data","labels","{}.txt".format(img_name[:-4]))
       des_path_train_txt = os.path.join(".","raster","labels","train","{}.txt".format(img_name[:-4]))
       shutil.copy(src_path_txt, des_path_train_txt)


    for j in tqdm(range(len(img_val_path_lst))):
       img_name = img_val_path_lst[j].split("/")[-1]
       des_path_val_img = os.path.join(".","raster","images","val",img_name)
       shutil.copy(img_val_path_lst[j], des_path_val_img)
       src_path_txt = os.path.join(".","data","labels","{}.txt".format(img_name[:-4]))
       des_path_val_txt = os.path.join(".","raster","labels","val","{}.txt".format(img_name[:-4]))
       shutil.copy(src_path_txt, des_path_val_txt)


