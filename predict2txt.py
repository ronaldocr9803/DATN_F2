from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import *
from train import  get_transform
from dataset import RasterDataset
import transforms as T
import utils
import glob
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
from model import fasterrcnn_resnet101_fpn


def init_model():
    # if model_name == "resnet50":
    #     model = build_model_resnet50fpn(3)
    # elif model_name == "resnet101":
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    model = fasterrcnn_resnet101_fpn()
    device = torch.device(dev)
    # checkpoint = torch.load("./checkpoint/chkpoint_colab_14.pt", map_location={'cuda:0': 'cpu'}) #read from last checkpoint
    checkpoint = torch.load("./checkpoint/chkpoint_colab_14.pt", map_location={'cuda:0': dev}) #read from last checkpoint

    model.load_state_dict(checkpoint['state_dict'])
    model.eval() #evaluation mode
    return model 

CLASS_NAMES = ["__background__", "tree"]

def get_prediction(model, img_path, threshold):
    img = Image.open(img_path) # Load the image
    my_transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
    img = my_transform(img) # Apply the transform to the image
    # import ipdb; ipdb.set_trace()
    pred = model([img]) # Pass the image to the model
    # pdb.set_trace()
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold] # Get list of index with score greater than threshold.
    if not pred_t:
        return [], [], []
    else:
        # pred_t = pred_t[-1]
        # pred_boxes = pred_boxes[:pred_t+1]
        # pred_class = pred_class[:pred_t+1]
        # pred_score = pred_score[:pred_t+1]

        # print(len(pred_boxes))
        # print(pred)
        return pred_boxes, pred_class, pred_score

if __name__ == "__main__":
    dataset_test = RasterDataset('data/validating_data/', get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    model = init_model()
    imgs = glob.glob(os.path.join('validating_data/', "000003085.jpg"))
    myBoundingBoxes = BoundingBoxes()
    if not os.path.exists('./groundtruths'):
        os.makedirs('groundtruths')
    if not os.path.exists('./detections/'):
        os.makedirs('./detections/')
    
    for idx in tqdm(range(len(dataset_test))):
        
        img, target = dataset_test.__getitem__(idx)
        img_name = dataset_test.__getname__(idx)
        with open('./groundtruths/{}.txt'.format(img_name.split(".")[0]), 'w') as f:
            for i in range(len(target['boxes'])):
                f.write("{} {} {} {} {}\n".format(CLASS_NAMES[target['labels'][i].item()],
                                                        target['boxes'][i][0].item(),
                                                        target['boxes'][i][1].item(),
                                                        target['boxes'][i][2].item(),
                                                        target['boxes'][i][3].item()))
        f.close()
        
        image_path = glob.glob(os.path.join('data/validating_data/', "{}".format(img_name)))[0]
        with open('./detections/{}.txt'.format(img_name.split(".")[0]), 'w') as f_predict:
            # logging.info("predict {}".format(img_name))
            #failed at predict 000003085
            pred_boxes, pred_class, pred_score = get_prediction(model, image_path, 0.5) # Get predictions
            if pred_boxes is None:
                # logging.info("Failes at {}".format(img_name))
                f_predict.write("")
                continue
            for idx_detect in range(len(pred_boxes)):
                f_predict.write("{} {} {} {} {} {}\n".format(pred_class[idx_detect],
                                                    pred_score[idx_detect],
                                                    pred_boxes[idx_detect][0][0],
                                                    pred_boxes[idx_detect][0][1],
                                                    pred_boxes[idx_detect][1][0],
                                                    pred_boxes[idx_detect][1][1]))
        f_predict.close()                                                    