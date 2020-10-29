# import _init_paths
# from podm.podm import get_pascal_voc_metrics

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import *
from train import build_model, get_transform
from dataset import SatelliteDataset
import transforms as T
import utils

def init_model():
    model = build_model(3)
    device = torch.device('cpu')
    checkpoint = torch.load("./checkpoint/chkpoint_9.pt", map_location={'cuda:0': 'cpu'}) #read from last checkpoint
    model.load_state_dict(checkpoint['state_dict'])
    model.eval() #evaluation mode
    return model 

CLASS_NAMES = ["__background__", "car","pool"]

def get_prediction(model, img_path, threshold):
  img = Image.open(img_path) # Load the image
  my_transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
  img = my_transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  # pdb.set_trace()
  pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  # print(len(pred_boxes))
  # print(pred)
  return pred_boxes, pred_class




if __name__ == "__main__":
    dataset_test = SatelliteDataset('test_data_images/test_data_images/images/', get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    print(CLASS_NAMES)
