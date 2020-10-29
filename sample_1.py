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
import glob
from PIL import Image
from torchvision import datasets, transforms

def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(currentPath, 'groundtruths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.GroundTruth,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath, 'detections')
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.Detected,
                confidence,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes

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
    # import ipdb; ipdb.set_trace()
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold] # Get list of index with score greater than threshold.
    if pred_t is None:
        return [], [], []
    else:
        pred_t = pred_t[-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
        # print(len(pred_boxes))
        # print(pred)
        return pred_boxes, pred_class, pred_score




if __name__ == "__main__":
    dataset_test = SatelliteDataset('validating_data/', get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    model = init_model()
    imgs = glob.glob(os.path.join('validating_data/', "000003085.jpg"))
    # rand_img = random.sample(imgs, 1) 
    # import ipdb; ipdb.set_trace()
    myBoundingBoxes = BoundingBoxes()
    for idx in range(len(dataset_test)):
        img, target = dataset_test.__getitem__(idx)
        for i in range(len(target['boxes'])):
            gt_boundingBox = BoundingBox(
                imageName = target["img_name"],
                classId = CLASS_NAMES[target['labels'][i].item()],
                x = target['boxes'][i][0].item() ,y = target['boxes'][i][1].item(),
                w = target['boxes'][i][2].item(), h=target['boxes'][i][3].item(),
                typeCoordinates = CoordinatesType.Absolute,
                bbType=BBType.GroundTruth,
                format=BBFormat.XYX2Y2
            )
            myBoundingBoxes.addBoundingBox(gt_boundingBox)
        image_path = glob.glob(os.path.join('validating_data/', "{}.jpg".format(target["img_name"])))[0]
        print("predict {}".format(target["img_name"]))
        #failed at predict 000003085
        pred_boxes, pred_class, pred_score = get_prediction(model, image_path, 0.5) # Get predictions
        if pred_boxes is None:
            continue
        # import ipdb; ipdb.set_trace()
        for idx_detect in range(len(pred_boxes)):
            detected_boundingBox = BoundingBox(
                imageName = target["img_name"],
                classId = pred_class[idx_detect],
                classConfidence = pred_score[idx_detect],
                x = pred_boxes[idx_detect][0][0] ,y = pred_boxes[idx_detect][0][1],
                w = pred_boxes[idx_detect][1][0], h=pred_boxes[idx_detect][1][1],
                typeCoordinates = CoordinatesType.Absolute,
                bbType=BBType.Detected,
                format=BBFormat.XYX2Y2
            )
            myBoundingBoxes.addBoundingBox(detected_boundingBox)         
    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()
        # Get metrics with PASCAL VOC metrics
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        myBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=0.3,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('%s: %f' % (c, average_precision))   
    


