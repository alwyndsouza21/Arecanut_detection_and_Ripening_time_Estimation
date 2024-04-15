#imports required 

import torch
import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.patches as patches
from torchvision.ops import nms

def create_detection_model(num_classes:int=2):
  """creates a FRCNN-resnet object detection model"""
  weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
  transforms= weights.transforms()
  model= torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
  in_features=model.roi_heads.box_predictor.cls_score.in_features
  for params in model.parameters():
    params.requires_grad=False
  model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)
  return model,transforms

#function to create efficientnet model for classificatiion..
def create_classification_model():
  
  Weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
  transforms = Weights.transforms()
  model = torchvision.models.efficientnet_v2_l(weights=Weights)
  for param in model.parameters():
      param.requires_grad = False
  #i created the model, and initialized it, thats where i got the exact no.of in_features!
  model.classifier = torch.nn.Sequential(
    torch.nn. Dropout(p=0.4, inplace=True),
     torch.nn. Linear(in_features=1280, out_features=6, bias=True)
   )
  return model,transforms


#creating the models amd then loading the weights and biases 
det_model,det_transforms=create_detection_model()
det_model.load_state_dict(torch.load("E:\Downloads\Arecanut_detection_and_Ripening_time_Estimation-main\Arecanut_detection_and_Ripening_time_Estimation-main\model_state_dicts\Faster_rcnn_model (1).pth",map_location=torch.device("cpu")))

cls_model,cls_transforms=create_classification_model()
cls_model.load_state_dict(torch.load("E:\Downloads\Arecanut_detection_and_Ripening_time_Estimation-main\Arecanut_detection_and_Ripening_time_Estimation-main\model_state_dicts\classification_effnet96_model.pth",map_location=torch.device("cpu")))


#returns tensors of the cropped region 
def crop_and_resize_image(image, bbox, output_size=(224, 224)):
  """
  Crop and resize the ROI from the image using the bounding box coordinates.

  Args:
  - image (PIL.Image): Input image.
  - bbox (list or tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
  - output_size (tuple): Desired output size in the format (height, width).

  Returns:
  - cropped_image_tensor (torch.Tensor): Tensor containing the resized cropped ROI.
  """
  x_min, y_min, x_max, y_max = bbox

  # Crop ROI from the image
  cropped_image = image.crop((x_min, y_min, x_max, y_max))

  # Resize cropped ROI
  transform = T.Compose([
      T.Resize(output_size),
      T.ToTensor(),
  ])

  cropped_image_tensor = transform(cropped_image)

  return cropped_image_tensor


def predict_and_plot_bounding_boxes(model, image_path):
  # Load and transform the image
  image = Image.open(image_path).convert("RGB")
  transform = T.Compose([T.ToTensor()])
  image_tensor = transform(image).unsqueeze(0) # Add batch dimension and move to device

  # Make prediction
  model.eval()
  with torch.no_grad():
    predictions = model(image_tensor)

  # Convert predictions to list of dictionaries
  predicted_boxes = [{
      "boxes": pred["boxes"].cpu().numpy(),
      "labels": pred["labels"].cpu().numpy(),
      "scores": pred["scores"].cpu().numpy(),
  } for pred in predictions]

  box_coordinates = predicted_boxes[0]["boxes"]  # Assuming a single image prediction
  scores = predicted_boxes[0]["scores"]

  # Perform Non-Maximum Suppression
  keep = nms(torch.tensor(box_coordinates), torch.tensor(scores), iou_threshold=0.05)
  filtered_boxes = [box for idx, box in enumerate(box_coordinates) if idx in keep]

  """cropped_images = []
  for box in filtered_boxes:
    cropped_image_tensor = crop_and_resize_image(image, box)
    cropped_images.append(cropped_image_tensor)"""
  # Plot the image and filtered bounding boxes
  fig, ax = plt.subplots(1)
  ax.imshow(image)

  for box in filtered_boxes:
    x1, y1, x2, y2 = box
    print(f"Scaled Box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")  # Debugging print

    # Create a Rectangle patch
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='None')

    # Add the patch to the Axes
    ax.add_patch(rect)
  ax.axis("off")
  plt.show()

#this is the function call for detection of fruits
#predict_and_plot_bounding_boxes(det_model,r"E:\Downloads\Arecanut_detection_and_Ripening_time_Estimation-main\Arecanut_detection_and_Ripening_time_Estimation-main\test_image.jpg")


def classify_image(image,model,classes):

  image=image.unsqueeze(dim=0)
  model.eval()
  with torch.inference_mode():
    pred=model(image)#we get the logits here
    pred_probs=torch.softmax(pred,dim=1)
    class_label=torch.argmax(pred_probs,dim=1).item()
  return classes[class_label]

def predict_bounding_boxes(model, image_path):
  # Load and transform the image
  image = Image.open(image_path).convert("RGB")
  transform = T.Compose([T.ToTensor()])
  image_tensor = transform(image).unsqueeze(0) # Add batch dimension and move to device

  # Make prediction
  model.eval()
  with torch.no_grad():
    predictions = model(image_tensor)

  # Convert predictions to list of dictionaries
  predicted_boxes = [{
      "boxes": pred["boxes"].cpu().numpy(),
      "labels": pred["labels"].cpu().numpy(),
      "scores": pred["scores"].cpu().numpy(),
  } for pred in predictions]

  box_coordinates = predicted_boxes[0]["boxes"]  # Assuming a single image prediction
  scores = predicted_boxes[0]["scores"]

  # Perform Non-Maximum Suppression
  keep = nms(torch.tensor(box_coordinates), torch.tensor(scores), iou_threshold=0.2)
  filtered_boxes = [box for idx, box in enumerate(box_coordinates) if idx in keep]
  filtered_scores=[score for idx,score in enumerate(scores)  if idx in keep]

  cropped_images = []
  for box in filtered_boxes:
    cropped_image_tensor = crop_and_resize_image(image, box)
    cropped_images.append(cropped_image_tensor)
  return cropped_images,filtered_boxes,filtered_scores




def predict(image_dir):
  classes=["it is infloroscence requires long time.","May require around 2.5 Months to ripen ","May require somewhere around 1.5 Months to ripen","May require around 1 Month to ripen","May require around 20 Days to ripen","Ready to harvest!!"]
  pred_classes=[]
  results={"id":[],"time_required":[]}
  cropped_tensors,boxes,scores=predict_bounding_boxes(det_model,image_dir)
  for img_tensor in cropped_tensors:
    pred_class=classify_image(img_tensor,cls_model,classes)
    pred_classes.append(pred_class)
  for id,classes in enumerate(pred_classes):
     results["id"].append(id+1)
     results["time_required"].append(classes) 

  plot_bboxes(image_dir,boxes,pred_classes,scores)
  return results





def plot_bboxes(image_dir, boxes, classes, scores=None, figsize=(10, 10)):
    """
    Plot bounding boxes on the image with class names.

    Args:
    - image_dir (str): Path to the input image.
    - boxes (list): List of bounding box coordinates [(x1, y1, x2, y2), ...].
    - classes (list): List of length N containing the class names corresponding to the bounding boxes.
    - scores (list, optional): List of confidence scores for each bounding box.
    - figsize (tuple, optional): Figure size (width, height).

    Returns:
    - None: Displays the plot.
    """
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = Image.open(image_dir)
    image = T(image)

    # Convert tensor to numpy array
    image = image.numpy()

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=figsize)

    # Display the image
    ax.imshow(np.transpose(image, (1, 2, 0)))

    # Iterate over all bounding boxes
    for i, box in enumerate(boxes):
        class_name = classes[i]

        # Extract coordinates
        x1, y1, x2, y2 = box

        # Calculate box width and height
        width = x2 - x1
        height = y2 - y1

        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the plot
        ax.add_patch(rect)

        # Add class name and score (if available) to the plot
        if scores is not None:
            score = scores[i]
            ax.text(x1, y1 - 5, f"id:{i+1}", color='white', fontsize=16)
        else:
            ax.text(x1, y1 - 5, f"id:{i+1}", color='white', fontsize=16)

    # Set axis properties
    ax.axis('off')

    # Show plot
    plt.show()
#this is the function call to classify the detected fruits

#results=predict(r"E:\Downloads\Arecanut_detection_and_Ripening_time_Estimation-main\Arecanut_detection_and_Ripening_time_Estimation-main\test_image.jpg")

#the function returns the results dictionary with id of the boxes and class