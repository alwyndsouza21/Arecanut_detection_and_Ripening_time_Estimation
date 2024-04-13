from flask import Flask, render_template,request
import torch
import requests
import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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


"""def download_model_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)

# Download the model
file_id = '1JK6NeBcjk6dr0L-KcGVV7qzinDb4uAaM'
destination = 'det_model.pth'
download_model_from_google_drive(file_id,destination)"""


model,transforms=create_detection_model()
model.load_state_dict(torch.load("det_model.pth"))

# Creating an app instance
app = Flask(__name__)

# Creating a route
@app.route("/")
def hello():
    return render_template("start_page.html")

@app.route("/detect",methods=['GET', 'POST'])
def detect():
  if request.method == 'POST':
  # Handle image detection here
    pass
  return render_template("detection_page.html")
  
@app.route("/estimate",methods=["GET","POST"])
def estimate():
  if request.method=="POST":
    #handle image estimation here
    pass
  return render_template("estimation_page.html")


if __name__ == '__main__':
    # Running the app with specified host and port
    app.run(debug=True, host='0.0.0.0', port=8080)
