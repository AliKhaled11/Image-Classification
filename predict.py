# importing all the needed packages
import torch
from workspace_utils import active_session
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from train_model import build_model, train_model
from train_preprocessing import preprocess
import json
from predict_args import predict_input_args
from predict_preprocessing import process_image

#Get the input arguments
in_args = predict_input_args()
image_path = in_args.path_to_image
checkpoint = in_args.checkpoint
topk = in_args.topk
category_names = in_args.category_names
gpu = in_args.gpu

#Read in the cat_to_name file
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
#Load a checkpoint and rebuild the model
def load_checkpoint(checkpoint):
    """
    Loads the checkpoint of the saved trained model.
    parameters:
    checkpoint - path to the model checkpoint
    returns:
    model - a trained CNN model
    """
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    return model

#Load model
model = load_checkpoint(checkpoint)

#Process input image
image = process_image(image_path)



#Define a prediction function
def predict(model, topk, image, gpu):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    parameters:
    model - trained CNN model
    topk - number of top predictions
    image - preprocessed image to be predicted
    gpu - a True value for enabling the GPU while training, and False for The CPU to train the model
    returns:
    probs - probabilities of the predictions
    classes - classes of the predictions
    '''
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    model.eval()
    print("Making predictions...")
    
    with torch.no_grad():
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.type(torch.FloatTensor)
        image = image.to(device)
        
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        probs, indices = ps.topk(topk, dim=1)
        probs = probs.cpu().numpy().reshape(topk)
        idx_to_class = {c:i for i, c in model.class_to_idx.items()}
        classes = [idx_to_class[int(i)] for i in indices[0]]
        
    return probs, classes

# Predict class and probabilities
print(f"Predicting top {topk} most likely flower names from image {image_path}.")

probs, classes = predict(model, topk, image, gpu)
classes_name = [cat_to_name[class_i] for class_i in classes]

print("\nFlower name : probability ")
print("---")

for i in range(len(probs)):
    print("{} : {}".format(classes_name[i], round(probs[i], 3)))
print("")