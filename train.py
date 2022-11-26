# importing all the needed packages
import torch
from workspace_utils import active_session
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from train_args import train_input_args
from train_model import build_model, train_model
from train_preprocessing import preprocess
#Get the input arguments
in_args = train_input_args()
arch = in_args.arch
hidden_units = in_args.hidden_units
epochs = in_args.epochs
data_dir = in_args.data_directory
save_dir = in_args.save_dir
learning_rate = in_args.learning_rate
gpu = in_args.gpu

#Build the model
model = build_model(arch, hidden_units)

#Preprocess the data
train_data, valid_data, test_data, train_loader, valid_loader, test_loader = preprocess(data_dir)

#Train the model
model, optimizer, criterion = train_model(model, learning_rate, gpu, epochs, train_loader, valid_loader)

# Save the checkpoint
model.to('cpu')
model.class_to_idx = train_data.class_to_idx

checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')

if save_dir == "save_dir":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir 
 
print('Checkpoint saved to {}.'.format(save_dir_name))

