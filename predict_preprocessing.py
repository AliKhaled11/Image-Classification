# importing all the needed packages
import numpy as np
from PIL import Image

# Image Preprocessing
def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array.
    parameters:
    image_path - path to image to be processed
    returns:
    np_image - a numpy array of the processed image
    '''
    
    image = Image.open(image_path)
    width, height = image.size   # Get dimensions

    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    aspect_ratio = width / height
    if aspect_ratio > 1:
        image = image.resize((round(aspect_ratio * 256), 256))
    else:
        image = image.resize((256, round(256 / aspect_ratio)))
    
    # Crop out the center 224x224 portion of the image
    width, height = image.size
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    image = image.crop((round(left), round(top), round(right), round(bottom)))
    
    # Convert color channels to 0-1
    np_image = np.array(image) / 255
    
    # Normalize the image
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image