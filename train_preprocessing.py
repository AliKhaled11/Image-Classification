# Import python modules
import torch
from workspace_utils import active_session
from torchvision import datasets, transforms, models

def preprocess(data_dir):
    """
    Loads and transforms the data for training, validation, testing.
    Parameters:
     data_dir - directory of all the data needed for training, validation, testing.
    Returns:
     train_data - A generic data loader where the training images are stored
     valid_data - A generic data loader where the validation images are stored
     test_data - A generic data loader where the testing images are stored
     train_loader - A data loader that stores the training images and their respective labels
     valid_loader - A data loader that stores the validation images and their respective labels
     test_loader  - A data loader that stores the testing images and their respective labels
    """
    #complete the directories paths
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # create the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    # Create the data loaders for each dataset
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader
    
    