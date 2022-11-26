# importing all the needed packages
import torch
from workspace_utils import active_session
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models


def build_model(arch, hidden_units):
    """
    Builds a nueral network model based on the given architecture and hidden units.
    
    parameters:
    arch - a pretrained architecture
    hidden_units - a number of hidden units inside the architecture
    returns:
    model - an image classification model
    """
    # Load the pretrained model from pytorch
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    print("{} Model has been built with {} hidden units.".format(arch, hidden_units))
    
    # create our classifier layers with the desired number of outputs
    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
                                    ('0', nn.Linear(25088, hidden_units)),
                                    ('1', nn.ReLU()),
                                    ('2', nn.Dropout(0.2)),
                                    ('3', nn.Linear(hidden_units, 102)),
                                    ('4', nn.LogSoftmax(dim=1))
                                    ]))
    
    
    return model

def train_model(model, learning_rate, gpu, epochs, train_loader, valid_loader):
    """
    Builds a nueral network model based on the given architecture and hidden units.
    
    parameters:
    model - a pretrained architecture
    gpu - a True value for enabling the GPU while training, and False for The CPU to train the model
    epochs - number of epochs used to train the network
    train_loader - A data loader that stores the training images and their respective labels
    valid_loader - A data loader that stores the validation images and their respective labels
    returns:
    model - a trained image classification model
    optimizer - an optimizer algorithm for the model
    citerion - a loss function for the model
    """
    # Create a loss function 
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Use GPU
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device);
    
    print('Training with {} learning rate, {} epochs, and {} computing.'.format(learning_rate, epochs, (gpu)*"cuda" + (not gpu)*"cpu"))
    # Keep the session active for the training
    with active_session():

        epochs = epochs
        train_losses, valid_losses = [], []
        for epoch in range(epochs):
            train_loss = 0
            for inputs, labels in train_loader:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            else:
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()
                    for inputs, labels in valid_loader:
                        # Move input and label tensors to the default device
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model(inputs)
                        valid_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                model.train()

                # At completion of epoch
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Valid Loss: {:.3f}.. ".format(valid_losses[-1]),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                
    print("Training is done.")           
    return model, optimizer, criterion