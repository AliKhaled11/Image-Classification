# Import python modules
import argparse

def train_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 7 command line arguments. If 
    the user fails to provide some or all of the 7 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Data training Folder as data_directory with default value 'flowers'
      2. CNN Model Architecture as --arch with default value 'vgg13'
      3. Network learning rate as --learning_rate with default value 0.001
      4. Network number of hidden units as --hidden_units with default value 512
      5. Training number of epochs as --epochs with default value 10
      6. Enabling the GPU for the training process as --gpu with default value False
      7. Directory to save checkpoints as --save_dir with default value 'save_dir'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', type = str, default = 'flowers', 
                        help = 'path to the folder of training data. default = "flowers"')
    
    parser.add_argument('--save_dir', type = str, default = 'save_dir', 
                        help = 'path to the folder of pet images. default = "save_dir"')
    
    parser.add_argument('--arch', type = str, default = 'vgg13',
                        help = 'the classifier required. default = "vgg13"')
    
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'the learning rate of the network. default = 0.001')
    
    parser.add_argument('--hidden_units', type = int, default = 512,
                        help = 'number of hiddden units of the architecture. default = 512')
    
    parser.add_argument('--epochs', type = int, default = 20,
                        help = 'number of epochs to train the model. default = 20')
    
    parser.add_argument('--gpu', type = str, default = True,
                        help = 'Use GPU for training, set to True or False. default = True')

    return  parser.parse_args()
