# Import python modules
import argparse

def predict_input_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 5 command line arguments. If 
    the user fails to provide some or all of the 5 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Path to image as /path/to/image with default value 'flowers/test/1/image_06743.jpg'
      2. Trained CNN model path as checkpoint with default value 'checkpoint.pth'
      3. Top k predictions as --topk with default value 3
      4. json file of category to name as --category_names with default value 'cat_to_name.json'
      5. Enabling the GPU for the training process as --gpu with default value False
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_image', type = str, default = 'flowers/test/1/image_06743.jpg', 
                        help = 'path to the image to be predicted. default = "flowers/test/1/image_06743.jpg"')
    
    parser.add_argument('checkpoint', type = str, default = 'checkpoint.pth', 
                        help = 'checkpoint of the trained model. default = "checkpoint.pth"')
    
    parser.add_argument('--topk', type = int, default = 3,
                        help = 'top k predictions. default = 3')
    
    parser.add_argument('--category_names', type = str, default = "cat_to_name.json",
                        help = 'json file of category to name. default = "cat_to_name.json"')
    
    parser.add_argument('--gpu', type = str, default = True,
                        help = 'Use GPU for training, set to True or False. default = True')

    return  parser.parse_args()