import os
import sys
import logging
from io import BytesIO
from PIL import Image
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CWD)
Image.MAX_IMAGE_PIXELS = 1000000000
logger = logging.getLogger()

import configs as cf


def create_directory_structure(folderList):
    """create folders if non-existent"""
    savepaths = {'base': cf.SAVEPATH}
    for folder in folderList:
        folderpath = os.path.join(cf.SAVEPATH, folder)
        savepaths[folder] = folderpath
        try:
            os.mkdir(folderpath)
        except FileExistsError:
            pass
    return savepaths


def printNlog(msg, level='info'):
    print(msg)
    if level == 'info':
        logger.info(msg)
    elif level == 'error':
        logger.error(msg)
    else:
        pass


def get_image_from_htk_response(resp):
    """Given a girder response, get np array image"""
    image_content = BytesIO(resp.content)
    image_content.seek(0)
    Image.open(image_content)
    image = Image.open(image_content)
    return image


def has_excessive_white_background(image, white_threshold=200, max_white_percentage=40):
    """
    Check if an image has more than the specified percentage of white background.
    
    Args:
        image: PIL Image object or numpy array
        white_threshold: Pixel values above this are considered "white" (default: 200)
        max_white_percentage: Maximum allowed percentage of white pixels (default: 40)
    
    Returns:
        bool: True if image should be filtered out (has too much white background)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # Handle grayscale or RGB images
    if len(image_array.shape) == 3:
        # For RGB images, check if all channels are above threshold
        white_pixels = np.all(image_array > white_threshold, axis=2)
    else:
        # For grayscale images
        white_pixels = image_array > white_threshold
    
    # Calculate percentage of white pixels
    white_percentage = (np.sum(white_pixels) / white_pixels.size) * 100
    
    return white_percentage > max_white_percentage
