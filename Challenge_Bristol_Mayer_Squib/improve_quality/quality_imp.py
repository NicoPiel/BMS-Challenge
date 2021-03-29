import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def quality_imp(img, tupel):
   
    kernel = np.ones(tupel, np.uint8)
    erosion = cv2.erode(img,kernel,iterations=1)
    
    return erosion