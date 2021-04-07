import utils as ut
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

location = os.path.join(os.getcwd(), "dataset")

# creates dataset of full sized images with sub-section obfurcated
ut.dataset_prep(location, 111)

# creates dataset of images cropped around face that are entirely obfurcated
ut.dataset_prep(location, 111, 125)
