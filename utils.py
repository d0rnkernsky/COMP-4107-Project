import cv2 as cv
import numpy as np


def detect_face_rect(in_img, scale_factor=1.3, min_neighbors=3, min_size=(30, 30)):
    """
        Detects a face (only one) on an image from Faces in the Wild dataset
    :param in_img: an image
    :return: face region a rectangle as a tuple in the form {x, y, w, h)
    """
    gray = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)
    cascades = cv.CascadeClassifier(
        f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')
    faces = cascades.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    return faces[0]


def pixelate(image, face_rect, blocks=5):
    '''
    returns picture with a region pixelated
    face_rect = (x, y, w, h)
    '''
    px_image = image
    # divide the image region into NxN blocks
    x_steps = np.linspace(
        face_rect[0], face_rect[0]+face_rect[2], blocks + 1, dtype="int")
    y_steps = np.linspace(
        face_rect[1], face_rect[1]+face_rect[3], blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = px_image[start_y:end_y, start_x:end_x]
            (B, G, R) = [int(x) for x in cv.mean(roi)[:3]]
            cv.rectangle(px_image, (start_x, start_y), (end_x, end_y),
                         (B, G, R), -1)
    # return the pixelated blurred px_image
    return px_image
