import cv2 as cv


def detect_face_rect(in_img, scale_factor=1.3, min_neighbors=3, min_size=(30, 30)):
    """
        Detects a face (only one) on an image from Faces in the Wild dataset
    :param in_img: an image
    :return: face region a rectangle as a tuple in the form {x, y, w, h)
    """
    gray = cv.cvtColor(in_img, cv.COLOR_RGB2GRAY)
    cascades = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')
    faces = cascades.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    return faces[0]
