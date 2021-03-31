import cv2 as cv
import numpy as np


def detect_face_rect(in_img, scale_factor=1.3, min_neighbors=3, min_size=(30, 30)):
    """
        Detects a face (only one) on an image from Faces in the Wild dataset
    :param in_img: an image
    :return: face region a rectangle as a tuple in the form {x, y, w, h)
    """
    in_img = in_img.copy()
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
    px_image = image.copy()
    # divide the image region into NxN blocks
    x_steps = np.linspace(
        face_rect[0], face_rect[0] + face_rect[2], blocks + 1, dtype="int")
    y_steps = np.linspace(
        face_rect[1], face_rect[1] + face_rect[3], blocks + 1, dtype="int")
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


def blur_image(image, face_rect, pad=9, n=10):
    """
    returns an image with the face_rect blurred
    """
    x1, y1, w1, h1 = face_rect
    orig = image.copy()
    blurred_image = image.copy()
    blurred_image = cv.GaussianBlur(
        blurred_image[x1:x1 + w1, y1:y1 + h1], (pad, pad), n)
    orig[x1:x1 + w1, y1:y1 + h1] = blurred_image
    return orig


def get_face_sizes(location):
    '''
    location of dataset folder "C:\\Users\\User\\Desktop\\dataset"
    creates 2 files
    face_sizes.txt
        holds the name of the image along with its weight and height
    sizes.txt
        simply holds the size of images that didn't throw an exception
    '''

    os.chdir(location)
    paths = os.listdir()
    sizes = []
    data = []

    for path in paths:
        try:
            img = cv.imread(path)
            face_rect = ut.detect_face_rect(img)
            x, y, w, h = face_rect
            if w != h:
                print(path, "non-square face")
            data.append((path, w, h))
            sizes.append(w)
        except:
            data.append((path, "error"))

    f = open(os.path.join(location, "face_sizes.txt"), "w")
    for item in data:
        if len(item) == 3:
            f.write(str(item[0]) + " " + str(item[1]) +
                    " " + str(item[2]) + "\n")
        else:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
    f.close()

    f = open(os.path.join(location, "sizes.txt"), "w")
    for item in sizes:
        f.write(str(item) + "\n")
    f.close()


def get_face_size_stats(location):
    '''
    location of dataset folder "C:\\Users\\User\\Desktop\\dataset\\sizes.txt"
    takes sizes.txt file and runs stats on it
    '''
    sizes = np.loadtxt(location, dtype=np.int16)
    print(len(sizes))
    fig, ax = plt.subplots(1, 1)
    ax.hist(sizes)
    ax.set_title("face sizes")
    ax.set_xlabel("size of face")
    ax.set_ylabel("number of faces")
    plt.show()
