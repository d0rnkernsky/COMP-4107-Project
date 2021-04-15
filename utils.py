import cv2 as cv
import numpy as np
from enum import Enum
import os


class PixelationLevel(Enum):
    Hard = 10
    Medium = 17
    Light = 25


class BlurLevel(Enum):
    Hard = (23, 25)
    Medium = (13, 15)
    Light = (9, 10)


def crop_face(img, face_rect, size):
    '''
    face_rect = (x, y, w, h)
    size = int
    size represents the height and width of the cropped photo
    will be centered around face_rect
    '''
    x, y, w, h = face_rect
    # print("face_rect")
    # print("x:", x)
    # print("y:", y)
    # print("w:", w)
    # print("h:", h)
    # print("x to x+w:", x, x+w)
    # print("y to y+h:", y, w+h)
    center_x = (x + x + w) // 2
    center_y = (y + y + h) // 2
    x = center_x - (size//2)
    y = center_y - (size//2)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x+size > 250:
        diff = x+size - 250
        x = x - diff
    if y+size > 250:
        diff = y+size - 250
        y = y - diff
    # print("cropped")
    # print("x:", x)
    # print("y:", y)
    # print("w:", size)
    # print("h:", size)
    # print("x to x+size:", x, x+size)
    # print("y to y+size:", y, w+size)
    img = img.copy()
    crop = img[y:y+size, x:x+size]
    return crop


def create_folders(location):
    # print(location)
    folder = os.path.abspath(os.path.join(location, ".."))
    # print(folder)
    if os.path.exists(folder) == False:
        os.makedirs(folder)


def resize_and_save(file_name, size):
    '''
    save cropped images into a new folder inside of dataset

    must run via cmd prompt or terminal (not in jupyter)

    creates necessary folders
    won't save images unless folders exist
    '''
    img = cv.imread(file_name)
    # print("\nfile_name", file_name)

    face_rect = None

    try:
        face_rect = detect_face_rect(img)
    except:
        print("detect_face_rect error on file:", file_name)
        return

    # print("Face_rect:", face_rect)

    # crop photo and center acround face
    crop_img = crop_face(img, face_rect, size)

    # print("crop_img", type(crop_img))
    # print("crop_img", crop_img.shape)
    # print("crop_img", crop_img.dtype)
    # cv.imshow("crop_img", crop_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # different save location folder for cropped images
    save_folder = os.path.abspath(
        os.path.join(os.getcwd(), str(size)))
    # print("save path", save_folder)

    cropped_location = os.path.join(
        save_folder, file_name)
    # print("cropped_location:", cropped_location)

    # create folders for saving files
    create_folders(cropped_location)

    # save as JPG
    cv.imwrite(cropped_location, crop_img, [cv.IMWRITE_JPEG_QUALITY, 100])


def preprocess_and_save(file_name, blur_level=None, pixel_level=None, size=None):
    '''
    if size is specificed then crop around face of that size
    if no size is specificed don't crop and simpliy obfurcate sub-section

    must run via cmd prompt or terminal (not in jupyter)

    creates necessary folders
    won't save images unless folders exist
    '''
    assert blur_level is not None
    assert pixel_level is not None

    assert isinstance(pixel_level, PixelationLevel)
    assert isinstance(blur_level, BlurLevel)

    # assert pixel_level.name == PixelationLevel.name

    img = cv.imread(file_name)
    # print("\nfile_name", file_name)

    face_rect = None

    try:
        face_rect = detect_face_rect(img)
    except:
        print("detect_face_rect error on file:", file_name)
        return

    if type(size) == int:
        # crop photo and center acround face
        crop_img = crop_face(img, face_rect, size)

        # print("crop_img", type(crop_img))
        # print("crop_img", crop_img.shape)
        # print("crop_img", crop_img.dtype)
        # cv.imshow("crop_img", crop_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # obfurcate entire image
        blurred = blur_image(crop_img, (0, 0, size, size), blur_level)
        pixelated = pixelate(crop_img, (0, 0, size, size), pixel_level)

        # different save location folder for cropped images
        save_folder = os.path.abspath(os.path.join(os.getcwd(), "..", "processed", str(size)))

    else:
        # print("img", type(img))
        # print("img", img.shape)
        # print("img", img.dtype)
        # cv.imshow("img", img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # obfurcate subsection of image
        blurred = blur_image(img, face_rect, blur_level)
        pixelated = pixelate(img, face_rect, pixel_level)

        # different save location folder for full sized images
        save_folder = os.path.abspath(
            os.path.join(os.getcwd(), "..", "processed", "250"))

    # print("blurred", type(blurred))
    # print("blurred", blurred.shape)
    # print("blurred", blurred.dtype)
    # cv.imshow("blurred", blurred)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # print("pixelated", type(pixelated))
    # print("pixelated", pixelated.shape)
    # print("pixelated", pixelated.dtype)
    # cv.imshow("pixelated", pixelated)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # print("save path", save_folder)
    blurred_location = os.path.join(
        save_folder, f"{blur_level.name}_blur", file_name)

    pixelated_location = os.path.join(
        save_folder, f"{pixel_level.name}_pixelate", file_name)

    # create folders for saving files
    create_folders(blurred_location)
    create_folders(pixelated_location)

    # save as JPG
    # print(blurred_location)
    # print(pixelated_location)
    cv.imwrite(
        blurred_location, blurred, [cv.IMWRITE_JPEG_QUALITY, 100])
    cv.imwrite(pixelated_location, pixelated,
               [cv.IMWRITE_JPEG_QUALITY, 100])


def dataset_prep(dataset_location, levels, size=None):
    '''
    dataset_location = path to dataset folder
    levels = 000
    each int being a switch for a level of obfuscation
    000 no levels
    100 light
    110 light, medium
    111 all levels
    size = int
    which will represnt the size of the cropped image

    saves new pictures in a processed folder on the same level as dataset

    must run via cmd prompt or terminal (not in jupyter)
    '''
    os.chdir(dataset_location)
    # paths = os.listdir()
    # create a dir list of files exclusively
    # having folders in list will cause errors
    # paths = os.listdir()
    paths = [f for f in os.listdir() if os.path.isfile(f)]

    if size != None:
        # create a new folder of cropped unobfuscated images
        for path in paths:
            # print("path", path)
            resize_and_save(path, size)

    if str(levels)[0] == "1":
        # create and save light obfuscated images
        for path in paths:
            # print("path", path)
            preprocess_and_save(path,
                                BlurLevel.Light, PixelationLevel.Light, size)

    if str(levels)[1] == "1":
        # create and save medium obfuscated images
        for path in paths:
            # print("path", path)
            preprocess_and_save(path,
                                BlurLevel.Medium, PixelationLevel.Medium, size)

    if str(levels)[2] == "1":
        # create and save hard obfuscated images
        for path in paths:
            # print("path", path)
            preprocess_and_save(path,
                                BlurLevel.Hard, PixelationLevel.Hard, size)


def detect_face_rect(in_img, scale_factor=1.1, min_neighbors=3, min_size=(30, 30)):
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


def pixelate(image, face_rect, level):
    '''
    returns picture with a region pixelated
    face_rect = (x, y, w, h)
    '''
    assert isinstance(level, PixelationLevel)
    level = level.value

    px_image = image.copy()
    # divide the image region into NxN blocks
    x_steps = np.linspace(
        face_rect[0], face_rect[0] + face_rect[2], level + 1, dtype="int")
    y_steps = np.linspace(
        face_rect[1], face_rect[1] + face_rect[3], level + 1, dtype="int")
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


def blur_image(image, face_rect, level):
    """
    returns an image with the face_rect blurred
    """
    assert isinstance(level, BlurLevel)

    pad, n = level.value
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
            face_rect = detect_face_rect(img)
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
    # print(len(sizes))
    fig, ax = plt.subplots(1, 1)
    ax.hist(sizes)
    ax.set_title("face sizes")
    ax.set_xlabel("size of face")
    ax.set_ylabel("number of faces")
    plt.show()
