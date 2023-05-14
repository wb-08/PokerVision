import cv2
import numpy as np
import yaml
import os
from time import sleep
from math import sqrt


def read_config_file(filename='config.yaml'):
    """
    Parameters:
        filename (str): config file name
    Returns: loaded_data (dict)
    """
    with open(filename, 'r') as stream:
        loaded_data = yaml.safe_load(stream)
        return loaded_data


def sort_bboxes(bounding_boxes, method):
    """
    Parameters:
         bounding_boxes(list of lists of int): bounding_boxes in [x_0, y_0, x_1, y_1] format
         method(int): the method of sorting bounding boxes.
         It can be left-to-right, bottom-to-top or top-to-bottom
    Returns:
        bounding_boxes (list of tuple of int): sorted bounding boxes.
        Each bounding box presented in [x_0, y_0, x_1, y_1] format
    """

    methods = ['left-to-right', 'bottom-to-top', 'top-to-bottom']
    if method not in methods:
        raise ValueError("Invalid method. Expected one of: %s" % methods)

    else:

        if method == 'left-to-right':
            bounding_boxes.sort(key=lambda tup: tup[0])

        elif method == 'bottom-to-top':
            bounding_boxes.sort(key=lambda tup: tup[1], reverse=True)

        elif method == 'top-to-bottom':
            bounding_boxes.sort(key=lambda tup: tup[1], reverse=False)
        return bounding_boxes


def mse(img, benchmark_img):
    """
    the 'Mean Squared Error' between two images that
    is the sum of the squared difference between the two images.
    NOTE: the two images must have the same dimension.
    Parameters:
        img(numpy.ndarray): image of a part of the table
        benchmark_img(numpy.ndarray): benchmark image.
        This image read from the folder.
    Returns:
        err (float): the error between two images
    """
    err = np.sum((img.astype("float") - benchmark_img.astype("float")) ** 2)
    err /= float(img.shape[0] * img.shape[1])
    return err


def image_comparison(img, benchmark_img, color_of_img):
    """
    Parameters:
        img(numpy.ndarray): image of a part of the table
        benchmark_img(str): path to benchmark image
        color_of_img(int): set in which format to read the image.
        It can be cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE
    Returns:
       err (float): the error between two images
    """
    colors = [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE]
    if color_of_img not in colors:
        raise ValueError("Invalid method. Expected one of: %s" % colors)
    benchmark_img = cv2.imread(benchmark_img, color_of_img)
    res_img = cv2.resize(img, (benchmark_img.shape[1], benchmark_img.shape[0]))
    if color_of_img == cv2.IMREAD_GRAYSCALE:
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    err = mse(res_img, benchmark_img)
    return err


def thresholding(img, value_1, value_2):
    """
    Parameters:
        img(numpy.ndarray): image of a part of the table
        value_1(int): threshold value
        value_2(int): the maximum value that is assigned
        to pixel values that exceed the threshold value
    Returns: binary_img(numpy.ndarray)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(img, value_1, value_2, cv2.THRESH_BINARY)
    return binary_img


def table_part_recognition(img, directory, color_of_img):
    """
    Parameters:
        img(numpy.ndarray): image of a part of the table
        directory(str): path to the template images
        color_of_img(int): set in which format to read the image.
        It can be cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE
    Returns:
        table_part(str): recognized suit, value (J, K etc.), pot etc.
    """
    err_dict = {}
    for full_image_name in os.listdir(directory):
        image_name = full_image_name.split('.')[0]
        err = image_comparison(img, directory + full_image_name, color_of_img)
        err_dict[image_name] = err
    table_part = min(err_dict, key=err_dict.get)
    return table_part


def convert_contours_to_bboxes(contours, min_height, min_width):
    """
    convert contours to bboxes, also remove all small bounding boxes
    Parameters:
        contours (tuple): each individual contour is a numpy array
         of (x, y) coordinates of boundary points of the object
        contours(list of tuples of int): bounding boxes suits
        and values (J, K etc.) in [x, y, w, h] format
    Returns:
        cards_bboxes(list of lists of int): bounding boxes suits
        and values (J, K etc.) in [x_0, y_0, x_1, y_1] format
    """
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    cards_bboxes = []
    for i in range(0, len(bboxes)):
        x, y, w, h = bboxes[i][0], bboxes[i][1], \
            bboxes[i][2], bboxes[i][3]
        if h >= min_height and w >= min_width:
            contour_coordinates = [x - 1, y - 1, x + w + 1, y + h + 1]
            cards_bboxes.append(contour_coordinates)
    return cards_bboxes


def card_separator(bboxes, separators):
    """
    determine which bounding box belongs to which card
    Parameters:
        bboxes(list of lists of int): bounding boxes suits and values (J, K etc.)
        separators(list of int): contains values where the card ends
    Returns:
       sorted_dct(dict) key - card number, value - bounding boxes
    """
    dct = {}
    for bbox in bboxes:
        for separator in separators:
            if bbox[2] < separator:
                dct[separators.index(separator)] = dct.get(separators.index(separator), []) + [bbox]
                break
    sorted_dct = {key: value for key, value in sorted(dct.items(), key=lambda item: int(item[0]))}
    return sorted_dct


def find_by_template(img, path_to_image):
    """
     Object detection using a "template".
     Parameters:
         img(numpy.ndarray): image of a part of the table/of the entire table
         path_to_image(str): the path to a template image
    Returns:
        max_val(float): largest value with the most likely match
        max_loc(tuple of int): location with the largest value.
        Location is presented in (x, y) format
    """
    template_img_gray = cv2.imread(path_to_image, 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img_gray, template_img_gray,
                               cv2.TM_CCOEFF_NORMED)
    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)
    return max_val, max_loc


def load_images(directory):
    """
    Parameters:
        directory(str): path to specific directory
    Returns:
        images(list of numpy.ndarray): images of a part of the table
        file_names(list of str): the name of the files in the folder
    """
    images = [cv2.imread(directory + file) for file in sorted(os.listdir(directory))]
    file_names = [file for file in sorted(os.listdir(directory))]
    return images, file_names


def find_closer_point(players_coordinates, button_coordinates):
    """
    find the distance between two points  and find the minimum distance
    Parameters:
        players_coordinates(dict): key - player number, value - player coordinates
        button_coordinates(tuple): Coordinates are presented in (x, y) format
    Returns:
        player_with_button(int): the number of the player who is closest to the buttonthe number of the player who is closest to the button
    """
    distance_dict = {}
    for player, player_coordinates in players_coordinates.items():
        distance = sqrt((player_coordinates[0] - button_coordinates[0]) ** 2
                        + (player_coordinates[1] - button_coordinates[1]) ** 2)
        distance_dict[player] = distance
    player_with_button = min(distance_dict, key=distance_dict.get)
    return player_with_button


def set_window_size():
    """
    set the application window in the right place and with the right size
    """
    sleep(3)
    cmd = 'wmctrl -r :ACTIVE: -e 0,0,0,1100,900'
    os.system(cmd)


