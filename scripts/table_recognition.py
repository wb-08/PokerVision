import cv2
from scripts.utils import sort_bboxes, thresholding, card_separator, table_part_recognition, \
    convert_contours_to_bboxes, read_config_file, find_by_template
import numpy as np


def detect_hero_step(img, cfg):
    """
    Based on the area under hero's cards,
    we determine whether hero should make a move
    Parameters:
        img(numpy.ndarray): image of the entire table
        cfg (dict): config file
    Returns:
        Boolean Value(True or False): True, if hero step now
    """
    res_img = img[cfg['hero_step_define']['y_0']:cfg['hero_step_define']['y_1'],
                  cfg['hero_step_define']['x_0']:cfg['hero_step_define']['x_1']]

    hsv_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2HSV_FULL)
    mask = cv2.inRange(hsv_img, np.array(cfg['hero_step_define']['lower_gray_color']),
                       np.array(cfg['hero_step_define']['upper_gray_color']))
    count_of_white_pixels = cv2.countNonZero(mask)
    return True if count_of_white_pixels > cfg['hero_step_define']['min_white_pixels'] else False


def detect_cards(img, cfg, separators, sort_bboxes_method, cards_coordinates, path_to_numbers, path_to_suits):
    """
    Parameters:
        img(numpy.ndarray): image of the entire table
        cfg (dict): config file
        separators(list of int): contains values where the card ends
        sort_bboxes_method(str): defines how we will sort the contours.
        It can be left-to-right, bottom-to-top, top-to-bottom
        cards_coordinates(str): path to cards coordinates
        path_to_numbers(str): path where located numbers (J, K etc.)
        path_to_suits(str) : path where located suits
    Returns:
        cards_name(list of str): name of the cards
    """
    cards_name = []
    img = img[cfg[cards_coordinates]['y_0']:cfg[cards_coordinates]['y_1'],
              cfg[cards_coordinates]['x_0']:cfg[cards_coordinates]['x_1']]
    binary_img = thresholding(img, 200, 255)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = convert_contours_to_bboxes(contours, 10, 2)
    bounding_boxes = sort_bboxes(bounding_boxes, method=sort_bboxes_method)
    cards_bboxes_dct = card_separator(bounding_boxes, separators)
    for _, cards_bboxes in cards_bboxes_dct.items():
        if len(cards_bboxes) == 3:
            cards_bboxes = [cards_bboxes[0]]

        elif len(cards_bboxes) == 0:
            return []

        elif len(cards_bboxes) > 3:
            raise ValueError("The number of bounding boxes should not be more than 3!")

        card_name = ''
        for key, bbox in enumerate(cards_bboxes):
            color_of_img, directory = (cv2.IMREAD_COLOR, cfg['paths'][path_to_suits]) if key == 0 \
                else (cv2.IMREAD_GRAYSCALE, cfg['paths'][path_to_numbers])
            res_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            card_part = table_part_recognition(res_img, directory, color_of_img)
            card_name = card_part + '01' if len(cards_bboxes) == 1 else card_name + card_part
        cards_name.append(card_name[::-1])
    return cards_name


def detect_hero_cards(img, cfg):
    """
    Parameters:
        img(numpy.ndarray): image of the entire table
        cfg (dict): config file
    Returns:
        cards_name(list of str): name of the hero's cards
    """
    separators = [cfg['hero_cards']['separator_1'], cfg['hero_cards']['separator_2']]
    sort_bboxes_method = 'bottom-to-top'
    cards_coordinates = 'hero_cards'
    path_to_numbers = 'hero_cards_numbers'
    path_to_suits = 'hero_cards_suits'
    cards_name = detect_cards(img, cfg, separators, sort_bboxes_method, cards_coordinates, path_to_numbers, path_to_suits)
    return cards_name


def detect_table_cards(img, cfg):
    """
    Parameters:
        img(numpy.ndarray): image of the entire table
        cfg (dict): config file
    Returns:
        cards_name(list of str): name of the cards on the table
    """
    separators = [cfg['table_cards']['separator_1'], cfg['table_cards']['separator_2'],
                  cfg['table_cards']['separator_3'], cfg['table_cards']['separator_4'], cfg['table_cards']['separator_5']]
    sort_bboxes_method = 'top-to-bottom'
    cards_coordinates = 'table_cards'
    path_to_numbers = 'table_cards_numbers'
    path_to_suits = 'table_cards_suits'
    cards_name = detect_cards(img, cfg, separators, sort_bboxes_method, cards_coordinates, path_to_numbers,
                              path_to_suits)
    return cards_name


def find_total_pot(img, cfg):
    """
    Parameters:
        img(numpy.ndarray): image of the entire table
        cfg (dict): config file
    Returns:
        number(str): number with total pot
    """
    img = img[cfg['pot']['y_0']:cfg['pot']['y_1'],
                  cfg['pot']['x_0']:cfg['pot']['x_1']]
    _, max_loc = find_by_template(img, cfg['paths']['pot_image'])
    bet_img = img[max_loc[1]-3:max_loc[1] + cfg['pot']['height'],
                  max_loc[0]+cfg['pot']['pot_template_width']:max_loc[0]+cfg['pot']['pot_template_width']+cfg['pot']['width']]
    binary_img = thresholding(bet_img, 105, 255)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = convert_contours_to_bboxes(contours, 3, 1)
    bounding_boxes = sort_bboxes(bounding_boxes, method='left-to-right')
    number = ''
    for bbox in bounding_boxes:
        number_img = bet_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        symbol = table_part_recognition(number_img, cfg['paths']['pot_numbers'], cv2.IMREAD_GRAYSCALE)
        number += symbol
    return number


