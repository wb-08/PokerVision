import cv2
from scripts.table_recognition import PokerTableRecognizer
from scripts.utils import sort_bboxes, thresholding, card_separator, table_part_recognition, \
    convert_contours_to_bboxes, find_by_template, find_closer_point, read_config_file
import numpy as np


class PokerStarsTableRecognizer(PokerTableRecognizer):

    def __init__(self, img, cfg):
        """
        Parameters:
            img(numpy.ndarray): image of the whole table
            cfg (dict): config file
        """
        self.img = img
        self.cfg = cfg

    def detect_hero_step(self):
        """
        Based on the area under hero's cards,
        we determine whether hero should make a move
        Returns:
            Boolean Value(True or False): True, if hero step now
        """
        res_img = self.img[self.cfg['hero_step_define']['y_0']:self.cfg['hero_step_define']['y_1'],
                           self.cfg['hero_step_define']['x_0']:self.cfg['hero_step_define']['x_1']]

        hsv_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2HSV_FULL)
        mask = cv2.inRange(hsv_img, np.array(self.cfg['hero_step_define']['lower_gray_color']),
                           np.array(self.cfg['hero_step_define']['upper_gray_color']))
        count_of_white_pixels = cv2.countNonZero(mask)
        return True if count_of_white_pixels > self.cfg['hero_step_define']['min_white_pixels'] else False

    def detect_cards(self, separators, sort_bboxes_method, cards_coordinates, path_to_numbers, path_to_suits):
        """
        Parameters:
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
        img = self.img[self.cfg[cards_coordinates]['y_0']:self.cfg[cards_coordinates]['y_1'],
                       self.cfg[cards_coordinates]['x_0']:self.cfg[cards_coordinates]['x_1']]
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
                color_of_img, directory = (cv2.IMREAD_COLOR, self.cfg['paths'][path_to_suits]) if key == 0 \
                    else (cv2.IMREAD_GRAYSCALE, self.cfg['paths'][path_to_numbers])
                res_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                card_part = table_part_recognition(res_img, directory, color_of_img)
                card_name = card_part + 'T' if len(cards_bboxes) == 1 else card_name + card_part
            cards_name.append(card_name[::-1])
        return cards_name

    def detect_hero_cards(self):
        """
        Returns:
            cards_name(list of str): name of the hero's cards
        """
        separators = [self.cfg['hero_cards']['separator_1'], self.cfg['hero_cards']['separator_2']]
        sort_bboxes_method = 'bottom-to-top'
        cards_coordinates = 'hero_cards'
        path_to_numbers = 'hero_cards_numbers'
        path_to_suits = 'hero_cards_suits'
        cards_name = self.detect_cards(separators, sort_bboxes_method, cards_coordinates,
                                       path_to_numbers, path_to_suits)
        return cards_name

    def detect_table_cards(self):
        """
        Returns:
            cards_name(list of str): name of the cards on the table
        """
        separators = [self.cfg['table_cards']['separator_1'], self.cfg['table_cards']['separator_2'],
                      self.cfg['table_cards']['separator_3'], self.cfg['table_cards']['separator_4'],
                      self.cfg['table_cards']['separator_5']]
        sort_bboxes_method = 'top-to-bottom'
        cards_coordinates = 'table_cards'
        path_to_numbers = 'table_cards_numbers'
        path_to_suits = 'table_cards_suits'
        cards_name = self.detect_cards(separators, sort_bboxes_method, cards_coordinates,
                                       path_to_numbers, path_to_suits)
        return cards_name

    def find_total_pot(self):
        """
        Returns:
            number(str): number with total pot
        """
        img = self.img[self.cfg['pot']['y_0']:self.cfg['pot']['y_1'],
              self.cfg['pot']['x_0']:self.cfg['pot']['x_1']]
        _, max_loc = find_by_template(img, self.cfg['paths']['pot_image'])
        bet_img = img[max_loc[1] - 3:max_loc[1] + self.cfg['pot']['height'],
                      max_loc[0] + self.cfg['pot']['pot_template_width']:
                      max_loc[0] + self.cfg['pot']['pot_template_width'] + self.cfg['pot']['width']]
        binary_img = thresholding(bet_img, 105, 255)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = convert_contours_to_bboxes(contours, 3, 1)
        bounding_boxes = sort_bboxes(bounding_boxes, method='left-to-right')
        number = ''
        for bbox in bounding_boxes:
            number_img = bet_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            symbol = table_part_recognition(number_img, self.cfg['paths']['pot_numbers'], cv2.IMREAD_GRAYSCALE)
            number += symbol
        return number

    def get_dealer_button_position(self):
        """
        determine who is closer to the dealer button
        Returns:
            player_info(dict): here is information about all players as it becomes available
        """
        player_info = {key: value for key in range(1, 7) for value in ['']}
        players_coordinates = self.cfg['player_center_coordinates']
        _, button_coordinates = find_by_template(self.img, self.cfg['paths']['dealer_button'])
        player_with_button = find_closer_point(players_coordinates, button_coordinates)
        player_info[player_with_button] = 'dealer_button'
        return player_info

    def get_missing_players(self, players_info, path_to_template_img, flag):
        """
        find players who are currently absent for various reasons
        Parameters:
            players_info(dict): key - player number, value - '' - if the player is in the game;
            '-' - if a player's seat is available; '-so-' - if the player is absent
            path_to_template_img(str): the path to the images where the benchmark images are located
            flag(str): It can be - and -so-
        Returns:
           players_info(dict): info about players
        """
        players_coordinates = self.cfg['players_coordinates']
        players_for_checking = [key for key, value in players_info.items() if value == '']
        for player, bbox in players_coordinates.items():
            if player != 1 and player in players_for_checking:
                player_img = self.img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                max_val, _ = find_by_template(player_img, self.cfg['paths'][path_to_template_img])
                if max_val > 0.8:
                    players_info[player] = flag
        return players_info

    def get_empty_seats(self, players_info):
        """
        find players whose places are currently vacant
        """
        path_to_template_img = 'empty_seat'
        flag = '-'
        players_info = self.get_missing_players(players_info, path_to_template_img, flag)
        return players_info

    def get_so_players(self, players_info):
        """
        find players who are not currently in the game
        """
        path_to_template_img = 'sitting_out'
        flag = '-so-'
        players_info = self.get_missing_players(players_info, path_to_template_img, flag)
        return players_info

    def assign_positions(self, players_info):
        """
        assign each player one of six positions if the player in the game
        Parameters:
            players_info(dict): info about players in {1:'', 2: 'dealer_button',3:'-so-' etc. } format
        Returns:
            players_info(dict): info about players in {1: 'BB', 2: 'SB', 3: '-so-' etc. } format
        """
        busy_seats = [k for k, v in players_info.items() if v != '-' and v != '-so-']
        exist_positions = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
        del exist_positions[3:3 + (6 - len(busy_seats))]
        player_with_button = [k for k, v in players_info.items() if v == 'dealer_button'][0]
        for index, player_number in enumerate(
                busy_seats[busy_seats.index(player_with_button):] + busy_seats[:busy_seats.index(player_with_button)]):
            if len(busy_seats) == 2:
                position = 'SB' if index == 0 else 'BB'
                players_info[player_number] = position
            else:
                players_info[player_number] = exist_positions[index]
        return players_info

    def find_players_bet(self, players_info):
        """
        Parameters:
            players_info(dict): info about players in {1:'BTN', 2:'SB', 3:'BB' etc. } format
        Returns:
            updated_players_info(dict): info about players in {'Hero':'BTN', 'SB':'', 'BB':'50' etc. } format
        """
        players_bet_location = self.cfg['players_bet']
        updated_players_info = {'Hero': players_info[1]}
        for i, location_coordinates in players_bet_location.items():
            if players_info[i] not in ('-so-', '-'):
                bet_img = self.img[location_coordinates[1]:location_coordinates[3],
                          location_coordinates[0]:location_coordinates[2]]
                binary_img = thresholding(bet_img, 105, 255)
                contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bounding_boxes = convert_contours_to_bboxes(contours, 3, 1)
                bounding_boxes = sort_bboxes(bounding_boxes, method='left-to-right')
                number = ''
                for bbox in bounding_boxes:
                    number_img = bet_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    symbol = table_part_recognition(number_img, self.cfg['paths']['pot_numbers'], cv2.IMREAD_GRAYSCALE)
                    number += symbol
                updated_players_info[players_info[i]] = number
            else:
                updated_players_info[i] = players_info[i]
        return updated_players_info


