from PIL import Image
import numpy as np
import cv2
from mss import mss
from pokerstars_recognition import PokerStarsTableRecognizer
from utils import read_config_file, set_window_size, remove_cards, data_concatenate
from equity import calc_equity
from info_box import update_label

sct = mss()
config = read_config_file()
set_window_size()

table_data = []
while True:
    updated_table_data = []
    monitor = {'top': 80, 'left': 70, 'width': config['table_size']['width'],
               'height': config['table_size']['height']}
    img = Image.frombytes('RGB', (config['table_size']['width'], config['table_size']['height']),
                          sct.grab(monitor).rgb)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    recognizer = PokerStarsTableRecognizer(img, config)
    hero_step = recognizer.detect_hero_step()
    if hero_step:
        hero_cards = recognizer.detect_hero_cards()
        table_cards = recognizer.detect_table_cards()
        total_pot = recognizer.find_total_pot()
        updated_table_data.append([hero_cards, table_cards, total_pot])
        if table_data == updated_table_data:
            pass
        else:
            table_data = updated_table_data
            deck = remove_cards(hero_cards, table_cards)
            equity = calc_equity(deck, hero_cards, table_cards)
            players_info = recognizer.get_dealer_button_position()
            players_info = recognizer.get_empty_seats(players_info)
            players_info = recognizer.get_so_players(players_info)
            players_info = recognizer.assign_positions(players_info)
            players_info = recognizer.find_players_bet(players_info)
            text = data_concatenate(hero_cards, table_cards, total_pot, equity, players_info)
            update_label(text)

