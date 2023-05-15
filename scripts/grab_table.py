from PIL import Image
import numpy as np
import cv2
from mss import mss
from table_recognition import detect_hero_step, detect_hero_cards, detect_table_cards,find_total_pot,  \
     get_dealer_button_position, get_empty_seats, get_so_players, assign_positions, find_players_bet
from utils import read_config_file, set_window_size, remove_cards, data_concatenate
from equity import calc_equity
from info_box import update_label

sct = mss()
config = read_config_file()
# set_window_size()

table_data = []
while True:
    updated_table_data = []
    monitor = {'top': 80, 'left': 70, 'width': config['table_size']['width'],
               'height': config['table_size']['height']}
    img = Image.frombytes('RGB', (config['table_size']['width'], config['table_size']['height']),
                          sct.grab(monitor).rgb)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    hero_step = detect_hero_step(img, config)
    if hero_step:
        hero_cards = detect_hero_cards(img, config)
        table_cards = detect_table_cards(img, config)
        total_pot = find_total_pot(img, config)
        updated_table_data.append([hero_cards, table_cards, total_pot])
        if table_data == updated_table_data:
            pass
        else:
            table_data = updated_table_data
            deck = remove_cards(hero_cards, table_cards)
            equity = calc_equity(deck, hero_cards, table_cards)
            players_info = get_dealer_button_position(img, config)
            players_info = get_empty_seats(img, config, players_info)
            players_info = get_so_players(img, config, players_info)
            players_info = assign_positions(players_info)
            players_info = find_players_bet(img, config, players_info)
            text = data_concatenate(hero_cards, table_cards, total_pot, equity, players_info)
            update_label(text)

