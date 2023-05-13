from PIL import Image
import numpy as np
import cv2
from mss import mss
from table_recognition import detect_hero_step
from utils import read_config_file

sct = mss()
config = read_config_file()


while True:
        monitor = {'top': 80, 'left': 70, 'width': config['table_size']['table_width'],
                   'height': config['table_size']['table_height']}
        img = Image.frombytes('RGB', (config['table_size']['table_width'], config['table_size']['table_height']),
                              sct.grab(monitor).rgb)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        hero_step = detect_hero_step(img, config)