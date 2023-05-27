from tkinter import *
from scripts.utils import read_config_file

cfg = read_config_file()
root = Tk()
root.geometry('{0}x{1}'.format(cfg['info_box_size']['width'], cfg['info_box_size']['height']))
root.title('PokerStarsHelper')
root.configure(background='ivory3')
lab = Label(root, anchor="w", justify=LEFT, font=("Arial", 18))
lab.pack(fill="both", expand=True)


def update_label(text):
    lab.configure(text=text)
    root.update()