import unittest
from scripts.table_recognition import detect_hero_step, detect_hero_cards, detect_table_cards, find_total_pot,\
    get_dealer_button_position, get_empty_seats, get_so_players, assign_positions, find_players_bet
from scripts.utils import read_config_file, load_images

cfg = read_config_file('../scripts/config.yaml')
test_cfg = read_config_file('test_config.yaml')


class TestTableRecognition(unittest.TestCase):

    def test_hero_step(self):
        images, file_names = load_images(test_cfg['paths']['hero_step'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestHeroStep Incorrect detection in the image", filename=filename):
                self.assertEqual(detect_hero_step(image, cfg), test_cfg['hero_step'][filename])

    def test_hero_cards(self):
        images, file_names = load_images(test_cfg['paths']['hero_cards'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestHeroCards Incorrect detection in the image", filename=filename):
                self.assertEqual(detect_hero_cards(image, cfg), test_cfg['hero_cards'][filename])

    def test_table_cards(self):
        images, file_names = load_images(test_cfg['paths']['table_cards'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestTableCards Incorrect detection in the image", filename=filename):
                self.assertEqual(detect_table_cards(image, cfg), test_cfg['table_cards'][filename])

    def test_total_pot(self):
        images, file_names = load_images(test_cfg['paths']['total_pot'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestTotalPot Incorrect detection in the image", filename=filename):
                self.assertEqual(find_total_pot(image, cfg), test_cfg['total_pot'][filename])

    def test_dealer_button_position(self):
        images, file_names = load_images(test_cfg['paths']['dealer_button_position'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestDealerButton Incorrect detection in the image", filename=filename):
                self.assertEqual(get_dealer_button_position(image, cfg), test_cfg['dealer_button_position'][filename])

    def test_player_position(self):
        images, file_names = load_images(test_cfg['paths']['player_position'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestPlayerPosition Incorrect detection in the image", filename=filename):
                players_info = get_dealer_button_position(image, cfg)
                players_info = get_empty_seats(image, cfg, players_info)
                players_info = get_so_players(image, cfg, players_info)
                self.assertEqual(assign_positions(players_info), test_cfg['player_position'][filename])

    def test_player_bet(self):
        images, file_names = load_images(test_cfg['paths']['player_bet'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestPlayerBet Incorrect detection in the image", filename=filename):
                players_info = get_dealer_button_position(image, cfg)
                players_info = get_empty_seats(image, cfg, players_info)
                players_info = get_so_players(image, cfg, players_info)
                players_info = assign_positions(players_info)
                self.assertEqual(find_players_bet(image, cfg, players_info), test_cfg['player_bet'][filename])


if __name__ == '__main__':
    unittest.main()
