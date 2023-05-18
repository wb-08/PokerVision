import unittest
from scripts.pokerstars_recognition import PokerStarsTableRecognizer
from scripts.utils import read_config_file, load_images

cfg = read_config_file('../scripts/config.yaml')
test_cfg = read_config_file('test_config.yaml')


class TestTableRecognition(unittest.TestCase):

    def test_hero_step(self):
        images, file_names = load_images(test_cfg['paths']['hero_step'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestHeroStep Incorrect detection in the image", filename=filename):
                recognizer = PokerStarsTableRecognizer(image, cfg)
                self.assertEqual(recognizer.detect_hero_step(), test_cfg['hero_step'][filename])

    def test_hero_cards(self):
        images, file_names = load_images(test_cfg['paths']['hero_cards'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestHeroCards Incorrect detection in the image", filename=filename):
                recognizer = PokerStarsTableRecognizer(image, cfg)
                self.assertEqual(recognizer.detect_hero_cards(), test_cfg['hero_cards'][filename])

    def test_table_cards(self):
        images, file_names = load_images(test_cfg['paths']['table_cards'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestTableCards Incorrect detection in the image", filename=filename):
                recognizer = PokerStarsTableRecognizer(image, cfg)
                self.assertEqual(recognizer.detect_table_cards(), test_cfg['table_cards'][filename])

    def test_total_pot(self):
        images, file_names = load_images(test_cfg['paths']['total_pot'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestTotalPot Incorrect detection in the image", filename=filename):
                recognizer = PokerStarsTableRecognizer(image, cfg)
                self.assertEqual(recognizer.find_total_pot(), test_cfg['total_pot'][filename])

    def test_dealer_button_position(self):
        images, file_names = load_images(test_cfg['paths']['dealer_button_position'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestDealerButton Incorrect detection in the image", filename=filename):
                recognizer = PokerStarsTableRecognizer(image, cfg)
                self.assertEqual(recognizer.get_dealer_button_position(), test_cfg['dealer_button_position'][filename])

    def test_player_position(self):
        images, file_names = load_images(test_cfg['paths']['player_position'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestPlayerPosition Incorrect detection in the image", filename=filename):
                recognizer = PokerStarsTableRecognizer(image, cfg)
                players_info = recognizer.get_dealer_button_position()
                players_info = recognizer.get_empty_seats(players_info)
                players_info = recognizer.get_so_players(players_info)
                self.assertEqual(recognizer.assign_positions(players_info), test_cfg['player_position'][filename])

    def test_player_bet(self):
        images, file_names = load_images(test_cfg['paths']['player_bet'])
        for image, filename in zip(images, file_names):
            with self.subTest("TestPlayerBet Incorrect detection in the image", filename=filename):
                recognizer = PokerStarsTableRecognizer(image, cfg)
                players_info = recognizer.get_dealer_button_position()
                players_info = recognizer.get_empty_seats(players_info)
                players_info = recognizer.get_so_players(players_info)
                players_info = recognizer.assign_positions(players_info)
                self.assertEqual(recognizer.find_players_bet(players_info), test_cfg['player_bet'][filename])


if __name__ == '__main__':
    unittest.main()
