from abc import ABC, abstractmethod


class PokerTableRecognizer(ABC):
    @abstractmethod
    def detect_hero_step(self):
        """
        Based on the area under hero's cards,
        we determine whether hero should make a move
        """
        pass

    @abstractmethod
    def detect_hero_cards(self):
        pass

    @abstractmethod
    def detect_table_cards(self):
        pass

    @abstractmethod
    def find_total_pot(self):
        pass

    @abstractmethod
    def get_dealer_button_position(self):
        """
        determine who is closer to the dealer button
        """
        pass

    @abstractmethod
    def get_empty_seats(self, players_info):
        """
        find players whose places are currently vacant
        """
        pass

    @abstractmethod
    def get_so_players(self, players_info):
        """
        find players who are not currently in the game
        """
        pass

    @abstractmethod
    def find_players_bet(self, players_info):
        pass







