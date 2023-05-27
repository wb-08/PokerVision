import numpy as np
import eval7


def calc_equity(deck, hero_cards, table_cards, iters=100000):
    """
    Parameters:
        deck(list of str): all other cards, not including cards that are on the table and hero cards
        hero_cards(list of str): cards that belong to the hero
        table_cards(list of str): cards that are on the table
        iters(int): the amount that the table generates
    Returns:

    """
    deck = [eval7.Card(card) for card in deck]
    table_cards = [eval7.Card(card) for card in table_cards]
    hero_cards = [eval7.Card(card) for card in hero_cards]
    max_table_cards = 5
    win_count = 0
    for _ in range(iters):
        np.random.shuffle(deck)
        num_remaining = max_table_cards - len(table_cards)
        draw = deck[:num_remaining+2]
        opp_hole, remaining_comm = draw[:2], draw[2:]
        player_hand = hero_cards + table_cards + remaining_comm
        opp_hand = opp_hole + table_cards + remaining_comm
        player_strength = eval7.evaluate(player_hand)
        opp_strength = eval7.evaluate(opp_hand)

        if player_strength > opp_strength:
            win_count += 1

    win_prob = (win_count / iters) * 100
    return round(win_prob, 2)

