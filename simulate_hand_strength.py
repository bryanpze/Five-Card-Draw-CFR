from hand_evaluator import Evaluator
from itertools import combinations
import pickle
import random
hand_strength = {}
poker_hand_value = {}
deck = ['2c','2d','2h','2s','3c','3d','3h',
        '3s',        '4c',        '4d',        '4h',
        '4s',        '5c',
        '5d',        '5h',
        '5s',        '6c',
        '6d',        '6h',
        '6s',        '7c',
        '7d',        '7h',
        '7s',        '8c',
        '8d',        '8h',
        '8s',        '9c',
        '9d',        '9h',
        '9s',        'tc',
        'td',        'th',
        'ts',        'jc',
        'jd',        'jh',
        'js','qc', 'qd','qh', 'qs','kc','kd','kh','ks','ac','ad','ah','as']
evaluator = Evaluator()
for potential_hand in combinations(deck,5):
    poker_hand_value[''.join(sorted(potential_hand))] = evaluator.eval_hand(potential_hand)['value']
i = 0
random.shuffle(deck)
for current_player_hand in combinations(deck,5):
    current_hard_repr = ''.join(sorted(current_player_hand))
    current_hand_value = poker_hand_value[current_hard_repr]
    if current_hand_value in hand_strength:
        continue
    else:
        deck_set = set(deck)
        player_hand_set = set(current_player_hand)
        remaining_deck = list(deck_set-player_hand_set)
        num_hands_worse = 0
        num_hands_better = 0
        num_hands_equal = 0
        for current_opponent_hand in combinations(remaining_deck,5):
            opponent_hand_value = poker_hand_value[''.join(sorted(current_opponent_hand))] 
            # Here incrementation should be based on the weighting used
            if current_hand_value>opponent_hand_value:
                num_hands_worse+=1
            elif current_hand_value<opponent_hand_value:
                num_hands_better+=1
            else:
                num_hands_equal+=1
        card_hand_strength = (num_hands_worse+0.5*num_hands_equal)/(num_hands_equal+num_hands_better+num_hands_worse)
        hand_strength[current_hand_value] = card_hand_strength
    i+=1
    print(i)
with open('hand_strength_probabilities.pkl', 'wb') as f:
    pickle.dump(hand_strength, f)
