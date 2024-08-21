import pickle
from hand_evaluator import Evaluator
from itertools import combinations
import re
from pymongo import MongoClient
import numpy as np
import random
with open('./hand_strength_probabilities.pkl','rb') as f:
    hand_strength_probs = pickle.load(f)

deck = ['2c',
        '2d',
        '2h',
        '2s',
        '3c',
        '3d',
        '3h',
        '3s',
        '4c',
        '4d',
        '4h',
        '4s',
        '5c',
        '5d',
        '5h',
        '5s',
        '6c',
        '6d',
        '6h',
        '6s',
        '7c',
        '7d',
        '7h',
        '7s',
        '8c',
        '8d',
        '8h',
        '8s',
        '9c',
        '9d',
        '9h',
        '9s',
        'tc',
        'td',
        'th',
        'ts',
        'jc',
        'jd',
        'jh',
        'js',
        'qc',
        'qd',
        'qh',
        'qs',
        'kc',
        'kd',
        'kh',
        'ks',
        'ac',
        'ad',
        'ah',
        'as']
client = MongoClient('localhost', 27017)
db = client['fivecarddatabase']
collection = db['fivecardcollection_abstraction_no_suit_2']

def parse_no_suits(card_string):
    return re.sub(r'[^atjqk0-9:]','',card_string)

random.shuffle(deck)
evaluator = Evaluator()
number_of_folds_by_basic = 0
number_of_folds_by_model = 0
i = 0
for current_player_hand in combinations(deck,5):
    current_hand_value = evaluator.eval_hand(current_player_hand)['value']
    infoset = parse_no_suits(''.join(sorted(current_player_hand)+[':']))
    node = collection.find_one({'_id': infoset})
    node['strategy_sum'] = np.array(node['strategy_sum'] )
    node["strategy_sum"] = node["strategy_sum"]/sum(node['strategy_sum'])
    if node['strategy_sum'][0]>=0.5:
        number_of_folds_by_model+=1
    if hand_strength_probs[current_hand_value]<0.5:
        number_of_folds_by_basic+=1
    i+=1
    if (i%1000)==0:
        print(i)
print(f'{number_of_folds_by_model} number of folds by model')
print(f'{number_of_folds_by_basic} number of folds by basic')