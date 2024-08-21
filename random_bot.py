import numpy as np
import random
import itertools
from hand_evaluator import Evaluator
from rule import Rule
class RandomBot:
    def __init__(self,bet_strategy='Random',rule = Rule()):
        self.rule = rule
        self.bet_actions = [0,1]
        for possible_raise in range(2,sum(self.rule.stack)+1):
            self.bet_actions.append(possible_raise)
        self.num_bet_actions = len(self.bet_actions)
        tmp = []
        tmp.append([])
        for i in range(1,6):
            tmp.extend([list(x) for x in list(itertools.combinations([0,1,2,3,4],i))])
        self.draw_actions_lookup = {}
        for index,val in enumerate(tmp):
            self.draw_actions_lookup[index] = val
        self.draw_actions = list(self.draw_actions_lookup.keys())
        self.num_draw_actions = len(self.draw_actions)
        self.evaluator = Evaluator()
        self.name = bet_strategy
    def get_random_action(self,round,valid_actions,poker_cards,history):
        if round==0 or round==2:
            actual_strategy = np.zeros(self.num_bet_actions)
            valid_action_strategy = np.zeros(len(valid_actions))
            if self.name=='Random':
                if len(valid_action_strategy)==1:
                    valid_action_strategy[0] = 1
                elif len(valid_action_strategy)==2:
                    valid_action_strategy[0] = 1/2
                    valid_action_strategy[1] = 1/2
                else:
                    valid_action_strategy[0] = 1/3
                    valid_action_strategy[1] = 1/3
                    if(len(valid_actions)-2)>0:
                        valid_action_strategy[2] = 1/3
                        # valid_action_strategy[2:] = 1/(len(valid_actions)-2)
            elif self.name=='Raise':
                if len(valid_action_strategy)==1:
                    valid_action_strategy[0] = 1
                elif len(valid_action_strategy)==2:
                    valid_action_strategy[1] = 1
                else:
                    valid_action_strategy[2] = 1
            elif self.name=='Call':
                if len(valid_action_strategy)==1:
                    valid_action_strategy[0] = 1
                elif len(valid_action_strategy)>1:
                    valid_action_strategy[1] = 1
            actual_strategy[valid_actions] = valid_action_strategy
            return random.choices(self.bet_actions,weights=actual_strategy,k=1)[0]
        else:
            hand_info = self.evaluator.eval_hand(poker_cards)
            # print(hand_info)
            if hand_info['handName'] in ['straight_flush','four of a kind','full house','straight']:
                return 0 
            elif hand_info['handName']=='three of a kind':
                cards_with_suit_stripped = [card[0] for card in poker_cards]
                non_trip_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=3])
                action = None
                for key in self.draw_actions_lookup:
                    if self.draw_actions_lookup[key]==non_trip_index:
                        action = key
                return action
            elif hand_info['handName']=='two pairs':
                cards_with_suit_stripped = [card[0] for card in poker_cards]
                non_pair_index = None
                for card_index, card in enumerate(cards_with_suit_stripped):
                    if cards_with_suit_stripped.count(card)==1:
                        non_pair_index = card_index
                for key in self.draw_actions_lookup:
                    if self.draw_actions_lookup[key]==[non_pair_index]:
                        action =key
                return action
            else:
                cards_with_number_stripped = [card[1] for card in poker_cards] 
                cards_with_suit_stripped = [card[0] for card in poker_cards]
                
                potential_flush = False
                for card_index, card in enumerate(cards_with_number_stripped):
                    if cards_with_number_stripped.count(card)==4:
                        potential_flush = True

                if potential_flush:
                    non_flush_index = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            non_flush_index = card_index
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==[non_flush_index]:
                            action = key
                    return action
                else:
                    cards_with_suit_stripped = sorted(cards_with_suit_stripped)
                    while 'a' in cards_with_suit_stripped:
                        cards_with_suit_stripped.remove('a')
                        cards_with_suit_stripped.append(1)
                        cards_with_suit_stripped.append(14)
                    while 't' in cards_with_suit_stripped:
                        cards_with_suit_stripped.remove('t')
                        cards_with_suit_stripped.append(10)
                    while 'j' in cards_with_suit_stripped:
                        cards_with_suit_stripped.remove('j')
                        cards_with_suit_stripped.append(11)
                    while 'q' in cards_with_suit_stripped:
                        cards_with_suit_stripped.remove('q')
                        cards_with_suit_stripped.append(12)
                    while 'k' in cards_with_suit_stripped:
                        cards_with_suit_stripped.remove('k')
                        cards_with_suit_stripped.append(13)
                    consecutive_numbers = 0
                    starting_number = None
                    cards_with_suit_stripped = sorted(list(map(int, cards_with_suit_stripped)))

                    for i in range(len(cards_with_suit_stripped)-1):
                        if int(cards_with_suit_stripped[i+1])-int(cards_with_suit_stripped[i])==1:
                            consecutive_numbers+=1
                            if starting_number==None:
                                starting_number = int(cards_with_suit_stripped[i])
                        else:
                            if consecutive_numbers!=3:
                                consecutive_numbers=0
                    if consecutive_numbers==3:
                        straight_numbers =  [str(starting_number),str(starting_number+1),str(starting_number+2),str(starting_number+3),str(starting_number+4)] 
                        replacements = {'1':'a','14':'a','10':'t','11':'j','12':'q','13':'k'}
                        straight_numbers = [replacements.get(char,char) for char in straight_numbers]
                        non_straight_index = None
                        cards_with_suit_stripped = [card[0] for card in poker_cards]
                        for index,card in enumerate(poker_cards):
                            if card[0] not in straight_numbers or cards_with_suit_stripped.count(card[0])==2:
                                non_straight_index = index
                        
                        for key in self.draw_actions_lookup:
                            if self.draw_actions_lookup[key]==[non_straight_index]:
                                action = key
                        return action
                    else:
                        if hand_info['handName']=='one pair':
                            cards_with_suit_stripped = [card[0] for card in poker_cards]
                            non_pair_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=2])
                            for key in self.draw_actions_lookup:
                                if self.draw_actions_lookup[key]==non_pair_index:
                                    action = key
                            return action
                        else:
                            drop_everything_not_high = [index for index,card in enumerate(poker_cards) if card[0] not in {'a','j','q','k'}]
                            for key in self.draw_actions_lookup:
                                if self.draw_actions_lookup[key]==drop_everything_not_high:
                                    action = key
                            return action
# print(RandomBot().get_random_action(1,[],['ah', 'qc', 'tc', 'jh', 'kc']))
                    