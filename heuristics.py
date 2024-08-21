import random
import numpy as np
from hand_evaluator import Evaluator
from itertools import combinations
import itertools
from rule import Rule
from collections import defaultdict
import pickle
import re
class HeuristicsBot:
    def __init__(self,basic_flag = True,  potential_flag = False,semi_bluffing_flag = False,pot_odds_flag = False, showdown_odds_flag = False,agent_modeling_flag = False,rule=Rule()):
        self.basic_flag = basic_flag
        self.potential_flag = potential_flag
        self.semi_bluffing_flag = semi_bluffing_flag
        self.pot_odds_flag = pot_odds_flag
        self.showdown_odds_flag = showdown_odds_flag
        self.agent_modeling_flag = agent_modeling_flag
        self.deck = ['2c',
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
        self.poker_hand_value = {}
        self.opp_card_weights = {}
        self.opp_action_freq = defaultdict(lambda:[0]*2)
        self.semi_bluffing = False
        for potential_hand in combinations(self.deck,5):
            val = self.evaluator.eval_hand(potential_hand)['value']
            self.poker_hand_value[''.join(sorted(potential_hand))] = val
            if val not in self.opp_card_weights:
                self.opp_card_weights[val] = 1
        if self.basic_flag:
            self.name = 'Basic'
            with open('./hand_strength_probabilities.pkl','rb') as f:
                self.hand_strength_probs = pickle.load(f)
        if self.potential_flag:
            self.name = 'Potential'
        if self.semi_bluffing_flag:
            self.name = 'Semi-Bluffing'
        if self.pot_odds_flag:
            self.name = 'Pot-Odds'
        if self.agent_modeling_flag:
            self.name = 'Agent-Modeling'
    def find_last_action(self,history):
        for index,char in enumerate(reversed(history)):
            if char.isalpha():
                if char=='c' or char=='r':
                    return [char,len(history)-index-1]
        
    def check_if_no_bet_otherwise_fold(self,history):
        splitted_histories = history.split('/')
        if len(splitted_histories)==1:
            if len(history)==0:
                # Fold if we are small blind
                return 0
            else:
                # Check if opponent does not bet, otherwise fold

                return 0 if 'r' in history else 1
        else:
            return 0 if 'r' in splitted_histories[2] else  1

    def get_random_action(self,round,valid_actions,poker_cards,history):
        if (history=='c' or history=='r4' or history==''):
            self.semi_bluffing = None
            if self.agent_modeling_flag:
                # reset weights at the start of the round
                self.opp_prev_freq_tmp = None
                self.opp_prev_action_tmp = None
                for card_strength in self.opp_card_weights:
                    self.opp_card_weights[card_strength] = 1
        if round==0:
            if self.agent_modeling_flag:
                # Update opp action frequencies
                if len(history)!=0:
                    last_action = self.find_last_action(history)
                    action_val = None
                    if (type(last_action))==list and last_action[0]=='r':
                        action_val = 1
                    elif (type(last_action))==list and last_action[0]=='c':
                        action_val = 0
                    if (type(last_action))==list :
                        opp_prev_history = history[:last_action[1]]
                        self.opp_prev_freq_tmp = self.opp_action_freq[opp_prev_history]
                        self.opp_prev_action_tmp = action_val
                        self.opp_action_freq[opp_prev_history][action_val]+=1
                # Reweigh
                if self.opp_prev_freq_tmp:
                    raise_threshold = 1- (self.opp_prev_freq_tmp[1]/sum(self.opp_prev_freq_tmp))
                    call_threshold = raise_threshold - (self.opp_prev_freq_tmp[0]/sum(self.opp_prev_freq_tmp))
                    if self.opp_prev_action_tmp==1:
                         mean =raise_threshold
                         std = 0.4*(1-mean)
                    elif self.opp_prev_action_tmp==0:
                        mean = call_threshold
                        std = 0.4*(1-mean)
                    for i in self.hand_strength_probs:
                        reweigh = (self.hand_strength_probs[i] - mean + std)/(2*std)
                        if reweigh<0.01:
                            reweigh = 0.01
                        elif reweigh>1:
                            reweigh = 1
                        self.opp_card_weights[i] = self.opp_card_weights[i]*reweigh 
            actual_strategy = np.zeros(self.num_bet_actions)
            valid_action_strategy = np.zeros(len(valid_actions))
            if (self.basic_flag and not self.potential_flag):
                current_hard_repr = ''.join(sorted(poker_cards)) 
                current_hand_value = self.poker_hand_value[current_hard_repr]
                if self.agent_modeling_flag:
                    hand_strength = self.simulate_hand_strength(poker_cards)
                else:
                    hand_strength = self.hand_strength_probs[current_hand_value]
                if hand_strength>=0.85:
                    if history.count('r')<=2:
                        if (len(valid_actions)-2)>0:
                            val = 2
                        else:
                            val = 1
                        valid_action_strategy[val] = 1
                    else:
                        valid_action_strategy[1] = 1
                elif hand_strength>=0.5:
                    valid_action_strategy[1] = 1
                else:
                    parsed_history = self.parse_history(history)
                    if self.pot_odds_flag:
                        if len(valid_actions)>2:
                            to_call = (valid_actions[-1]-1)/2
                            if to_call<=2:
                                to_call = 2
                        else:
                            to_call = 2
                        pot_odds = to_call/(parsed_history['potsize']+to_call)
                        if hand_strength>=pot_odds:
                            valid_action_strategy[1] = 1
                        else:
                            val = self.check_if_no_bet_otherwise_fold(history)
                            valid_action_strategy[val] = 1
                    else:
                        val = self.check_if_no_bet_otherwise_fold(history)
                        valid_action_strategy[val] = 1
                    val = self.check_if_no_bet_otherwise_fold(history)
                    valid_action_strategy[val] = 1
                actual_strategy[valid_actions] = valid_action_strategy
                return random.choices(self.bet_actions,weights=actual_strategy,k=1)[0]
            elif (self.basic_flag and self.potential_flag):
                parsed_history = self.parse_history(history)
                current_hard_repr = ''.join(sorted(poker_cards)) 
                current_hand_value = self.poker_hand_value[current_hard_repr]
                hand_strength = self.hand_strength_probs[current_hand_value]
                # hand_strength = self.simulate_hand_strength(poker_cards)
                hand_potential = self.simulate_hand_potential(poker_cards)
                ehs = hand_strength+(1-hand_strength)*hand_potential
                if ehs>=0.85:
                    if history.count('r')<=2:
                        self.semi_bluffing = hand_potential
                        val =  2 if (len(valid_actions)-2)>0 else 1
                        valid_action_strategy[val] = 1
                    else:
                        valid_action_strategy[1] = 1
                elif ehs>=0.5:
                    valid_action_strategy[1] = 1
                else:
                    if self.pot_odds_flag:
                        if len(valid_actions)>2:
                            to_call = (valid_actions[-1]-1)/2
                            if to_call<=2:
                                to_call = 2
                        else:
                            to_call = 2
                        pot_odds = to_call/(parsed_history['potsize']+to_call)
                        if hand_potential>=pot_odds:
                            valid_action_strategy[1] = 1
                        else:
                            val = self.check_if_no_bet_otherwise_fold(history)
                            valid_action_strategy[val] = 1
                    else:
                        val = self.check_if_no_bet_otherwise_fold(history)
                        valid_action_strategy[val] = 1
                actual_strategy[valid_actions] = valid_action_strategy
                return random.choices(self.bet_actions,weights=actual_strategy,k=1)[0]
        elif round==1:
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
        elif round==2:
            if self.agent_modeling_flag:
                # Update opp action frequencies
                splitted_histories = history.split('/')
                if len(splitted_histories[2])!=0:
                    last_action = self.find_last_action(history)
                    action_val = None
                    if (type(last_action))==list and last_action[0]=='r':
                        action_val = 1
                    elif (type(last_action))==list and last_action[0]=='c':
                        action_val = 0
                    if (type(last_action))==list :
                        opp_prev_history = history[:last_action[1]]                
                        self.opp_prev_freq_tmp = self.opp_action_freq[opp_prev_history]
                        self.opp_prev_action_tmp = action_val
                        self.opp_action_freq[opp_prev_history][action_val]+=1

                
                # Reweigh
                if self.opp_prev_freq_tmp:
                    raise_threshold = 1- (self.opp_prev_freq_tmp[1]/sum(self.opp_prev_freq_tmp))
                    call_threshold = raise_threshold - (self.opp_prev_freq_tmp[0]/sum(self.opp_prev_freq_tmp))
                    if self.opp_prev_action_tmp==1:
                         mean =raise_threshold
                         std = 0.4*(1-mean)
                    elif self.opp_prev_action_tmp==0:
                        mean = call_threshold
                        std = 0.4*(1-mean)
                    for i in self.hand_strength_probs:
                        reweigh = (self.hand_strength_probs[i] - mean + std)/(2*std)
                        if reweigh<0.01:
                            reweigh = 0.01
                        elif reweigh>1:
                            reweigh = 1
                        self.opp_card_weights[i] = self.opp_card_weights[i]*reweigh 
            actual_strategy = np.zeros(self.num_bet_actions)
            valid_action_strategy = np.zeros(len(valid_actions))
            if (self.basic_flag or self.potential_flag):
                current_hard_repr = ''.join(sorted(poker_cards)) 
                current_hand_value = self.poker_hand_value[current_hard_repr]
                if self.agent_modeling_flag:
                    hand_strength = self.simulate_hand_strength(poker_cards)
                else:
                    hand_strength = self.hand_strength_probs[current_hand_value]
                parsed_history = self.parse_history(history)
                if hand_strength>=0.85:
                    splitted_histories = history.split('/')
                    if splitted_histories[2].count('r')<=2:
                        val =  2 if (len(valid_actions)-2)>0 else 1
                        valid_action_strategy[val] = 1
                    else:
                        valid_action_strategy[1] = 1
                elif hand_strength>=0.5:
                    if self.semi_bluffing and self.semi_bluffing>=(2*(valid_actions[-1]-1))/(parsed_history['potsize']+4*(valid_actions[-1]-1)+2*(valid_actions[-1]-1)):
                        val =  2 if (len(valid_actions)-2)>0 else 1
                        valid_action_strategy[val] = 1
                    else:
                        valid_action_strategy[1] = 1
                else:
                    if self.semi_bluffing and self.semi_bluffing>=(2*(valid_actions[-1]-1))/(parsed_history['potsize']+4*(valid_actions[-1]-1)+2*(valid_actions[-1]-1)):
                        val =  2 if (len(valid_actions)-2)>0 else 1
                        valid_action_strategy[val] = 1
                    elif self.pot_odds_flag:
                        if len(valid_actions)>2:
                            to_call = (valid_actions[-1]-1)/2
                            if to_call<=2:
                                to_call = 2
                        else:
                            to_call = 2
                        pot_odds = to_call/(parsed_history['potsize']+to_call)
                        if hand_strength>=pot_odds:
                            valid_action_strategy[1] = 1
                        else:
                            val = self.check_if_no_bet_otherwise_fold(history)
                            valid_action_strategy[val] = 1

                    else:
                        val = self.check_if_no_bet_otherwise_fold(history)
                        valid_action_strategy[val] = 1
                actual_strategy[valid_actions] = valid_action_strategy
                return random.choices(self.bet_actions,weights=actual_strategy,k=1)[0]
    
    def get_round(self,infoset):
        # Round 0: first betting round, Round 1: drawing round, Round 2: second betting round
        rounds = infoset.split('/')
        return len(rounds)-1
        

    def simulate_hand_strength(self,current_player_hand):
        # print('start')
        deck_set = set(self.deck)
        current_hard_repr = ''.join(sorted(current_player_hand))
        current_hand_value = self.poker_hand_value[current_hard_repr]
        player_hand_set = set(current_player_hand)
        remaining_deck = list(deck_set-player_hand_set)
        num_hands_worse = 0
        num_hands_better = 0
        num_hands_equal = 0
        for current_opponent_hand in combinations(remaining_deck,5):
            opponent_hand_value = self.poker_hand_value[''.join(sorted(current_opponent_hand))] 
            # Here incrementation should be based on the weighting used
            if current_hand_value>opponent_hand_value:
                num_hands_worse+=self.opp_card_weights[opponent_hand_value]
            elif current_hand_value<opponent_hand_value:
                num_hands_better+=self.opp_card_weights[opponent_hand_value]
            else:
                num_hands_equal+=self.opp_card_weights[opponent_hand_value]
        hand_strength = (num_hands_worse+0.5*num_hands_equal)/(num_hands_equal+num_hands_better+num_hands_worse)
        # print(hand_strength)
        return hand_strength
    def parse_history(self,history):
        potsize = 0
        player_total_stakes = [0]*self.rule.num_players
        player_current_stakes = [self.rule.blinds[i] for i in range(self.rule.num_players)]
        current_max_bet = self.rule.blinds[1]
        player_start_index = 2
        folded_players = set()
        numbers = re.findall(r'r(\d+)', history) 
        numbers = (int(n) for n in numbers)
        for i, action in enumerate(history):
            if player_start_index%self.rule.num_players in folded_players:
                player_start_index+=1
            if action=='c':
                player_current_stakes[player_start_index%self.rule.num_players] = current_max_bet
                player_start_index+=1
            elif action == 'r':
                num = next(numbers)
                player_current_stakes[player_start_index%self.rule.num_players] = num
                player_start_index+=1
                current_max_bet = num
                i += len(str(num))  # Skip over the number
            elif action=='f':
                potsize+=player_current_stakes[player_start_index%self.rule.num_players]
                player_total_stakes[player_start_index%self.rule.num_players]+=player_current_stakes[player_start_index%self.rule.num_players]
                player_current_stakes[player_start_index%self.rule.num_players] = 0
                folded_players.add(player_start_index%self.rule.num_players)
                player_start_index+=1

            elif action=='b':
                player_start_index+=1

            elif action=='/':
                # reset max bet to 0
                # After first round, round always starts with player to the left of the dealer, which is the small blind
                player_start_index = 0
                current_max_bet = 0
                player_total_stakes = [x + y for x,y in zip(player_total_stakes,player_current_stakes)]
                potsize += sum(player_current_stakes) 
                player_current_stakes = [0]*self.rule.num_players
        potsize+=sum(player_current_stakes)

        # Valid Actions
        # If the infoset is already in node_map, we can skip this as the node has already been created
        current_player = (player_start_index)%self.rule.num_players
        current_round = self.get_round(history)
        if current_round==1:
            valid_actions = self.draw_actions
        else:
            valid_actions = []
            # Each Player should be allowed to fold
            valid_actions.append(0)
            # Player should be allowed to call as long he still has the capital
            call_bet = current_max_bet
            current_player_capital = self.rule.stack[current_player]-player_current_stakes[current_player]-player_total_stakes[current_player]
            able_to_call = current_player_capital >= call_bet-player_current_stakes[current_player]
            if able_to_call:
                valid_actions.append(1)
            # To raise, player needs to raise an amount equal to twice the current largest bet
            min_bet = max(current_max_bet*2,self.rule.blinds[1])
            max_bet = current_player_capital
            if max_bet>=min_bet:
                if self.rule.bet_abstraction=='raise':
                    valid_actions.append(min_bet+1)
                else:
                    valid_actions.extend(list(range(min_bet+1,max_bet+2,self.rule.bet_abstraction)))
        
        # Round Ended
        players_remaining = self.rule.num_players - len(folded_players)
        round_ended = False
        if current_round==1:
            round_1_seq = history.split('/')[1]
            # Round ends when all players still playing has drawn their cards
            # b corresponds to number of burnt cards seen from player x in X players still playing
            # d corresponds to draw action taken from draw_actions
            num_players_already_drawn = 0
            for i in range(len(round_1_seq)):
                if round_1_seq[i]=='b' or round_1_seq[i]=='d':
                    num_players_already_drawn+=1
            if num_players_already_drawn==players_remaining:
                round_ended = True
        elif current_round==0:
            if players_remaining==1:
                round_ended = True
            else:
                round_0_seq = history.split('/')[0]
                num_calls = 0
                for i in range(len(round_0_seq)-1,-1,-1):
                    if round_0_seq[i]=='c':
                        num_calls+=1
                    elif round_0_seq[i]=='r':
                        num_calls+=1
                        break
                if num_calls==players_remaining:
                    round_ended = True        
        else:
            if players_remaining==1:
                round_ended = True
            round_2_seq = history.split('/')[2]
            num_calls = 0
            for i in range(len(round_2_seq)-1,-1,-1):
                if round_2_seq[i]=='c':
                    num_calls+=1
                elif round_2_seq[i]=='r':
                    num_calls+=1
                    break
            if num_calls==players_remaining:
                round_ended = True
        if (players_remaining==1 and round_ended) or (current_round==2 and round_ended):
            return {'terminal':True,'players_remaining':players_remaining,'round_ended':round_ended,'folded_players':folded_players,'player_total_stakes':player_total_stakes,'player_current_stakes':player_current_stakes,'potsize':potsize,'current_round':current_round}
                      
        return {'potsize':potsize,'current_round':current_round,'current_player':current_player,'valid_actions':valid_actions,'folded_players':folded_players,'round_ended':round_ended}

    def simulate_hand_potential(self,current_player_hand):

        # Assumptions
        # Will only play if at least a pair, a potential straight (4 cards with consecutive numbering), a potential flush
        # If straight flush/flush/straight/four of a kind/full house, do not draw
        # Draw to triples
        # Draw to two pairs
        # Draw to pairs but first consider potential straight or potential flush
        # Draw to high cards J,A,K,Q

        # Running the original (uncommented function) takes too long
        # We simulate a few hands and infer a value from these numbers
        hand_info = self.evaluator.eval_hand(current_player_hand)
        deck_set = set(self.deck)
        player_hand_set = set(current_player_hand)
        remaining_deck_set = deck_set-player_hand_set
        remaining_deck = list(remaining_deck_set)
        current_hand_value = hand_info['value']
        if (hand_info['handName'] in ['straight_flush','four of a kind','full house','straight']):
            return 0
        elif hand_info['handName']=='three of a kind':
            # 2223a - 0.1053
            # 22234 - 0.1051
            # aaaqk - 0.106
            # cards_with_suit_stripped = [card[0] for card in current_player_hand]
            # non_trip_index = [i for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=3]
            # for card_index in sorted(non_trip_index,reverse=True):
            #     del current_player_hand[card_index]
            # behind_ahead = 0
            # behind_tie = 0
            # tie_ahead = 0
            # behind_sum = 0
            # tie_sum = 0
            # random.shuffle(remaining_deck)
            # i = 0
            # # We take a sample as drawing five cards takes too long for entire 47 cards
            # for current_opponent_hand in combinations(remaining_deck[:35],5):
            #     opponent_hand_value = self.poker_hand_value[''.join(sorted(current_opponent_hand))]                 
            #     remaining_deck_for_draw = remaining_deck_set-set(current_opponent_hand)
            #     for drawed_cards in combinations(remaining_deck_for_draw,2):
            #         new_hand_value = self.poker_hand_value[''.join(sorted(current_player_hand+list(drawed_cards)))]
            #         if new_hand_value>opponent_hand_value and current_hand_value<opponent_hand_value:
            #             behind_ahead+=1
            #             behind_sum+=1
            #         elif new_hand_value==opponent_hand_value and current_hand_value<opponent_hand_value:
            #             behind_tie+=1
            #             behind_sum+=1
            #         elif new_hand_value>opponent_hand_value and current_hand_value==opponent_hand_value:
            #             tie_ahead+=1
            #             tie_sum+=1
            #         elif current_hand_value<opponent_hand_value:
            #             behind_sum+=1
            #         elif current_hand_value==opponent_hand_value:
            #             tie_sum+=1
            # print((behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5))
            # return (behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5)    
            return 0.105
        elif hand_info['handName']=='two pairs':
            # This is how to simulate it manually
            # Shows value of 0.087-0.9% of the time regardless of what values the two pairs are
            # aakkq - 0.0895
            # 22334 - 0.0879
            # cards_with_suit_stripped = [card[0] for card in current_player_hand]
            # non_pair_index = None
            # for card_index, card in enumerate(cards_with_suit_stripped):
            #     if cards_with_suit_stripped.count(card)==1:
            #         non_pair_index = card_index
            # del current_player_hand[non_pair_index]
            # behind_ahead = 0
            # behind_tie = 0
            # tie_ahead = 0
            # behind_sum = 0
            # tie_sum = 0
            # # print(current_hand_value)
            # for current_opponent_hand in combinations(remaining_deck,5):
            #     # print('h')
            #     opponent_hand_value = self.poker_hand_value[''.join(sorted(current_opponent_hand))]                 
            #     remaining_deck_for_draw = remaining_deck_set-set(current_opponent_hand)
            #     for drawed_cards in combinations(remaining_deck_for_draw,1):
            #         new_hand_value = self.poker_hand_value[''.join(sorted(current_player_hand+list(drawed_cards)))]
            #         if new_hand_value>opponent_hand_value and current_hand_value<opponent_hand_value:
            #             behind_ahead+=1
            #             behind_sum+=1
            #         elif new_hand_value==opponent_hand_value and current_hand_value<opponent_hand_value:
            #             behind_tie+=1
            #             behind_sum+=1
            #         elif new_hand_value>opponent_hand_value and current_hand_value==opponent_hand_value:
            #             tie_ahead+=1
            #             tie_sum+=1
            #         elif current_hand_value<opponent_hand_value:
            #             behind_sum+=1
            #         elif current_hand_value==opponent_hand_value:
            #             tie_sum+=1
            # print((behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5))
            return 0.09    

        else:
            cards_with_number_stripped = [card[1] for card in current_player_hand] 
            cards_with_suit_stripped = [card[0] for card in current_player_hand]
            
            potential_flush = False
            potential_straight = False
            for card_index, card in enumerate(cards_with_number_stripped):
                if cards_with_number_stripped.count(card)==4:
                    potential_flush = True
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
            cards_with_suit_stripped = sorted(list(map(int, cards_with_suit_stripped))
)
            for i in range(len(cards_with_suit_stripped)-1):
                if (int(cards_with_suit_stripped[i+1])-int(cards_with_suit_stripped[i]))==1:
                    consecutive_numbers+=1
                    if starting_number==None:
                        starting_number = int(cards_with_suit_stripped[i])
                else:
                    if consecutive_numbers!=3:
                        consecutive_numbers=0
            if consecutive_numbers==3:
                potential_straight=True
            # print(f'potential_flush {potential_flush}')
            # print(f'potential_straight {potential_straight}')
            
            if potential_flush and potential_straight:
                # 2345 diamonds + k hearts = 0.426
                # 2345 diamonds + 7 hearts = 0.474
                
                # 3456 diamonds + a hearts = 0.4109
                # 3456 diamonds + 8 hearts = 0.506

                # 4567 diamonds + a hearts = 0.426
                # 4567 diamonds + 2 hearts = 0.513

                # 5678 diamonds + a hearts = 0.441
                # 5678 diamonds + 2 hearts = 0.524
                # 0.083 

                # 6789 diamonds + a hearts = 0.456
                # 678a diamonds + 2 hearts = 0.533
                # 0.077
                 
                # 789t diamond  + a hearts = 0.469
                # 789t diamond  + 2 hearts = 0.539
                # 0.07
                
                # 89tj diamond  + a hearts = 0.481
                # 89tj diamond  + 2 hearts = 0.540
                # 0.059
                
                # 9tjq diamond + a hearts = 0.492
                # 9tjq diamond + 2 hearts = 0.534
                # 0.042

                # tjqk diamonds +8 hearts = 0.516
                # tjqk diamonds +2 hearts = 0.517
                # 0.001
                
                # value is higher when disposed card is low value
                # This effect is largest when the straight is low value
                # we start with a baseline value (the value when the disposed card is lowest)
                # decrease the value as disposed card gets large
                if starting_number==2:
                    baseline = 0.474
                    lowest_number = 7
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.048/(13-lowest_number)*(odd_number-lowest_number))
                elif starting_number==3:
                    baseline = 0.506
                    lowest_number = 8
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.0951/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==4:
                    baseline = 0.513
                    lowest_number = 2
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.087/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==5:
                    baseline = 0.524
                    lowest_number = 2
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.083/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==6:
                    baseline = 0.533
                    lowest_number = 2
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.077/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==7:
                    baseline = 0.539
                    lowest_number = 2
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.07/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==8:
                    baseline = 0.540
                    lowest_number = 2
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.059/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==9:
                    baseline = 0.534
                    lowest_number = 2
                    odd_number = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            odd_card = current_player_hand[card_index]
                            if odd_card[0]=='t':
                                odd_number = 10
                            elif odd_card[0]=='j':
                                odd_number = 11
                            elif odd_card[0]=='q':
                                odd_number=12
                            elif odd_card[0]=='k':
                                odd_number = 13
                            elif odd_card[0]=='a':
                                odd_number = 14
                            else:
                                odd_number = int(odd_card[0])
                    return baseline-(0.042/(14-lowest_number)*(odd_number-lowest_number))
                else:
                    return 0.516
                # non_flush_index = None
                # for card_index, card in enumerate(cards_with_number_stripped):
                #     if cards_with_number_stripped.count(card)==1:
                #         non_flush_index = card_index
                # del current_player_hand[non_flush_index]
                # behind_ahead = 0
                # behind_tie = 0
                # tie_ahead = 0
                # behind_sum = 0
                # tie_sum = 0
                # # print(current_hand_value)
                # for current_opponent_hand in combinations(remaining_deck,5):
                #     # print('h')
                #     opponent_hand_value = self.poker_hand_value[''.join(sorted(current_opponent_hand))]                 
                #     remaining_deck_for_draw = remaining_deck_set-set(current_opponent_hand)
                #     for drawed_cards in combinations(remaining_deck_for_draw,1):
                #         new_hand_value = self.poker_hand_value[''.join(sorted(current_player_hand+list(drawed_cards)))]
                #         if new_hand_value>opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_ahead+=1
                #             behind_sum+=1
                #         elif new_hand_value==opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_tie+=1
                #             behind_sum+=1
                #         elif new_hand_value>opponent_hand_value and current_hand_value==opponent_hand_value:
                #             tie_ahead+=1
                #             tie_sum+=1
                #         elif current_hand_value<opponent_hand_value:
                #             behind_sum+=1
                #         elif current_hand_value==opponent_hand_value:
                #             tie_sum+=1
                # print((behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5))
                # return (behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5)    
            elif potential_flush:
                # Probability of potential flush is 13C4*(52-13)C1/52C5 = 0.0107                 
                
                # 2456 diamond + a heart - 0.344
                # 2456 diamond + 8 heart - 0.44
               
                # 2467 diamond + 9 heart - 0.38
                # 2467 diamond + 8 heart - 0.382
                # 2467 diamond + a heart - 0.287
               
                # 9qka diamond + 2 heart - 0.382
                # 9qka diamond + 8 heart - 0.382
               
                # tqka diamond + 9 heart - 0.45
                # tqka diamond + 2 heart - 0.45
               
                # 8qka diamond + 9 heart - 0.379
                # 8qka diamond + 2 heart - 0.378

                # 2567 diamond + a heart - 0.292
                # 2567 diamond + 3 heart - 0.382

                # 23ka diamond + 4 heart - 0.324
                # 23ka diamond + 5 heart - 0.324

                # 2367 diamond + a heart - 0.283
                # 2367 diamond + j - 0.363
                # 2367 diamond + q - 0.348
                # 2367 diamond + k - 0.323
                # 2367 diamond + 8 heart - 0.379

                # 2jqk diamond + 3 heart - 0.383
                # 2jqk diamond + 9 heart - 0.380
                
                # 2tqk diamond + 3 heart - 0.382
                # 2tqk diamond + 9 heart - 0.379
                
                # 'ad','kd','td','qd','2h' - 0.45
                # 'ad','kd','td','qd','9h' - 0.45

                # 'ad','jd','td','9d','kh' - 0.368
                # 'ad','jd','td','9d','2h' - 0.386

                # '2d','3d','5d','7d','ah' - 0.28
                # '2d','3d','5d','7d','4h' - 0.474
                # '2d','3d','5d','7d','6h' - 0.376

                # '7d','8d','td','kd','ah' - 0.341
                # '7d','8d','td','kd','2h' - 0.397
                # '7d','8d','qd','kd','ah' - 0.348
                # '7d','8d','qd','kd','2h' - 0.393

                # '3d','8d','td','kd','ah' - 0.321
                # '3d','8d','td','kd','2h' - 0.385
                # '3d','8d','qd','kd','ah' - 0.327
                # '3d','8d','qd','kd','2h' - 0.38

                # hard to infer anything, we calculate the mean and std dev and generate a random number
                return np.random.normal(0.371,0.0477)
                # non_flush_index = None
                # for card_index, card in enumerate(cards_with_number_stripped):
                #     if cards_with_number_stripped.count(card)==1:
                #         non_flush_index = card_index
                # del current_player_hand[non_flush_index]
                # behind_ahead = 0
                # behind_tie = 0
                # tie_ahead = 0
                # behind_sum = 0
                # tie_sum = 0
                # # print(current_hand_value)
                # for current_opponent_hand in combinations(remaining_deck,5):
                #     # print('h')
                #     opponent_hand_value = self.poker_hand_value[''.join(sorted(current_opponent_hand))]                 
                #     remaining_deck_for_draw = remaining_deck_set-set(current_opponent_hand)
                #     for drawed_cards in combinations(remaining_deck_for_draw,1):
                #         new_hand_value = self.poker_hand_value[''.join(sorted(current_player_hand+list(drawed_cards)))]
                #         if new_hand_value>opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_ahead+=1
                #             behind_sum+=1
                #         elif new_hand_value==opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_tie+=1
                #             behind_sum+=1
                #         elif new_hand_value>opponent_hand_value and current_hand_value==opponent_hand_value:
                #             tie_ahead+=1
                #             tie_sum+=1
                #         elif current_hand_value<opponent_hand_value:
                #             behind_sum+=1
                #         elif current_hand_value==opponent_hand_value:
                #             tie_sum+=1
                # return (behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5)    
            elif potential_straight:
                # '2d','3s','4h','5d','kh' - 0.278
                # '2d','3s','4h','5d','7h' - 0.333
                # 0.055

                # '3d','4s','5h','6d','ah' - 0.263
                #'3d','4s','5h','6d','8h' - 0.372
                # 0.109

                # '4d','5s','6h','7d','ah' - 0.279
                # '4d','5s','6h','7d','2h' - 0.379
                # 0.1

                # '5d','6s','7h','8d','ah' - 0.294
                # '5d','6s','7h','8d','2h' - 0.390
                # 0.096

                # '6d','7s','8h','9d','ah' - 0.308
                # '6d','7s','8h','9d','2h' - 0.399
                # 0.091

                # '7d','8s','9h','td','ah' - 0.322
                # '7d','8s','9h','td','2h' - 0.404
                # 0.082

                # '8d','9s','th','jd','ah' - 0.334
                # '8d','9s','th','jd','2h' - 0.403
                # 0.069

                # '9d','ts','jh','qd','ah' - 0.345
                # '9d','ts','jh','qd','2h' - 0.392
                # 0.047

                if starting_number==2:
                    baseline = 0.333
                    lowest_number = 7
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-(0.055/(13-lowest_number)*(odd_number-lowest_number))
                elif starting_number==3:
                    baseline = 0.372
                    lowest_number = 8
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-(0.109/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==4:
                    baseline = 0.379
                    lowest_number = 2
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-(0.1/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==5:
                    baseline = 0.390
                    lowest_number = 2
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-( 0.096/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==6:
                    baseline = 0.399
                    lowest_number = 2
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-(0.091/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==7:
                    baseline = 0.404
                    lowest_number = 2
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-(0.082/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==8:
                    baseline = 0.403
                    lowest_number = 2
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-( 0.069/(14-lowest_number)*(odd_number-lowest_number))
                elif starting_number==9:
                    baseline = 0.392
                    lowest_number = 2
                    odd_number = None
                    straight_numbers =  [starting_number,starting_number+1,starting_number+2,starting_number+3,starting_number+4] 
                    for i in range(len(cards_with_suit_stripped)):
                        if cards_with_suit_stripped.count(cards_with_suit_stripped[i])==2 or (int(cards_with_suit_stripped[i]) not in straight_numbers and int(cards_with_suit_stripped[i])!=1):
                            odd_number = int(cards_with_suit_stripped[i])
                    return baseline-(0.047/(14-lowest_number)*(odd_number-lowest_number))
                else:
                    return 0.368

                # straight_numbers =  [str(starting_number),str(starting_number+1),str(starting_number+2),str(starting_number+3),str(starting_number+4)] 
                # replacements = {'1':'a','14':'a','10':'t','11':'j','12':'q','13':'k'}
                # straight_numbers = [replacements.get(char,char) for char in straight_numbers]
                # cards_with_suit_stripped = [card[0] for card in current_player_hand]
                # for card in current_player_hand:
                #     if card[0] not in straight_numbers or cards_with_suit_stripped.count(card[0])==2:
                #         current_player_hand.remove(card)
                # behind_ahead = 0
                # behind_tie = 0
                # tie_ahead = 0
                # behind_sum = 0
                # tie_sum = 0
                # # print(current_hand_value)
                # for current_opponent_hand in combinations(remaining_deck,5):
                #     # print('h')
                #     opponent_hand_value = self.poker_hand_value[''.join(sorted(current_opponent_hand))]                 
                #     remaining_deck_for_draw = remaining_deck_set-set(current_opponent_hand)
                #     for drawed_cards in combinations(remaining_deck_for_draw,1):
                #         new_hand_value = self.poker_hand_value[''.join(sorted(current_player_hand+list(drawed_cards)))]
                #         if new_hand_value>opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_ahead+=1
                #             behind_sum+=1
                #         elif new_hand_value==opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_tie+=1
                #             behind_sum+=1
                #         elif new_hand_value>opponent_hand_value and current_hand_value==opponent_hand_value:
                #             tie_ahead+=1
                #             tie_sum+=1
                #         elif current_hand_value<opponent_hand_value:
                #             behind_sum+=1
                #         elif current_hand_value==opponent_hand_value:
                #             tie_sum+=1
                # print((behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5))
                # return (behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5)    
            elif hand_info['handName']=="one pair":
                # aakqt - 0.217
                # aa28k - 0.2204
                # 2243k - 0.2653
                # 22a3k - 0.264
                # 7743k - 0.256
                # 77kqt - 0.253
                # print('one pair')
                baseline = 0.2653
                double_card = None
                cards_with_suit_stripped = [card[0] for card in current_player_hand]
                for card in cards_with_suit_stripped:
                    if cards_with_suit_stripped.count(card)==2:
                        double_card = card
                if double_card=='2':
                    return baseline
                elif double_card=='3':
                    return baseline - 9/1750
                elif double_card=='4':
                    return baseline - (9/1750)*2
                elif double_card=='5':
                    return baseline - (9/1750)*3
                elif double_card=='6':
                    return baseline - (9/1750)*4
                elif double_card=='7':
                    return baseline - (9/1750)*5
                elif double_card=='8':
                    return baseline - (9/1750)*6
                elif double_card=='9':
                    return baseline - (9/1750)*7
                elif double_card=='t':
                    return baseline - (9/1750)*8
                elif double_card=='j':
                    return baseline - (9/1750)*9
                elif double_card=='q':
                    return baseline - (9/1750)*10
                elif double_card=='k':
                    return baseline - (9/1750)*11
                elif double_card=='a':
                    return baseline - (9/1750)*12
                # cards_with_suit_stripped = [card[0] for card in current_player_hand]
                # non_double_index = [i for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=2]
                # for card_index in sorted(non_double_index,reverse=True):
                #     del current_player_hand[card_index]
                # behind_ahead = 0
                # behind_tie = 0
                # tie_ahead = 0
                # behind_sum = 0
                # tie_sum = 0
                # random.shuffle(remaining_deck)
                # i = 0
                # # We take a sample as drawing five cards takes too long for entire 47 cards
                # for current_opponent_hand in combinations(remaining_deck[:35],5):
                #     opponent_hand_value = self.poker_hand_value[''.join(sorted(current_opponent_hand))]                 
                #     remaining_deck_for_draw = remaining_deck_set-set(current_opponent_hand)
                #     for drawed_cards in combinations(remaining_deck_for_draw,3):
                #         new_hand_value = self.poker_hand_value[''.join(sorted(current_player_hand+list(drawed_cards)))]
                #         if new_hand_value>opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_ahead+=1
                #             behind_sum+=1
                #         elif new_hand_value==opponent_hand_value and current_hand_value<opponent_hand_value:
                #             behind_tie+=1
                #             behind_sum+=1
                #         elif new_hand_value>opponent_hand_value and current_hand_value==opponent_hand_value:
                #             tie_ahead+=1
                #             tie_sum+=1
                #         elif current_hand_value<opponent_hand_value:
                #             behind_sum+=1
                #         elif current_hand_value==opponent_hand_value:
                #             tie_sum+=1
                # print((behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5))
                # return (behind_ahead+behind_tie*0.5+tie_ahead*0.5)/(behind_sum+tie_sum*0.5)
            # We do not return potential for pairs and high cards as it takes too long
            return 0

# loki = LokiBot()
# print(loki.simulate_hand_strength(['7d','7s','kh','qd','th']))
# print(loki.simulate_hand_strength(['kd','ts','jh','qd','2h']))
# loki.simulate_hand_potential(['7d','7s','4h','3d','kh'])

