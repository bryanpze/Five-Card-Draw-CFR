from hand_evaluator import Evaluator
from rule import Rule
from five_card_node_es import FiveCardNodeES
import re
import itertools
import sys
import time
import random
import numpy as np
import pymongo
from pymongo import MongoClient
sys.setrecursionlimit(1500)

# random.seed(7)
class MultiplyByTwoIterator:
    def __init__(self, max_value):
        self.current_value = 2
        self.max_value = max_value

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_value <= self.max_value:
            result = self.current_value
            self.current_value *= 2
            return result
        else:
            raise StopIteration

class ExternalSamplingHighSuitMini:
    def __init__(self,rule = Rule(bet_abstraction='raise'),update=False,linear_cfr = 1):
        self.update = update
        self.linear_cfr = linear_cfr
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
        # Bet actions must have 0:fold 1:call
        self.bet_actions = [0,1]
        # Other possible actions include all valid raises, in this case assume that max raise is equal to the sum of all player stacks
        # Here possible raises can only take integer values from 1 until sum of all player stacks
        # Value of 2 in this case is equal to a raise of 1
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
        self.cards = random.sample(self.deck,self.rule.num_players*5*2)
        self.node_map = {}
        self.history_map = {}
        # self.node_map = shelve.open('infoset_dict.db')
        self.evaluator = Evaluator()
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['fivecarddatabase']
        self.collection = self.db['fivecardcollection_abstraction_actual_big_perhaps']
        self.exploitability = [0,0]
        self.name = 'ExternalSampling Highest Suit Mini'

    def get_random_action(self,current_round,valid_actions,poker_cards,history):
        #Information translation
        infoset = self.parse_highest_suits_mini(''.join(sorted(poker_cards)+[':']))+history
        node = self.collection.find_one({'_id': infoset})
        if not node:
            node_strategy = np.repeat(1/len(valid_actions),len(valid_actions))
        else:        
            node['strategy_sum'] = np.array(node['strategy_sum'])
            node['regret_sum'] = np.array(node['regret_sum'])
            
            node_strategy = self.get_node_strategy(node['strategy_sum'],len(valid_actions))
        if current_round==1:
            actual_strategy = np.zeros(self.num_draw_actions)
            actual_strategy[valid_actions] = node_strategy
            sampled_action = self.get_actions(actual_strategy,'draw')
        else:
            actual_strategy = np.zeros(self.num_bet_actions)
            actual_strategy[valid_actions] = node_strategy
            sampled_action = self.get_actions(actual_strategy,'bet')
        return sampled_action

    def update_strategy_sum(self,node_infoset,node_strategy,action_probs):
        node_strategy=node_strategy+action_probs
        self.collection.update_one({'_id':node_infoset},{'$set':{'strategy_sum':node_strategy.tolist()}})
    
    def update_regret_sum(self,node_infoset,node_regret_sum_update):
        self.collection.update_one({'_id':node_infoset},{'$set':{'regret_sum':node_regret_sum_update.tolist()}})

    def get_node_strategy(self,node_regret_sum,num_valid_actions):
        regret_sum = np.array(node_regret_sum)
        regret_sum[regret_sum<0] = 0
        normalizing_sum = sum(regret_sum)
        if normalizing_sum>0:
            return regret_sum/normalizing_sum
        else:
            return np.repeat(1/num_valid_actions,num_valid_actions)

    def get_average_node_strategy(self,node_strategy_sum,num_valid_actions):
        node_strategy_sum = np.array(node_strategy_sum)
        normalizing_sum = sum(node_strategy_sum)
        if normalizing_sum>0:
            return node_strategy_sum/normalizing_sum
        else:
            return np.repeat(1/num_valid_actions,num_valid_actions)

    def generate_documents(self):
        for key,value in self.node_map.items():
            yield{
                '_id':key,
                'strategy_sum':value.strategy_sum.tolist(),
                'regret_sum':value.regret_sum.tolist()
            }
    def train(self,num_iterations = 10000000):
        # for card in ['as','4h','4s','4c','4d']:
        #     self.deck.remove(card)
        expected_value = 0
        for i in range(num_iterations):
            # Ten cards are needed for each player (5 initial hole cards, 5 additional drawing cards)
            number_of_cards_needed = self.rule.num_players*5*2
            self.cards = random.sample(self.deck,number_of_cards_needed)        
            # self.cards[0:5] = ['as','4h','4s','4c','4d']
            for j in range(self.rule.num_players):
                self.cards[10*j:10*j+5] = sorted(self.cards[10*j:10*j+5])
            tmp = self.cards
            expected_value+=(self.cfr('','',0))
            self.cards = tmp
            expected_value+=(self.cfr('','',1))
            if(i+1)%1000==0:
                print(f'Iteration {i+1}: Exepcted Value: {expected_value/1000}')
                expected_value = 0
                # print(f"First one {time.time() - start_time}")
                # self.cfr('c','c',0)
                # self.cfr('c','c',1)
                # self.cfr(f'r{self.rule.blinds[1]*2}',0)
                # self.cfr(f'r{self.rule.blinds[1]*2}',1)
                # for j in range(self.rule.blinds[1]*2,self.rule.stack[0]):
                #     self.cfr(f'r{j}',f'r{j}',1)
            # if i%50==0:
            #     my_shelf =  shelve.open('infoset.db','c')
            #     my_shelf.update(self.node_map)
            #     my_shelf.close()
            # print("--- %s seconds ---" % (time.time() - start_time))
    def calc_payoff(self,parsed_history):
        # Terminal state, everyone but one has folded
        if (parsed_history['players_remaining']==1 and parsed_history['round_ended']):
            payoff = [0]*self.rule.num_players
            winner = set(range(self.rule.num_players))-parsed_history['folded_players']
            assert(len(winner)==1)
            winner = list(winner)[0]
            payoff[winner] = parsed_history['potsize']
            player_total_stakes = [x + y for x,y in zip(parsed_history['player_total_stakes'],parsed_history['player_current_stakes'])]
            for i in range(self.rule.num_players):
                payoff[i]-=player_total_stakes[i]
        # Terminal state, showdown between two or more
        elif (parsed_history['current_round']==2 and parsed_history['round_ended']):
            payoff = [0]*self.rule.num_players
            card_strength = [0]*self.rule.num_players
            unfolded_players = set(range(self.rule.num_players))-parsed_history['folded_players']
            for player in unfolded_players:
                card_strength[player] = self.evaluator.eval_hand(self.cards[player*10:player*10+5])['value']
            winner = np.argmax(card_strength)
            payoff[winner] = parsed_history['potsize']
            player_total_stakes = [x + y for x,y in zip(parsed_history['player_total_stakes'],parsed_history['player_current_stakes'])]
            for i in range(self.rule.num_players):
                payoff[i]-=player_total_stakes[i]
        # print(parsed_history['payoff'][player])
        return payoff

    def cfr(self,history, opp_history, player):
        if history in self.history_map:
            parsed_history = self.history_map[history]
        else:
            parsed_history = self.parse_history(history)
            self.history_map[history] = parsed_history
        # Check if terminal
        if len(history)>100:
            print(history)
            raise ValueError("Infoset has been running for too long")
        if 'terminal' in parsed_history:
            payoff = self.calc_payoff(parsed_history)
            return payoff[player]
        # Check if new round
        if parsed_history['round_ended']==True:
            history = history+'/'
            opp_history = opp_history+'/'
            parsed_history = self.parse_history(history)
        if self.rule.card_abstraction=='No Cards':
            infoset = history
        elif self.rule.card_abstraction=='No Suits':
            infoset = self.parse_no_suits(''.join(sorted(self.cards[parsed_history['current_player']*10:parsed_history['current_player']*10+5])+[':']))+history
        elif self.rule.card_abstraction=='Highest Suit Mini':
            infoset = self.parse_highest_suits_mini(''.join(sorted(self.cards[parsed_history['current_player']*10:parsed_history['current_player']*10+5])+[':']))+history
        else:
            infoset = ''.join(sorted(self.cards[parsed_history['current_player']*10:parsed_history['current_player']*10+5])+[':'])+history
        current_round = parsed_history['current_round']
        current_player = parsed_history['current_player']
        valid_actions = parsed_history['valid_actions']
        num_valid_actions = len(valid_actions)
        if self.update:
            node = self.collection.find_one({'_id': infoset})
            if not node:
                node = {'_id':infoset,'strategy_sum':num_valid_actions*[0],'regret_sum':num_valid_actions*[0]}
                self.collection.insert_one(node)
            node['strategy_sum'] = np.array(node['strategy_sum'])
            node['regret_sum'] = np.array(node['regret_sum'])
            node_strategy = self.get_node_strategy(node['regret_sum'],num_valid_actions)
        else:
            if infoset in self.node_map:
                node = self.node_map[infoset]
            else:
                node = FiveCardNodeES(infoset,num_valid_actions)
                self.node_map[infoset]=node

            node_strategy = node.get_strategy()
        if current_round==1:
            # Drawing actions are semi-private information, when the opponent draws you only know the number of cards drawn, when the player draws, he knows what he is disposing
            if self.rule.num_players==2 and current_player!=player:
                actual_strategy = np.zeros(self.num_draw_actions)
                actual_strategy[valid_actions] = node_strategy
                sampled_action = self.get_actions(actual_strategy,'draw')
                if self.update:
                    self.update_strategy_sum(node['_id'],node['strategy_sum'],node_strategy*self.linear_cfr)
                else:    
                    node.update_strategy(node_strategy*self.linear_cfr)
                sampled_action_repr = 'd'+str(sampled_action)
                what_the_opponent_sees = len(self.draw_actions_lookup[sampled_action])
                what_the_opponent_sees = 'b'+str(what_the_opponent_sees)
                # Update draw action in cards
                self.update_draw_action(current_player,self.draw_actions_lookup[sampled_action])
                return self.cfr(opp_history+what_the_opponent_sees,history+sampled_action_repr,player)
        else:
            # Bet actions Are public information, can be handled the same way for each player
            if self.rule.num_players==2 and current_player!=player:
                actual_strategy = np.zeros(self.num_bet_actions)
                actual_strategy[valid_actions] = node_strategy
                sampled_action = self.get_actions(actual_strategy,'bet')
                if self.update:
                    self.update_strategy_sum(node['_id'],node['strategy_sum'],node_strategy*self.linear_cfr)
                else:      
                    node.update_strategy(node_strategy*self.linear_cfr)
                if sampled_action == 0:
                    sampled_action = 'f'
                elif sampled_action == 1:
                    sampled_action = 'c'
                elif sampled_action > 1:
                    sampled_action = 'r'+str(sampled_action-1)
                what_the_opponent_sees = sampled_action
                return self.cfr(opp_history+what_the_opponent_sees,history+sampled_action,player)        

        action_utils = np.zeros(num_valid_actions)
        for index,action in enumerate(valid_actions):
            counterfactual_action, what_the_opponent_sees = self.convert_action_to_representation(action,current_round)
            # Update draw action if draw action
            if counterfactual_action[0]=='d':
                self.update_draw_action(current_player,self.draw_actions_lookup[action])
            action_utils[index] = self.cfr(opp_history+what_the_opponent_sees,history+counterfactual_action,player)
            # Revert draw action to original deck
            if counterfactual_action[0]=='d':
                self.update_draw_action(current_player,self.draw_actions_lookup[action])
        util = np.sum(action_utils*node_strategy)
        if self.update:
            self.update_regret_sum(node['_id'],node['regret_sum']+(action_utils-util)*self.linear_cfr)
        else:    
            node.regret_sum += (action_utils-util)*self.linear_cfr

        return util
    def parse_highest_suits_mini(self,card_string):
        suit_string = re.sub(r'[^cdsh]','',card_string)
        suit_frequency = {}
        for char in suit_string:
            suit_frequency[char] = suit_frequency.get(char,0)+1
        most_frequent_suit = max(suit_frequency,key=suit_frequency.get)
        if suit_frequency[most_frequent_suit]>=4:
            additional_info = ''.join(['1' if char==most_frequent_suit else '0' for char in suit_string])
            return additional_info+'|'+re.sub(r'[^atjqk0-9:]','',card_string)
        else:
            return re.sub(r'[^atjqk0-9:]','',card_string)
    def parse_no_suits(self,card_string):
        return re.sub(r'[^atjqk0-9:]','',card_string)
    def convert_action_to_representation(self,action,round):
        if round==1:
            action_repr = 'd'+str(action)
            what_the_opponent_sees = len(self.draw_actions_lookup[action])
            what_the_opponent_sees = 'b'+str(what_the_opponent_sees)
            return [action_repr,what_the_opponent_sees]
        else:
            if action == 0:
                action = 'f'
            elif action == 1:
                action = 'c'
            elif action > 1:
                action = 'r'+str(action-1)
            what_the_opponent_sees = action
            return [action,what_the_opponent_sees]

    def get_actions(self,strategy,round_type):
        r = random.random()
        index = 0
        while(r >= 0 and index < len(strategy)):
            r -= strategy[index]
            index += 1
        if round_type=='draw':
            return self.draw_actions[index-1]
            # return random.choices(self.draw_actions,weights=strategy,k=1)[0]
        elif round_type=='bet':
            return self.bet_actions[index-1]
            # return random.choices(self.bet_actions,weights=strategy,k=1)[0]

    def get_round(self,infoset):
        # Round 0: first betting round, Round 1: drawing round, Round 2: second betting round
        rounds = infoset.split('/')
        return len(rounds)-1
        
    def update_draw_action(self,player,draw_cards):
        player_actual_card_start_index = player*10
        player_chance_card_start_index = player_actual_card_start_index+5
        for draw_here in draw_cards:
            self.cards[player_actual_card_start_index+draw_here],self.cards[player_chance_card_start_index+draw_here] = self.cards[player_chance_card_start_index+draw_here],self.cards[player_actual_card_start_index+draw_here]

    # This function should only be called when there is still a player playing
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

# to_update = True
# linear_cfr = 5
# trainer = ExternalSamplingHighSuitMini(update=to_update,linear_cfr=linear_cfr,rule = Rule(bet_abstraction='raise',card_abstraction='Highest Suit Mini'))

# trainer.train(10000)

# # trainer.train(100000)
# if to_update==False:
#     client = MongoClient('localhost', 27017)
#     db = client['fivecarddatabase']
#     collection = db['fivecardcollection_abstraction_actual_big_perhaps']
#     for key, value in trainer.node_map.items():        
#         try:
#             # collection.insert_many(trainer.generate_documents(), ordered=False)
#             collection.insert_one({
#                 '_id':key,
#                 'strategy_sum':value.strategy_sum.tolist(),
#                 'regret_sum':value.regret_sum.tolist()
#             })
#         except pymongo.errors.DuplicateKeyError as e:
#             pass
