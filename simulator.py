import re
from random_bot import RandomBot
import random
from rule import Rule
import itertools
import numpy as np
from hand_evaluator import Evaluator
import csv
from heuristics import HeuristicsBot
from article_bot import ArticleBot
from external_sampling_no_suit import ExternalSamplingNoSuit
from external_sampling_highest_suit_mini import ExternalSamplingHighSuitMini
from collections import defaultdict
import sys
import scipy.stats as stats

sys.setrecursionlimit(1500)

# Uses 888 seed
random.seed(9)
class Simulator:
    def __init__(self,num_iterations = 100,bot_0 =ExternalSamplingNoSuit(constrained_action=True),bot_1 = HeuristicsBot(agent_modeling_flag = True),rule = Rule(bet_abstraction='raise')):
        self.num_iterations = num_iterations
        self.rule = rule
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
        self.cards = random.sample(self.deck,self.rule.num_players*5*2)
        self.bot_0 = bot_0
        self.bot_1 = bot_1
        self.evaluator = Evaluator()

    def simulate(self):
        small_blind_payoffs = []
        big_blind_payoffs = []
        bot_1_payoffs = []
        bot_0_name = self.bot_0.name
        bot_1_name = self.bot_1.name
        bot_0_starting_hand = defaultdict(list)
        bot_1_starting_hand = defaultdict(list)
        for i in range(self.num_iterations):
            number_of_cards_needed = self.rule.num_players*5*2
            self.cards = random.sample(self.deck,number_of_cards_needed)
            for j in range(self.rule.num_players):
                self.cards[10*j:10*j+5] = sorted(self.cards[10*j:10*j+5])
            tmp = self.cards
            bot_1_hand = self.evaluator.eval_hand(self.cards[0:5])['handName']
            bot_2_hand = self.evaluator.eval_hand(self.cards[10:15])['handName']
            bot_1_payoff = self.play_round('','')
            bot_0_starting_hand[bot_1_hand].append(bot_1_payoff)
            bot_1_starting_hand[bot_2_hand].append(-bot_1_payoff)
            # Bot 1 Bot 2 switch cards and seating arrangements
            self.bot_0, self.bot_1 = self.bot_1,self.bot_0
            self.cards = tmp
            bot_2_hand = self.evaluator.eval_hand(self.cards[0:5])['handName']
            bot_1_hand = self.evaluator.eval_hand(self.cards[10:15])['handName']
            bot_2_payoff = self.play_round('','')
            bot_0_starting_hand[bot_1_hand].append(-bot_2_payoff)
            bot_1_starting_hand[bot_2_hand].append(bot_2_payoff)
            # Payoff when small blind
            small_blind_payoffs.append(bot_1_payoff)
            # Payoff when big blind (the negated payoff of the small blind)
            big_blind_payoffs.append(-bot_2_payoff)
            bot_1_payoffs.append(bot_1_payoff-bot_2_payoff)
            # Switch back
            self.bot_0,self.bot_1 = self.bot_1,self.bot_0
            if (i%100)==0:
                print(sum(bot_1_payoffs))
        print('------bot-1------')
        for hand in bot_0_starting_hand:
            print(f'Hand Type: {hand}')
            print(f'Number of Hands: {len(bot_0_starting_hand[hand])}')
            print(f'Number of Wins: {len([num for num in bot_0_starting_hand[hand] if num >= 0])}')
            print(f'Win Percentage: {len([num for num in bot_0_starting_hand[hand] if num >= 0])/len(bot_0_starting_hand[hand])}')
            print(f'Expected Payoff: {sum(bot_0_starting_hand[hand])/len(bot_0_starting_hand[hand])}')
            # print(f'P-value (profitable): {stats.ttest_1samp(bot_0_starting_hand[hand], popmean = 0,alternative='greater')}')
            # print(f'T-test (sample average is greater): {stats.ttest_ind(bot_0_starting_hand[hand], bot_1_starting_hand[hand], equal_var=False,alternative='greater')}')
        print('------bot-2------')
        for hand in bot_1_starting_hand:
            print(f'Hand Type: {hand}')
            print(f'Number of Hands: {len(bot_1_starting_hand[hand])}')
            print(f'Number of Wins: {len([num for num in bot_1_starting_hand[hand] if num >= 0])}')
            print(f'Win Percentage: {len([num for num in bot_1_starting_hand[hand] if num >= 0])/len(bot_1_starting_hand[hand])}')
            print(f'Expected Payoff: {sum(bot_1_starting_hand[hand])/len(bot_1_starting_hand[hand])}')
            # print(f'P-value: {stats.ttest_1samp(bot_1_starting_hand[hand], popmean = 0,alternative='greater')}')
        with open(f'./simulation_results/Small Blind Payoff of {bot_0_name} - {bot_0_name} V {bot_1_name}.csv','w',newline='') as f:
            writer = csv.writer(f)
            for val in small_blind_payoffs:
                writer.writerow([val])
        with open(f'./simulation_results/Big Blind Payoff of {bot_0_name} - {bot_0_name} V {bot_1_name}.csv','w',newline='') as f:
            writer = csv.writer(f)
            for val in big_blind_payoffs:
                writer.writerow([val])

        with open(f'./simulation_results/Average Payoff of {bot_0_name} - {bot_0_name} V {bot_1_name}.csv','w',newline='') as f:
            writer = csv.writer(f)
            for val in bot_1_payoffs:
                writer.writerow([val])
    def update_draw_action(self,player,draw_cards):
        player_actual_card_start_index = player*10
        player_chance_card_start_index = player_actual_card_start_index+5
        for draw_here in draw_cards:
            self.cards[player_actual_card_start_index+draw_here],self.cards[player_chance_card_start_index+draw_here] = self.cards[player_chance_card_start_index+draw_here],self.cards[player_actual_card_start_index+draw_here]

    def play_round(self,history, opp_history):
        parsed_history = self.parse_history(history)
        # print(parsed_history)
        if 'payoff' in parsed_history:
            return parsed_history['payoff'][0]
        if parsed_history['round_ended']==True:
            history = history+'/'
            opp_history = opp_history+'/'
            parsed_history = self.parse_history(history)
        current_round = parsed_history['current_round']
        valid_actions = parsed_history['valid_actions']
        current_player = parsed_history['current_player']
        if current_player==0:
            player_action = self.bot_0.get_random_action(current_round,valid_actions,self.cards[parsed_history['current_player']*10:parsed_history['current_player']*10+5],history)
        elif current_player==1:
            player_action = self.bot_1.get_random_action(current_round,valid_actions,self.cards[parsed_history['current_player']*10:parsed_history['current_player']*10+5],history)
        if current_round==1:
            sampled_action_repr = 'd'+str(player_action)
            what_the_opponent_sees = len(self.draw_actions_lookup[player_action])
            what_the_opponent_sees = 'b'+str(what_the_opponent_sees)
            # Update draw action in cards
            self.update_draw_action(current_player,self.draw_actions_lookup[player_action])
            return self.play_round(opp_history+what_the_opponent_sees,history+sampled_action_repr)
        else:
            if player_action == 0:
                player_action = 'f'
            elif player_action == 1:
                player_action = 'c'
            elif player_action > 1:
                player_action = 'r'+str(player_action-1)
            what_the_opponent_sees = player_action
            return self.play_round(opp_history+what_the_opponent_sees,history+player_action)

    def get_round(self,infoset):
        # Round 0: first betting round, Round 1: drawing round, Round 2: second betting round
        rounds = infoset.split('/')
        return len(rounds)-1

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
            # print(history)
            # print(valid_actions)
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
        # Terminal state, everyone but one has folded
        if (players_remaining==1 and round_ended):
            payoff = [0]*self.rule.num_players
            winner = set(range(self.rule.num_players))-folded_players
            assert(len(winner)==1)
            winner = list(winner)[0]
            payoff[winner] = potsize
            player_total_stakes = [x + y for x,y in zip(player_total_stakes,player_current_stakes)]
            for i in range(self.rule.num_players):
                payoff[i]-=player_total_stakes[i]
            return {'payoff':payoff}              
        # Terminal state, showdown between two or more
        if (current_round==2 and round_ended):
            payoff = [0]*self.rule.num_players
            card_strength = [0]*self.rule.num_players
            unfolded_players = set(range(self.rule.num_players))-folded_players
            for player in unfolded_players:
                card_strength[player] = self.evaluator.eval_hand(self.cards[player*10:player*10+5])['value']
            winner = np.argmax(card_strength)
            payoff[winner] = potsize
            player_total_stakes = [x + y for x,y in zip(player_total_stakes,player_current_stakes)]
            for i in range(self.rule.num_players):
                payoff[i]-=player_total_stakes[i]
            return {'payoff':payoff}              
        return {'potsize':potsize,'current_round':current_round,'current_player':current_player,'valid_actions':valid_actions,'folded_players':folded_players,'round_ended':round_ended}

Simulator(1000).simulate()