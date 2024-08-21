import numpy as np
import random
from itertools import combinations
from hand_evaluator import Evaluator
from rule import Rule

class ArticleBot:
    def __init__(self,rule = Rule(),conservative = False):
        self.rule = rule
        self.conservative = conservative
        self.bet_actions = [0,1]
        for possible_raise in range(2,sum(self.rule.stack)+1):
            self.bet_actions.append(possible_raise)
        self.num_bet_actions = len(self.bet_actions)
        tmp = []
        tmp.append([])
        for i in range(1,6):
            tmp.extend([list(x) for x in list(combinations([0,1,2,3,4],i))])
        self.draw_actions_lookup = {}
        for index,val in enumerate(tmp):
            self.draw_actions_lookup[index] = val
        self.draw_actions = list(self.draw_actions_lookup.keys())
        self.num_draw_actions = len(self.draw_actions)
        self.evaluator = Evaluator()
        self.name = "Article"
        if self.conservative:
            self.name = "Article Conservative"
        self.hand_info = None
        self.potential_flush = None
        self.potential_straight = None
        self.royal_cat_hop = None    
        self.trip_strategy_bet = None
        self.trip_strategy_draw = None
        self.eight_kicker = None
    def check_potential_flush(self,current_player_hand):
        cards_with_number_stripped = [card[1] for card in current_player_hand] 
        for card_index, card in enumerate(cards_with_number_stripped):
            if cards_with_number_stripped.count(card)==4:
                return True
        return False
    def check_royal_cat_hop(self,current_player_hand):
        for potential_hand in combinations(["ad","kd","td","qd","jd"],3):
            if set(potential_hand).issubset(current_player_hand):
                return True
        for potential_hand in combinations(["as","ks","ts","qs","js"],3):
            if set(potential_hand).issubset(current_player_hand):
                return True
        for potential_hand in combinations(["as","ks","ts","qs","js"],3):
            if set(potential_hand).issubset(current_player_hand):
                return True
        for potential_hand in combinations(["as","ks","ts","qs","js"],3):
            if set(potential_hand).issubset(current_player_hand):
                return True
    def check_potential_straight(self,current_player_hand,starting_number=False):
        cards_with_suit_stripped = [card[0] for card in current_player_hand]
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
        if starting_number:
            return starting_number
        return consecutive_numbers==3
    def check_duplicate_value(self,current_player_hand,num_duplicate):
        cards_with_suit_stripped = [card[0] for card in current_player_hand]
        for card in cards_with_suit_stripped:
            if cards_with_suit_stripped.count(card)==num_duplicate:
                return card
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
            
    def get_random_action(self, round,valid_actions,poker_cards,history):
        if (history=='c' or history=='r4' or history==''):
            self.hand_info = self.evaluator.eval_hand(poker_cards)
            self.potential_flush = self.check_potential_flush(poker_cards)
            self.potential_straight = self.check_potential_straight(poker_cards)
            self.royal_cat_hop = self.check_royal_cat_hop(poker_cards)
            # Reset previous set flags
            self.trip_strategy_bet = None
            self.trip_strategy_draw = None
            self.eight_kicker = None
            self.num_draw = None
        if round==0:
            actual_strategy = np.zeros(self.num_bet_actions)
            valid_action_strategy = np.zeros(len(valid_actions))
            smallest_pair_value = 8193
            if (self.hand_info['value']<smallest_pair_value) and (not  self.royal_cat_hop ) and (not self.potential_flush) and (not self.potential_straight):
                val = self.check_if_no_bet_otherwise_fold(history)
                valid_action_strategy[val] = 1
            else:
                if self.hand_info['handName']=='full house':
                    val =  2 if (len(valid_actions)-2)>0 else 1
                    if self.conservative:
                        if history.count('r')>=2:
                            val = 1
                    valid_action_strategy[val] = 1
                elif self.hand_info['handName']=='straight':
                    val =  2 if (len(valid_actions)-2)>0 else 1
                    if self.conservative:
                        if history.count('r')>=2:
                            val = 1
                    valid_action_strategy[val] = 1
                elif self.hand_info['handName']=='flush':
                    val =  2 if (len(valid_actions)-2)>0 else 1
                    if self.conservative:
                        if history.count('r')>=2:
                            val = 1
                    valid_action_strategy[val] = 1
                elif self.hand_info['handName']=='straight flush':
                    val =  2 if (len(valid_actions)-2)>0 else 1
                    if self.conservative:
                        if history.count('r')>=2:
                            val = 1
                    valid_action_strategy[val] = 1
                elif self.hand_info['handName']=='two pairs':
                    # Call with high 2 pair, raise with medium or low 2 pair:
                    first_two_pair_value = self.check_duplicate_value(poker_cards,2)
                    second_two_pair_value = self.check_duplicate_value(poker_cards[::-1],2)
                    high_values = ['q','k','a']
                    if first_two_pair_value in high_values or second_two_pair_value in high_values:
                        valid_action_strategy[1] = 1
                    else:
                        val =  2 if (len(valid_actions)-2)>0 else 1
                        if self.conservative:
                            if history.count('r')>=2:
                                val = 1
                        valid_action_strategy[val] = 1
                elif self.hand_info['handName']=='three of a kind':
                    trip_value = self.check_duplicate_value(poker_cards,3)
                    if trip_value in ['8','9','t','j','q','k','a']:
                        # 1/3 of the time draw one card, 1/3 stand pat, 1/3 draw 2
                        self.trip_strategy_draw = np.random.choice([0,1,2],p=[1/3,1/3,1/3])
                        # in cases where stand pat or draw 1, raise in 1/2 of the time
                        if self.trip_strategy_draw<2:
                            val = np.random.choice([1,2],p=[0.5,0.5])
                            if val==2:
                                self.trip_strategy_bet = 1
                            else:
                                self.trip_strategy_bet = 0
                            if (len(valid_actions)-2)>0:
                                if self.conservative:
                                    if history.count('r')>=2:
                                        val = 1
                                valid_action_strategy[val] = 1
                            else:
                                valid_action_strategy[1] = 1
                        else:
                            self.trip_strategy_bet = 0
                            valid_action_strategy[1]=1
                    else:
                        # low trip
                        self.trip_strategy_bet = 0
                        self.trip_strategy_draw = 1
                        valid_action_strategy[1] = 1
                elif self.hand_info['handName']=="one pair":
                    pair_value = self.check_duplicate_value(poker_cards,2)
                    if pair_value in ['a','q','k']:
                        # Generally raise, premium pairs, randomise between 3/4 raising 1/4 calling
                        val = np.random.choice([1,2],p=[0.25,0.75])
                        if(len(valid_actions)-2)>0:
                            if self.conservative:
                                if history.count('r')>=2:
                                    val = 1
                            valid_action_strategy[val] = 1
                        else:
                            valid_action_strategy[1]=1
                    elif pair_value=='j':
                        valid_action_strategy[1] = 1
                    else:
                        # check if need to break up to pursue flush or straight
                        if (self.potential_straight and self.potential_flush):
                            val =  2 if (len(valid_actions)-2)>0 else 1
                            if self.conservative:
                                if history.count('r')>=2:
                                    val = 1
                            valid_action_strategy[val] = 1
                        elif (pair_value in ['2','3','4','5','6','7','8','9','t']) and self.potential_flush:
                            valid_action_strategy[1] = 1
                        elif (pair_value in ['2','3','4','5','6','7','8','9']) and self.potential_straight:
                            valid_action_strategy[1] = 1
                        else:
                            cards_with_suit_stripped = [card[0] for card in poker_cards]
                            if pair_value=='t':
                                for x in cards_with_suit_stripped:
                                    if x in ['j','q','k','a']:
                                        valid_action_strategy[1] = 1
                                val =  self.check_if_no_bet_otherwise_fold(history)
                                valid_action_strategy[val] = 1
                            elif pair_value=='9':
                                counter = 0
                                for x in cards_with_suit_stripped:
                                    if x=='a':
                                        counter+=2
                                    elif x in ['t','j','q','k']:
                                        counter+=1
                                val =  1 if counter>=2 else self.check_if_no_bet_otherwise_fold(history)
                                valid_action_strategy[val] = 1
                            elif pair_value=='8':
                                counter = 0
                                ace_condition = False
                                for x in cards_with_suit_stripped:
                                    if x=='a':
                                        counter+=2
                                        ace_condition = True
                                    elif x in ['9','t','j','q','k']:
                                        counter+=1
                                    if counter<3 and ace_condition:
                                        self.eight_kicker=True
                                val =  1 if (counter>=3 or ace_condition==True) else self.check_if_no_bet_otherwise_fold(history)
                                valid_action_strategy[val] = 1
                            else:
                                if 'a' in cards_with_suit_stripped:
                                    valid_action_strategy[1]=1
                                else:
                                    valid_action_strategy[0] = 1
                    
                elif  self.hand_info['handName']=='four of a kind':
                    # bet minimum amount
                    if 'r' in history:
                        valid_action_strategy[1] = 1
                    else: 
                        val = 2 if (len(valid_actions)-2)>0 else 1
                        if self.conservative:
                            if history.count('r')>=2:
                                val = 1
                        valid_action_strategy[val] = 1
                
                elif self.potential_flush:
                    valid_action_strategy[1] = 1
                elif self.potential_straight:
                    valid_action_strategy[1] = 1
                elif self.royal_cat_hop:
                    valid_action_strategy[1] = 1
                else:
                    val = self.check_if_no_bet_otherwise_fold(history)
                    valid_action_strategy[val] = 1
            actual_strategy[valid_actions] = valid_action_strategy
            return random.choices(self.bet_actions,weights=actual_strategy,k=1)[0]
        elif round==1:
            if self.hand_info['handName'] in ['straight flush','flush','full house','straight']:
                self.num_draw = 0
                return 0 

            elif self.hand_info['handName']=="one pair":
                pair_value = self.check_duplicate_value(poker_cards,2)
                if pair_value in ['a','q','k','j']:
                    # draw three cards
                    cards_with_suit_stripped = [card[0] for card in poker_cards]
                    non_pair_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=2])
                    action = None
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==non_pair_index:
                            action = key
                    self.num_draw = 3
                    return action
                elif (self.potential_straight and self.potential_flush):
                    cards_with_number_stripped = [card[1] for card in poker_cards] 
                    non_flush_index = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            non_flush_index = card_index
                    action = None
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==[non_flush_index]:
                            action = key
                    self.num_draw = 1
                    return action
                elif (pair_value in ['2','3','4','5','6','7','8','9','t']) and self.potential_flush:
                    cards_with_number_stripped = [card[1] for card in poker_cards] 
                    non_flush_index = None
                    for card_index, card in enumerate(cards_with_number_stripped):
                        if cards_with_number_stripped.count(card)==1:
                            non_flush_index = card_index
                    action = None
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==[non_flush_index]:
                            action = key
                    self.num_draw = 1
                    return action                
                elif (pair_value in ['2','3','4','5','6','7','8','9']) and self.potential_straight:
                    starting_number = self.check_potential_straight(poker_cards,True)
                    straight_numbers =  [str(starting_number),str(starting_number+1),str(starting_number+2),str(starting_number+3),str(starting_number+4)] 
                    replacements = {'1':'a','14':'a','10':'t','11':'j','12':'q','13':'k'}
                    straight_numbers = [replacements.get(char,char) for char in straight_numbers]
                    non_straight_index = None
                    cards_with_suit_stripped = [card[0] for card in poker_cards]
                    for index,card in enumerate(poker_cards):
                        if card[0] not in straight_numbers or cards_with_suit_stripped.count(card[0])==2:
                            non_straight_index = index
                    action = None
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==[non_straight_index]:
                            action = key
                    self.num_draw = 1
                    return action
                elif pair_value in ['9','10']:
                    # draw three cards
                    cards_with_suit_stripped = [card[0] for card in poker_cards]
                    non_pair_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=2])
                    action = None
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==non_pair_index:
                            action = key
                    self.num_draw = 3
                    return action

                elif pair_value=='8':
                    if self.eight_kicker:
                        cards_with_suit_stripped = [card[0] for card in poker_cards]
                        # leave ace as kicker
                        disposable_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if ((cards_with_suit_stripped.count(card)!=2) or card!='a')])
                        action = None
                        for key in self.draw_actions_lookup:
                            if self.draw_actions_lookup[key]==disposable_index:
                                action = key
                        self.num_draw = 2
                        return action
                    else:
                        cards_with_suit_stripped = [card[0] for card in poker_cards]
                        non_pair_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=2])
                        action = None
                        for key in self.draw_actions_lookup:
                            if self.draw_actions_lookup[key]==non_pair_index:
                                action = key
                        self.num_draw = 3
                        return action
                else:
                    cards_with_suit_stripped = [card[0] for card in poker_cards]
                    if 'a' in cards_with_suit_stripped:
                        disposable_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if ((cards_with_suit_stripped.count(card)!=2) or card!='a')])
                        action = None
                        for key in self.draw_actions_lookup:
                            if self.draw_actions_lookup[key]==disposable_index:
                                action = key
                        self.num_draw = 2
                        return action
                    else:
                        non_pair_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=2])
                        action = None
                        for key in self.draw_actions_lookup:
                            if self.draw_actions_lookup[key]==non_pair_index:
                                action = key
                        self.num_draw = 3
                        return action
            elif  self.hand_info['handName']=='four of a kind':
                cards_with_suit_stripped = [card[0] for card in poker_cards]
                non_quad_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=4])
                action = None
                for key in self.draw_actions_lookup:
                    if self.draw_actions_lookup[key]==non_quad_index:
                        action = key
                self.num_draw = 1
                return action
            elif self.hand_info['handName']=='three of a kind':
                cards_with_suit_stripped = [card[0] for card in poker_cards]
                non_trip_index = sorted([int(i) for i,card in enumerate(cards_with_suit_stripped) if cards_with_suit_stripped.count(card)!=3])
                if self.trip_strategy_draw == 0:
                    self.num_draw = 0
                    return 0 
                elif self.trip_strategy_draw ==1:
                    action = None
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==[non_trip_index[0]]:
                            action = key
                    self.num_draw = 1
                    return action
                elif self.trip_strategy_draw==2:
                    action = None
                    for key in self.draw_actions_lookup:
                        if self.draw_actions_lookup[key]==[non_trip_index[0],non_trip_index[1]]:
                            action = key
                    self.num_draw = 2
                    return action
            elif self.potential_flush:
                cards_with_number_stripped = [card[1] for card in poker_cards] 
                non_flush_index = None
                for card_index, card in enumerate(cards_with_number_stripped):
                    if cards_with_number_stripped.count(card)==1:
                        non_flush_index = card_index
                action = None
                for key in self.draw_actions_lookup:
                    if self.draw_actions_lookup[key]==[non_flush_index]:
                        action = key
                self.num_draw = 1
                return action
            elif self.potential_straight:
                starting_number = self.check_potential_straight(poker_cards,True)
                straight_numbers =  [str(starting_number),str(starting_number+1),str(starting_number+2),str(starting_number+3),str(starting_number+4)] 
                replacements = {'1':'a','14':'a','10':'t','11':'j','12':'q','13':'k'}
                straight_numbers = [replacements.get(char,char) for char in straight_numbers]
                non_straight_index = None
                cards_with_suit_stripped = [card[0] for card in poker_cards]
                for index,card in enumerate(poker_cards):
                    if card[0] not in straight_numbers or cards_with_suit_stripped.count(card[0])==2:
                        non_straight_index = index
                action = None
                for key in self.draw_actions_lookup:
                    if self.draw_actions_lookup[key]==[non_straight_index]:
                        action = key
                self.num_draw = 1
                return action
            elif self.royal_cat_hop:
                action = None
                for key in self.draw_actions_lookup:
                    if self.draw_actions_lookup[key]==[0,1]:
                        action = key
                self.num_draw = 2
                return action
            else:
                drop_everything_not_high = [index for index,card in enumerate(poker_cards) if card[0] not in {'a','j','q','k'}]
                for key in self.draw_actions_lookup:
                    if self.draw_actions_lookup[key]==drop_everything_not_high:
                        action = key
                # technically could be 4 or 5
                self.num_draw = 4
                return action
        else:
            actual_strategy = np.zeros(self.num_bet_actions)
            valid_action_strategy = np.zeros(len(valid_actions))
            if self.num_draw==0:
                # Raise original pat hands
                val = 2 if (len(valid_actions)-2)>0 else 1
                if self.conservative:
                    if history.count('r')>=3:
                        val = 1
                valid_action_strategy[val] = 1
            else:
                new_hand_info = self.evaluator.eval_hand(poker_cards)
                # Raise new quads
                if new_hand_info['handName']=='four of a kind' and self.hand_info['handName']!='four of a kind':
                    val = 2 if (len(valid_actions)-2)>0 else 1
                    if self.conservative:
                        if history.count('r')>=3:
                            val = 1
                    valid_action_strategy[val] = 1
                # Raise new full house
                elif new_hand_info['handName']=='full house' and self.hand_info['handName']!='full house':
                    val =  2 if (len(valid_actions)-2)>0 else 1
                    if self.conservative:
                        if history.count('r')>=3:
                            val = 1
                    valid_action_strategy[val] = 1
                # Raise new straight flush
                elif new_hand_info['handName']=='straight flush' and self.hand_info['handName']!='straight flush':
                    val = 2 if (len(valid_actions)-2)>0 else 1
                    if self.conservative:
                        if history.count('r')>=3:
                            val = 1
                    valid_action_strategy[val] = 1
                elif self.hand_info['handName']=='three of a kind':
                    if self.trip_strategy_draw==1:
                        a = np.random.choice([1,2],p=[0.5,0.5])
                        if a==2:
                            val =  a if (len(valid_actions)-2)>0 else 1
                            if self.conservative:
                                if history.count('r')>=3:
                                    val = 1
                            valid_action_strategy[val] = 1
                        else:
                            valid_action_strategy[a] = 1
                    else:
                        valid_action_strategy[1] = 1
                elif new_hand_info['handName']=='two pairs':
                    # Call with high 2 pair, raise with medium or low 2 pair:
                    first_two_pair_value = self.check_duplicate_value(poker_cards,2)
                    second_two_pair_value = self.check_duplicate_value(poker_cards[::-1],2)
                    high_values = ['q','k','a']
                    if first_two_pair_value in high_values or second_two_pair_value in high_values:
                        valid_action_strategy[1] = 1
                    else:
                        val =  2 if (len(valid_actions)-2)>0 else 1
                        if self.conservative:
                            if history.count('r')>=3:
                                val = 1
                        valid_action_strategy[val] = 1
                elif new_hand_info['handName']=='flush':
                    valid_action_strategy[1] = 1
                elif new_hand_info['handName']=='straight':
                    valid_action_strategy[1] = 1
                elif self.hand_info['handName']=='four of a kind':
                    valid_action_strategy[1] = 1
                elif self.royal_cat_hop:
                    valid_action_strategy[1] = 1
                elif new_hand_info['handName']=='one pair':
                    if new_hand_info['value'] <= self.hand_info['value']:
                        # fold unimproved pairs
                        val = self.check_if_no_bet_otherwise_fold(history)
                        valid_action_strategy[val] = 1
                    else:
                        pair_value = self.check_duplicate_value(poker_cards,2)
                        # minimum requirement to stay in the game
                        if pair_value in ['8','9','t','j','q','k','a']:
                            valid_action_strategy[1] = 1
                        else:
                            val = self.check_if_no_bet_otherwise_fold(history)
                            valid_action_strategy[val] = 1
                else:
                    val =  self.check_if_no_bet_otherwise_fold(history)
                    valid_action_strategy[val] = 1
            actual_strategy[valid_actions] = valid_action_strategy
            return random.choices(self.bet_actions,weights=actual_strategy,k=1)[0]
