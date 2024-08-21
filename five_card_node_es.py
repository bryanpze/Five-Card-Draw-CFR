import numpy as np
class FiveCardNodeES:
    # Takes a list of valid actions
    # __slots__ = ['info_set','strategy_sum','regret_sum','num_valid_actions']
    def __init__(self,info_set,num_valid_actions):
        self.info_set = info_set
        # self.actual_strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_valid_actions)
        self.regret_sum = np.zeros(num_valid_actions)
        self.num_valid_actions = num_valid_actions
        # self.strategy = np.zeros(len(valid_actions))
        
    def get_strategy(self):
        regret_sum = self.regret_sum
        regret_sum[regret_sum<0] = 0
        normalizing_sum = sum(self.regret_sum)
        if normalizing_sum>0:
            return  self.regret_sum/normalizing_sum
        else:
            return np.repeat(1/self.num_valid_actions, self.num_valid_actions)
            
    def update_strategy(self,action_probs):
        self.strategy_sum+=action_probs
    
    def get_average_strategy(self):
        strategy_sum = self.strategy_sum
        normalizing_sum = sum(strategy_sum)
        if normalizing_sum>0:
            return strategy_sum/normalizing_sum
        else:
            return np.repeat(1/self.num_valid_actions,self.num_valid_actions)
        