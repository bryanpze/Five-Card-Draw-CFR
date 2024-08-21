import numpy as np
class FiveCardNode:
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
        regret_sum[regret_sum<0.001] = 0
        normalizing_sum = sum(regret_sum)
        if normalizing_sum==0:
            return np.repeat(1/self.num_valid_actions, self.num_valid_actions)
        else:
            return  regret_sum/normalizing_sum
    
    def update_strategy(self,action):
        self.strategy_sum[action]+=1
    
    def get_average_strategy(self):
        strategy_sum = self.strategy_sum
        # strategy_sum[strategy_sum<0.001] = 0
        normalizing_sum = sum(strategy_sum)
        if normalizing_sum==0:
            return np.repeat(1/self.num_valid_actions,self.num_valid_actions)
        else:
            return strategy_sum/normalizing_sum