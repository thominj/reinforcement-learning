from reinforcement_learning.states import State
from reinforcement_learning.rewards import Reward

class Scorer(object):
    
    def get_reward(self, state: State):
        return Reward(state.value - 0)