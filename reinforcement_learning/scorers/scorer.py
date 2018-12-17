from reinforcement_learning.states.state import State
from reinforcement_learning.rewards.reward import Reward

class Scorer(object):
    
    def get_reward(self, state: State):
        return Reward(state.value - 0)