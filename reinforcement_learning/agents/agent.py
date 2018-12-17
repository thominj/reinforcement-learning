import random
from typing import List

from reinforcement_learning.states.state import State
from reinforcement_learning.actions.action import Action
from reinforcement_learning.rewards.reward import Reward

class Agent(object):

    def __init__(self, state: State, action_list: List[Action]):
        self._state = state
        self._action_list = action_list

    def choose_action(self, state: State, reward: Reward):
        return random.choice(self._action_list)

    def learn(self, state: State, reward: Reward):
        pass
    