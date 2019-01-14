"""A collection of Agents that can be used for Reinforcement Learning
"""
import abc
import random
from typing import List

class Agent(abc.ABC):
    """Abstract Base Class for agents.
    """
    def __init__(self, action_list: List['Action']):
        self._action_list = action_list

    @abc.abstractmethod
    def choose_action(self):
        """Chooses an action from the action list based on some policy.
        """

    @abc.abstractmethod
    def learn(self):
        """Learns from the previous action and reward.
        """

class RandomAgent(Agent):
    '''Randomly selects an action without considering reward or state.
    '''

    def choose_action(self):
        """Randomly chooses an action from the action_list.
        """
        return random.choice(self._action_list)

    def learn(self):
        """RandomAgents can't learn, so this does nothing.
        """
