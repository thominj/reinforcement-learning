"""A collection of Agents that can be used for Reinforcement Learning
"""
import abc
import random
from typing import List

import reinforcement_learning.base as base

class Agent(abc.ABC):
    """Abstract Base Class for agents.
    """
    def __init__(self, action_list: List['Action']):
        self._action_list = action_list

    @abc.abstractmethod
    def choose_action(self, state: "base.State"):
        """Chooses an action from the action list based on some policy.
        """

    @abc.abstractmethod
    def learn(self, state: "base.State", reward: "base.Reward"):
        """Learns from the previous action and reward.
        """

class RandomAgent(Agent):
    """Randomly selects an action without considering reward or state.
    """

    def choose_action(self, state: "base.State"):
        """Randomly chooses an action from the action_list.
        """
        return random.choice(self._action_list)

    def learn(self, state: "base.State", reward: "base.Reward"):
        """RandomAgents can't learn, so this does nothing.
        """

class LearningRandomAgent(Agent):
    """Randomly selects actions, but modifies probability of selecting an action in the future based on reward.
    """

    def init(self, action_list: List['Action'], learning_rate=0.1, min_prob=0.01):
        self.learning_rate = learning_rate
        self.min_prob = min_prob
        super().__init__(self, action_list)

    def choose_action(self, state: "base.State"):
        """Randomly chooses an action from the action_list, with choice weighted by the probability table.
        """
        return random.choice(self._action_list)

    def learn(self, state: "base.State", reward: "base.Reward"):
        """Updates weights based on reward.
        """
