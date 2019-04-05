"""A collection of Agents that can be used for Reinforcement Learning
"""
import abc
import random
from typing import List

import numpy as np

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

    def __init__(self, action_list: List['Action'], learning_rate=0.1, min_prob=0.0001):
        self.learning_rate = learning_rate
        self.min_prob = min_prob
        self.probability_map = np.array([1.0/float(len(action_list)) for _ in action_list])
        self.probability_mask = np.ones(len(action_list))
        self.prev_reward = base.Reward(0)
        super().__init__(action_list)


    def choose_action(self, state: "base.State"):
        """Randomly chooses an action from the action_list, with choice weighted by the probability table.
        """
        temp_probability_map = self.probability_map * self.probability_mask
        temp_probability_map = temp_probability_map / np.sum(temp_probability_map)
        try:
            action = np.random.choice(self._action_list, p=(temp_probability_map))
        except:
            import pdb; pdb.set_trace()
        self.prev_action_index = self._action_list.index(action)
        return action

    def learn(self, state: "base.State", reward: "base.Reward"):
        """Updates weights based on reward.
        """
        adjustment_amount = (reward.value - self.prev_reward.value)*self.learning_rate
        self.probability_map[self.prev_action_index] += adjustment_amount

        for index, prob in enumerate(self.probability_map):
            # Add probability for this reward and adjust others to maintain normalization             
            if prob < self.min_prob:
                self.probability_map[index] = self.min_prob
        
        self.probability_map = self.probability_map / np.sum(self.probability_map)
