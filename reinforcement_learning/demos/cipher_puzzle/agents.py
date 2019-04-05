"""Classes needed to build a Cipher Puzzle environment."""
import datetime
import random
import re
from typing import List, Dict

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Input, Concatenate, Dense, Activation
from keras.preprocessing import sequence, text
from keras.utils import to_categorical

import reinforcement_learning.base as base
import reinforcement_learning.agents as agents
import reinforcement_learning.view_models as view_models

class CipherPuzzleNeuralAgent(agents.Agent):

    def __init__(self, action_list: List['Action']):        
        self._action_list = action_list

        # input layers for puzzle and current state
        # pass them both through embedding layers
        # concatenate the results
        # Dense layers
        # Output softmax of size len(action_list)

    def choose_action(self, state: "base.State"):
        # encode the puzzle and current state
        # pad them
        # pass them as model inputs to get prediction
        # save prediction vector (last Q)
        # get max value index -or- random selection
        # save the one we picked
        # return the action from the list
        pass

    def learn(self, state: "base.State", reward: "base.Reward"):
        # encode the puzzle and the current state
        # pad them
        # pass them as model inputs to get predictions
        # get max value
        # multiply by decay param and add reward
        # this goes into the predictions array at the index of the action we took last time
        # train the model with modified predictions as "target Q"
        # need to save checkpoints on some frequency
        pass

class CipherPuzzleLearningRandomAgent(agents.LearningRandomAgent):

    alpha = set('abcdefghijklmnopqrstuvwxyz')
    last_puzzle = None

    one_letter_word_map = {
        0: 'ia'
    }

    two_letter_word_map = {
        0: 'otibaswhdmugn',
        1: 'fontseyrpm',
    }

    def __init__(self, action_list: List["CipherPuzzle.CipherMutateAction"], learning_rate=0.1, min_prob=0.01):
        self.action_map = {}
        for index, action in enumerate(action_list):
            from_to_pair = [(k,v) for k,v in action.move.items()][0]
            self.action_map[from_to_pair] = index
        super().__init__(action_list, learning_rate, min_prob)

    def choose_action(self, state: "CipherPuzzle.CipherPuzzleState"):
        """Tries to intelligently modify probabilities, then randomly chooses an action from the action_list,
        with choice weighted by the probability table.
        """
        if self._is_new_puzzle(state):
            self._initialize_probability_map(state)
        return super().choose_action(state)

    def _initialize_probability_map(self, state: "CipherPuzzle.CipherPuzzleState"):
        # reset the map
        self.probability_map = np.ones(len(self._action_list))

        # find all letters that are missing in the initial puzzle
        missing_letters = self.alpha.difference(self.alpha.intersection(set(state.puzzle)))

        # set the corresponding probabilities to zero
        for index, action in enumerate(self._action_list):
            if action.name[0] in missing_letters:
                self.probability_map[index] = 0

        # find one-letter words, set probabilities for the matching letters and i, a to 10 times the base prob
        words = state.puzzle.split()
        word_lengths = {}
        for word in words:
            word = str(self.alpha.intersection(set(word)))
            word_lengths[word] = len(word)

        one_letter_words = []
        two_letter_words = []
        for word, length in word_lengths.items():
            if length == 1:
                one_letter_words.append(word)
            if length == 2:
                two_letter_words.append(word)

        for word in one_letter_words:
            for target in self.one_letter_word_map[0]:
                if word != target:
                    self.probability_map[self.action_map[(word, target)]] = 2.0 * self.probability_map[self.action_map[(word, target)]]

        for word in two_letter_words:
            for index, char in enumerate(word):
                for target in self.two_letter_word_map[index]:
                    if char != target:
                        self.probability_map[self.action_map[(char, target)]] = 2.0 * self.probability_map[self.action_map[(char, target)]]

        self.probability_map = [p/sum(self.probability_map) for p in self.probability_map]

    def _is_new_puzzle(self, state: "CipherPuzzle.CipherPuzzleState"):
        return self.last_puzzle != state.puzzle

    def _update_mask(self, state: "CipherPuzzle.CipherPuzzleState", action: ""):
        pass