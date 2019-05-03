"""Agents for solving Cipher Puzzles."""
import datetime
import os
import random
import re
from typing import List, Dict

import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Embedding, Input, Concatenate, Dense, Dropout, Flatten, GRU
from keras.optimizers import SGD
from keras.preprocessing import sequence, text
from keras.utils import to_categorical

import reinforcement_learning.base as base
import reinforcement_learning.agents as agents
import reinforcement_learning.view_models as view_models
import reinforcement_learning.demos.cipher_puzzle.base as cp_base

class CipherPuzzleNeuralAgent(agents.Agent):

    max_puzzle_length = 250
    random_prob = 1.0
    decay_param = 0.99
    checkpoints_dir = 'reinforcement_learning/demos/cipher_puzzle/checkpoints/'
    save_freq = 100
    step_counter = 0
    use_existing_checkpoint = True
    model = None

    def __init__(self, action_list: List['base.Action']):        
        self._action_list = action_list
        self._vocabulary = self.Vocabulary()

        if (self.use_existing_checkpoint):
            checkpoint_file = self.find_last_checkpoint(self.checkpoints_dir)
            if (checkpoint_file):
                self.model = load_model(os.path.join(self.checkpoints_dir, checkpoint_file))
        if self.model is None:
            self.init_model()

    def find_last_checkpoint(self, directory):
        files = os.listdir(directory)
        max_index = 0
        last_checkpoint = None
        for filepath in files:
            if filepath[-3:] == '.h5':
                index = int(filepath[:-3].split('-')[-1])
                if index > max_index:
                    max_index = index
                    last_checkpoint = filepath
        return last_checkpoint                

    def init_model(self):
        # input layers for puzzle and current state
        puzzle_input = Input(shape=(self.max_puzzle_length,))
        current_state_input = Input(shape=(self.max_puzzle_length,))

        # pass them both through embedding layers
        embedded_puzzle = Embedding(29, 1024)(puzzle_input)
        embedded_current_state = Embedding(29, 1024)(current_state_input)

        # GRU layers
        input_gru = GRU(1024, activation='relu')(embedded_puzzle)
        current_state_gru = GRU(1024, activation='relu')(embedded_current_state)

        # concatenate the results
        concat = Concatenate()([input_gru, current_state_gru])

        # Dropout
        dropout1 = Dropout(0.1)(concat)

        # Dense layers
        dense1 = Dense(512, activation='relu')(concat)
        dropout2 = Dropout(0.1)(dense1)
#        dense2 = Dense(1024, activation='relu')(dense1)
#        dense3 = Dense(512, activation='relu')(dense2)

        # Output layer
        output = Dense(len(self._action_list), activation='softmax')(dropout2)
        
        self.model = Model(inputs=[puzzle_input, current_state_input], outputs=output)
        optimizer = SGD(lr=1e-03, momentum=0.1, decay=1e-04, nesterov=False)
        self.model.compile(optimizer, 'categorical_crossentropy')

    def choose_action(self, state: "base.State"):
        # encode the puzzle and current output
        puzzle = self._vocabulary.encode(state.puzzle.rstrip(), self.max_puzzle_length)
        current_output = self._vocabulary.encode(state.current_output.rstrip(), self.max_puzzle_length)

        # pass them as model inputs to get prediction
        self.last_predictions = self.model.predict([puzzle.reshape(1,-1), current_output.reshape(1,-1)], batch_size=1)

        # get max value index -or- random selection
        if random.uniform(0,1) < self.random_prob:
            self.last_action_index = random.choice(range(len(self._action_list)))
            self.random_prob_decay()
        else:
            self.last_action_index = np.argmax(self.last_predictions)

        # return the action from the list
        return self._action_list[self.last_action_index]

    def learn(self, state: "base.State", reward: "base.Reward"):
        # encode the puzzle and the current state
        puzzle = self._vocabulary.encode(state.puzzle.rstrip(), self.max_puzzle_length)
        current_output = self._vocabulary.encode(state.current_output.rstrip(), self.max_puzzle_length)

        # pass them as model inputs to get predictions
        predictions = self.model.predict([puzzle.reshape(1,-1), current_output.reshape(1,-1)], batch_size=1)

        # get max value and index
        max_value = np.max(predictions)

        # multiply by decay param and add reward value to get target value for the last action
        self.last_predictions[0, self.last_action_index] = reward.value + self.decay_param*max_value

        # train the model with modified predictions as "target Q"
        loss = self.model.train_on_batch([puzzle.reshape(1,-1), current_output.reshape(1,-1)], self.last_predictions)

        # save checkpoints maybe
        self.step_counter += 1
        if self.step_counter % self.save_freq == 0:
            self.model.save(os.path.join(self.checkpoints_dir, 'cipher_puzzle_agent-{}.h5'.format(self.step_counter)))

        print(f"loss: {loss}")

    def random_prob_decay(self, decay_rate = 0.0001):
        if self.random_prob >= 0:
            self.random_prob = self.random_prob - decay_rate

    class Vocabulary():

        alpha = 'abcdefghijklmnopqrstuvwxyz '
        map = {
                '_PAD': 0,
                '_UNK': 1,
            }

        def __init__(self):
            for index, char in enumerate(self.alpha):
                self.map[char] = index + 2
            
            self.reverse_map = {v: k for k, v in self.map.items()}

        def encode(self, string: str, max_length=None):
            if not max_length:
                max_length = len(string)
            encoded = np.zeros(max_length)
            for index, char in enumerate(string[:max_length]):
                value = self.map.get(char)
                if value:
                    encoded[index] = value
                else:
                    encoded[index] = self.map.get('_UNK')

            return encoded[:max_length]

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

    def __init__(self, action_list: List["cp_base.CipherMutateAction"], learning_rate=0.1, min_prob=1):
        self.action_map = {}
        for index, action in enumerate(action_list):
            [(origin, target)] = action.move.items()
            self.action_map[(origin, target)] = index
        super().__init__(action_list, learning_rate, min_prob)

    def choose_action(self, state: "cp_base.CipherPuzzleState"):
        """Tries to intelligently modify probabilities, then randomly chooses an action from the action_list,
        with choice weighted by the probability table.
        """
        if self._is_new_puzzle(state):
            self._initialize_probability_map(state)
            self.last_puzzle = state.puzzle
        self._update_probability_mask(state)
        action = super().choose_action(state)
        return action

    def _initialize_probability_map(self, state: "cp_base.CipherPuzzleState"):
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
        one_letter_words = []
        two_letter_words = []
        for word in words:
            word = ''.join(self.alpha.intersection(set(word)))
            if len(word) == 1:
                one_letter_words.append(word)
            if len(word) == 2:
                two_letter_words.append(word)

        for char in one_letter_words:
            for target in self.one_letter_word_map[0]:
                if char != target:
                    self.probability_map[self.action_map[(target, char)]] = 10.0 * self.probability_map[self.action_map[(target, char)]]

        for word in two_letter_words:
            for index, char in enumerate(word):
                for target in self.two_letter_word_map[index]:
                    if char != target:
                        self.probability_map[self.action_map[(target, char)]] = 2.0 * self.probability_map[self.action_map[(target, char)]]

 #       self.probability_map = np.array([p/sum(self.probability_map) for p in self.probability_map])

    def _is_new_puzzle(self, state: "cp_base.CipherPuzzleState"):
        return self.last_puzzle != state.puzzle

    def _update_probability_mask(self, state: "cp_base.CipherPuzzleState"):
        self.probability_mask = np.ones(len(self._action_list))
        for origin, target in state.cipher.reverse_map.items():
            if origin != target:
                self.probability_mask[self.action_map[(target, origin)]] = 0

