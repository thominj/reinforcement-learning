"""Classes needed to build a Ciper Puzzle environment."""
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

class CipherMutateAction(base.Action):
    """An action that mutates a cipher

        Args:
            name (string): A name or short description of the action
            move (dict): A dict describing the mutation, e.g. {'a':'j'}
                        would modify the cipher so that 'a' is converted
                        to 'j' and whatever used to be converted to 'j' is
                        now converted to whatever 'a' used to be converted to.
    """
    def __init__(self, name: str, move: Dict):
        super().__init__(name, move)

class CipherPuzzleEnvironment(base.Environment):
    """Environment for training agents to solve a Cipher Puzzle.
    """
    def __init__(self, initial_state: 'State', words: List[str], phrases: List[str]):
        scorer = RecognizedWordAndPhraseScorer(words, phrases)
        updater = CipherPuzzleUpdater()
        super().__init__(initial_state, scorer, updater)

    def _configure_actions(self):
        chars = 'abcdefghijklmnopqrstuvwxyz'
        for i in chars:
            for j in chars:
                if i != j:
                    self.add_action(CipherMutateAction("{} to {}".format(i, j), {i:j}))

class CipherPuzzleEnvironmentGenerator(base.EnvironmentGenerator):
    """Generates Cipher Puzzle environments."""

    def __init__(self, words_list: List, phrases_list: List):
        self.words_list = words_list
        self.phrases_list = phrases_list

    def new_environment(self):
        random_phrase = random.choice(self.phrases_list)
        cipher = Cipher()
        puzzle = cipher.encipher(random_phrase.lower())
        initial_state = CipherPuzzleState(puzzle=puzzle)
        return CipherPuzzleEnvironment(
            initial_state=initial_state,
            words=self.words_list,
            phrases=self.phrases_list)

class RecognizedWordAndPhraseScorer(base.Scorer):
    """Calculates reward based on recognized words and phrases.

    Args:
        words: (list) A list of words as strings
        phrases: (list) A list of phrases
    """
    def __init__(self, words: List, phrases: List):
        # lowercase and remove duplicates and trailing whitespace
        self._words = {word.lower().rstrip() for word in words}
        self._phrases = {phrase.lower().rstrip() for phrase in phrases}

    @property
    def words(self):
        """List of words scorer can recognize
        """
        return self._words

    @property
    def phrases(self):
        """List of phrases scorer can recognize
        """
        return self._phrases

    def score(self, state: 'CipherPuzzleState'):
        """Computes the reward for the current state.
        """
        current_output = state.current_output

        # count number of recognized words
        recognized_word_count = 0
        potential_words = current_output.split()
        for word in potential_words:
            word = re.sub(r'[^a-zA-Z]', '', word)
            if word in self.words:
                recognized_word_count += 1

        # check to see if phrase is recognized
        if current_output in self.phrases:
            score = 0.5
        else:
            score = 0.0

        # return weighted score
        score += (recognized_word_count / len(potential_words))*0.5
        return base.Reward(value=score)

class CipherPuzzleState():
    """Represents the state of a cipher puzzle.

    Args:
        puzzle: A string of enciphered text
        cipher: (Cipher) A cipher object
    """

    def __init__(self, puzzle: str, cipher: 'Cipher' = None):
        self._puzzle = puzzle
        if cipher:
            self._cipher = cipher
        else:
            self._cipher = Cipher()

    @property
    def puzzle(self):
        """The puzzle we are trying to solve.
        """
        return self._puzzle

    @property
    def cipher(self):
        """The key used to decipher the puzzle.
        """
        return self._cipher

    @property
    def current_output(self):
        """Plaintext output using the current cipher.
        """
        return self._get_current_output()

    def _get_current_output(self):
        # Apply the cipher to the puzzle and return the output
        return self.cipher.decipher(self.puzzle)

class CipherPuzzleUpdater(base.Updater):
    """Applies a CipherMutateAction to a CipherPuzzleState

    Args:
        state (CipherPuzzleState): The state that will be mutated.
        action (CipherMutateAction): The mutation that will be applied.

    """

    def update_state(self, state: 'CipherPuzzleState', action: 'CipherMutateAction'):
        state.cipher.mutate(action)

class Cipher():
    """A class for enciphering and deciphering text by using a key.

    Args:
        initial_map: (dict) A 1:1 mapping of characters to characters.

    Raises:
        CipherException when initial_map is not 1:1.
    """
    chars = list('abcdefghijklmnopqrstuvwxyz')

    def __init__(self, initial_map=None):
        if initial_map:
            keys = set(initial_map.keys())
            values = set(initial_map.values())
            if len(keys) != len(values):
                raise CipherException('Cipher map is not one-to-one.')
            self.map = initial_map
        else:
            self.map = self._generate_random_map()
        self.reverse_map = {v: k for k, v in self.map.items()}

    def _generate_random_map(self):
        random_chars = random.sample(self.chars, len(self.chars))
        return {k: v for k, v in zip(self.chars, random_chars)}

    def encipher(self, text: str):
        """Uses cipher to convert plain text string to ciphertext.
        """
        cipher_text = ''
        for char in text:
            if self.map.get(char):
                cipher_text += self.map.get(char)
            else:
                cipher_text += char
        return cipher_text

    def decipher(self, cipher_text: str):
        """Uses cipher to convert ciphertext to plain text
        """
        plain_text = ''
        for char in cipher_text:
            if self.reverse_map.get(char):
                plain_text += self.reverse_map.get(char)
            else:
                plain_text += char
        return plain_text

    def mutate(self, action: 'CipherMutateAction'):
        """Mutates the cipher.

        The move consists of a character key and value. The mutation will swap the
        characters in the map. For example, given a map:

        {
            'a': 'z',
            'b': 'x',
            'c': 'y',
        }

        A CipherMutateAction containing the move {'a': 'x'} would result in 'a' pointing
        to 'x' and 'b' pointing to 'z', since 'b' was formerly pointing to 'x' and 'a' was
        formerly pointing to 'z'. The final map would look like:

        {
            'a': 'x',
            'b': 'z',
            'c': 'y',
        }

        Args:
            action (CipherMutateAction): Contains a move indicating which character
            should change, and what it should change to.
        """
        # get key and value (dict has one item)
        [(key, value)] = action.move.items()

        # get current value for this key
        current_value = self.map.get(key)

        # get current key for this value
        current_key = self.reverse_map.get(value)

        # change map value at action key to action value
        self.map[key] = value
        self.reverse_map[value] = key

        # change map value at current key to current value
        self.map[current_key] = current_value
        self.reverse_map[current_value] = current_key

# @todo(jdt) I don't think we need output classes anymore if we are using ViewModels
class CipherPuzzleOutput(base.Output):

    def display(self, state: 'CipherPuzzleState', reward: 'Reward', trainer: 'Trainer'):
        print(f'Current output: {state.current_output}')
        print(f'Puzzle: {state.puzzle}')
        print(f'Scenario: {trainer.scenario_count}, Trial: {trainer.trial}')        

class CipherPuzzlePrintViewModel(view_models.ViewModel):

    best_solution = ''
    best_score = 0

    def __init__(self):
        print("Timestamp, Scenario, Step, Puzzle, Solution, Score, Best Solution, Best Score")

    def update(self, scenario_count: int, step_count: int, environment: 'base.Environment', agent: 'agents.Agent'):
        
        if environment.reward.value > self.best_score:
            self.best_solution = environment.state.current_output
            self.best_score = environment.reward.value

        if step_count % 1 == 0:
            print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                datetime.datetime.now(),
                scenario_count,
                step_count,
                environment.state.puzzle.rstrip(),
                environment.state.current_output,
                environment.reward.value,
                self.best_solution,
                self.best_score,
                min(agent.probability_map),
                max(agent.probability_map)))

class CipherPuzzleAgent(agents.Agent):

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

    def __init__(self, action_list: List['Action'], learning_rate=0.1, min_prob=0.01):
        self.action_map = {}
        for index, action in enumerate(action_list):
            from_to_pair = [(k,v) for k,v in action.move.items()][0]
            self.action_map[from_to_pair] = index
        super().__init__(action_list, learning_rate, min_prob)

    def choose_action(self, state: "base.State"):
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

class CipherException(Exception):
    """Raised when an exception occurs in a Cipher object.
    """
