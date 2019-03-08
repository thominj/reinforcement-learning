"""Classes needed to build a Ciper Puzzle environment."""
import random
import re
from typing import List, Dict

import reinforcement_learning.base as base

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
                self.add_action(CipherMutateAction("{} to {}".format(i, j), {i:j}))

class RecognizedWordAndPhraseScorer(base.Scorer):
    """Calculates reward based on recognized words and phrases.

    Args:
        words: (list) A list of words as strings
        phrases: (list) A list of phrases
    """
    def __init__(self, words: List, phrases: List):
        # lowercase and remove duplicates
        self._words = {word.lower() for word in words}
        self._phrases = {phrase.lower() for phrase in phrases}

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

class CipherPuzzleOutput(base.Output):

    def display(self, state: 'CipherPuzzleState', reward: 'Reward', trainer: 'Trainer'):
        print(f'Current output: {state.current_output}')
        print(f'Puzzle: {state.puzzle}')
        print(f'Scenario: {trainer.scenario_count}, Trial: {trainer.trial}')        

class CipherException(Exception):
    """Raised when an exception occurs in a Cipher object.
    """
