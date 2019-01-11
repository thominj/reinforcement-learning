import random
import re
from typing import List, Dict

from reinforcement_learning.base import Action, Agent, Environment, Reward, Scorer, State, Updater

class CipherMutateAction(Action):

    def __init__(self, name: str, move: Dict):
        self.name = name
        self.move = move

class RandomAgent(Agent):
    '''Randomly selects an action without considering reward or state.
    '''
    def __init__(self, state: 'State', action_list: List['Action']):
        self._state = state
        self._action_list = action_list

    def choose_action(self, state: 'State', reward: 'Reward'):
        return random.choice(self._action_list)

    def learn(self, state: 'State', reward: 'Reward'):
        pass
    
class CipherPuzzleEnvironment(Environment):

    # @todo(jdt) Why can't I get inheritance to work correctly for this class?
    def __init__(self, initial_state: 'State', words: List[str], phrases: List[str]):
        self.__state = initial_state
        self.__scorer = RecognizedWordAndPhraseScorer(words, phrases)
        self.__updater = CipherPuzzleUpdater()
        self.__configure_actions()

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state: 'State'):
        self.__state = state

    def update(self, action: 'Action'):
        self.__updater.update_state(self.__state, action)
        return self.__state

    @property
    def reward(self):
        return self.__scorer.get_reward(self.__state)

    @property
    def action_list(self):
        return self.__action_list
    
    def __configure_actions(self):
        chars = 'abcdefghijklmnopqrstuvwxyz'
        self.__action_list = []
        for i in chars:
            for j in chars:
                self.__action_list.append(CipherMutateAction("{} to {}".format(i,j), {i:j}))

class RecognizedWordAndPhraseScorer(Scorer):

    def __init__(self, words: List, phrases: List):
        # TODO: lowercase and remove duplicates
        self.words = words
        self.phrases = phrases

    def get_reward(self, state: 'CipherPuzzleState'):
        current_output = state.current_output

        # count number of recognized words
        recognized_word_count = 0
        # TODO: can we also ignore all characters not in cipher list, so that punctuation
        # does not mess up score?
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
        return Reward(value=score) 

class CipherPuzzleState(State):

    def __init__(self, puzzle: str, cipher: 'Cipher' = None):
        self.puzzle = puzzle
        if cipher:
            self.cipher = cipher
        else:
            self.cipher = Cipher()

    @property
    def current_output(self):
        return self._get_current_output()

    def _get_current_output(self):
        # Apply the cipher to the puzzle and return the output
        return self.cipher.decipher(self.puzzle)

class CipherPuzzleUpdater(Updater):

    def update_state(self, state: 'State', action: 'Action'):
        state.cipher.mutate(action)

class Cipher(object):

    chars = list('abcdefghijklmnopqrstuvwxyz')

    def __init__(self, initial_map = None):
        if initial_map:
            self.map = initial_map
            # TODO: need to check that map is 1:1
        else:
            self.map = self._generate_random_map()
        self.reverse_map = {v: k for k, v in self.map.items()}

    def _generate_random_map(self):
        random_chars =  random.sample(self.chars, len(self.chars))  
        return {k: v for k, v in zip(self.chars, random_chars)}
    
    def encipher(self, text: str):
        cipher_text = ''
        for char in text:
            if self.map.get(char):
                cipher_text += self.map.get(char)
            else:
                cipher_text += char
        return cipher_text
    
    def decipher(self, cipher_text: str):
        plain_text = ''
        for char in cipher_text:
            if self.reverse_map.get(char):
                plain_text += self.reverse_map.get(char)
            else:
                plain_text += char
        return plain_text

    def mutate(self, action: 'CipherMutateAction'):
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