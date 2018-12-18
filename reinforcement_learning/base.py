import random
from typing import List

class Action(object):

    def __init__(self, name: str, move: int):
        self.name = name
        self.move = move

class Agent(object):

    def __init__(self, state: 'State', action_list: List['Action']):
        self._state = state
        self._action_list = action_list

    def choose_action(self, state: 'State', reward: 'Reward'):
        return random.choice(self._action_list)

    def learn(self, state: 'State', reward: 'Reward'):
        pass
    
class Environment(object):
    
    def __init__(self, initial_state: 'State'):
        self.__state = initial_state
        self.__scorer = Scorer()
        self.__updater = Updater()
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

    def __configure_actions(self):
        self.__action_list = []
        self.__action_list.append(Action('up', 1))
        self.__action_list.append(Action('down', -1))

    @property
    def action_list(self):
        return self.__action_list

class Reward(object):
    
    def __init__(self, value):
        self.value = value

class Scorer(object):
    
    def get_reward(self, state: 'State'):
        return Reward(state.value - 0) 

class State(object):

    def __init__(self, initial_value):
        self.value = initial_value

class Updater(object):

    def update_state(self, state: 'State', action: 'Action'):
        state.value += action.move
    