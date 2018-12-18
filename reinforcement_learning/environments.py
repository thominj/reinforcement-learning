from typing import List

from reinforcement_learning.states import State
from reinforcement_learning.scorers import Scorer
from reinforcement_learning.updaters import Updater
from reinforcement_learning.actions import Action

class Environment(object):
    
    def __init__(self, 
                 initial_state: State):
        self.__state = initial_state
        self.__scorer = Scorer()
        self.__updater = Updater()
        self.__configure_actions()

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state: State):
        self.__state = state

    def update(self, action: Action):
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
    