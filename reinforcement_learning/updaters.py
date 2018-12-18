from reinforcement_learning.actions import Action
from reinforcement_learning.states import State

class Updater(object):

    def update_state(self, state: State, action: Action):
        state.value += action.move