from reinforcement_learning.actions.action import Action
from reinforcement_learning.states.state import State

class Updater(object):

    def update_state(self, state: State, action: Action):
        state.value += action.move