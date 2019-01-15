"""Base classes for reinforcement learning
"""

import abc

class Action():
    """Base class for actions that Agents can choose to take.
    """

    def __init__(self, name: str, move: int):
        self.__name = name
        self.__move = move

    @property
    def name(self):
        """Short description of this action.
        """
        return self.__name

    @property
    def move(self):
        """The actual move this action will perform.
        """
        return self.__move

class Environment(abc.ABC):
    """Base class for a Reinforcement Learning Environment

    Environments are what Agents interact with. The contain a state, a scorer, and an updater.
    Agents observe the state, perform actions which update the state using the updater, and
    receive a reward from the scorer based on the new state.
    """

    def __init__(self, initial_state: 'State', scorer: 'Scorer', updater: 'Updater'):
        self._state = initial_state
        self._scorer = scorer
        self._updater = updater
        self._action_list = []
        self._configure_actions()

    @property
    def state(self):
        """The current state of the environment.
        """
        return self._state

    @state.setter
    def state(self, state: 'State'):
        self._state = state

    def update(self, action: 'Action'):
        """Updates the state using the provided Action

        Args:
            action (Action): Describes a change in state.
        """
        self._updater.update_state(self._state, action)
        return self._state

    @property
    def reward(self):
        """The reward for the current state
        """
        return self._scorer.score(self._state)

    @abc.abstractmethod
    def _configure_actions(self):
        pass

    @property
    def action_list(self):
        """List of actions that agents can take.
        """
        return self._action_list

    def add_action(self, action: 'Action'):
        """Adds an action to the action list.
        """
        self._action_list.append(action)

class Reward():
    """Reward computed by a Scorer for a given State.
    """
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

class Scorer(abc.ABC):
    """Computes a Reward based on the current State.
    """
    @abc.abstractmethod
    def score(self, state: 'State') -> 'Reward':
        """Returns the score for a given State.
        """
        return Reward(0)

class State():
    """All information about an environment needed by the Scorer
    to compute a Reward, and by an Agent to decide on a new Action.
    """
    def __init__(self, initial_value):
        self.value = initial_value

class Updater():
    """Updates the State according to the given Action.
    """
    def update_state(self, state: 'State', action: 'Action'):
        """Applies the action to the state.
        """
        state.value += action.move

class Output(abc.ABC):

    @abc.abstractmethod
    def display(self, state: 'State', reward: 'Reward', trainer: 'Trainer'):
        """Displays the relevent information about the current state and reward.
        """

class Trainer():

    def __init__(self):
        self._scenario_count = 0
        self._trial = 0

    @property
    def scenario_count(self):
        return self._scenario_count

    @property
    def trial(self):
        return self._trial

class EnvironmentGenerator(abc.ABC):

    @abc.abstractmethod
    def new_environment(self) -> 'Environment':
        pass
