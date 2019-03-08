"""A collection of ViewModels that can be used for displaying simulation results.
"""
import abc

import reinforcement_learning.base as base
import reinforcement_learning.agents as agents

class ViewModel(abc.ABC):

    @abc.abstractmethod
    def update(self, scenario_count: int, step_count: int, environment: 'base.Environment', agent: 'agents.Agent'):
        """Updates the view model with the state of the simulation.
        """
