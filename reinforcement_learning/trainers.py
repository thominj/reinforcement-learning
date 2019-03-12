import random

import reinforcement_learning.agents as agents
import reinforcement_learning.base as base

class MonteCarloTrainer(base.Trainer):
    """Uses a RandomAgent to select Actions. Stores reward and state between
    actions, and reverts to the previous state if the reward for a new action is
    lower than the previous reward, with a small chance for keeping a worse state.
    """

    def __init__(
        self,
        environment_generator: 'base.EnvironmentGenerator',
        agent: 'agents.RandomAgent',
        output: 'base.Output',
        num_scenarios: int = 1,
        max_trials: int = 100,
        keep_prob: float = 0.1,

    ):
        self._environment_generator = environment_generator
        self._num_scenarios = num_scenarios
        self._max_trials = max_trials
        self._keep_prob = keep_prob
        self._agent = agent
        self._output = output
        super().__init__()

    def train(self):
        self._scenario_count = 0
        while self._scenario_count < self._num_scenarios:
            self._scenario_count += 1
            self._environment = self._environment_generator.new_environment()
            self._trial = 0
            while self._trial < self._max_trials:
                self._trial += 1
                state = self._environment.state
                reward = self._environment.reward
                action = self._agent.choose_action()
                self._environment.update(action)
                new_reward = self._environment.reward
                if new_reward.value < reward.value:
                    if random.random() < self._keep_prob:
                        self._environment.state = state
                
                self._output.send(self._environment.state, self._environment.reward, self)



