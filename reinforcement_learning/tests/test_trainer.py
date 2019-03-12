import reinforcement_learning.agents as agents
import reinforcement_learning.base as base
import reinforcement_learning.demos.cipher_puzzle as cipher_puzzle
import reinforcement_learning.trainers as trainers

def test_monte_carlo_trainer():

    class TestScorer(base.Scorer):
        def score(self, state: 'State'):
            return base.Reward(state.value - 0)

    class TestEnvironment(base.Environment):

        def __init__(self, initial_state):
            super().__init__(
                initial_state=initial_state,
                scorer=TestScorer(),
                updater=base.Updater())

        def _configure_actions(self):
            self.add_action(base.Action('up', 1))
            self.add_action(base.Action('down', -1))

    class TestEnvironmentGenerator(base.EnvironmentGenerator):
        
        def new_environment(self):
            initial_state = base.State(0)
            return TestEnvironment(initial_state=initial_state)

    class TestOutput(base.Output):

        def __init__(self):
            self.states = []
            self.rewards = []
            self.trainers = []

        def display(self, state: 'State', reward: 'Reward', trainer: 'Trainer'):
            self.states.append(state)
            self.rewards.append(reward)
            self.trainers.append(trainer)

    environment_generator = TestEnvironmentGenerator()
    environment = environment_generator.new_environment()
    agent = agents.RandomAgent(action_list=environment.action_list)
    trainer = trainers.MonteCarloTrainer(
        environment_generator=environment_generator, 
        agent=agent, 
        output=TestOutput(),
        num_scenarios=1,
        max_trials=10,
        keep_prob=0.5,
    )