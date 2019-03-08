import reinforcement_learning.agents as agents
import reinforcement_learning.base as base
import reinforcement_learning.simulator as simulator
import reinforcement_learning.view_models as view_models

def test_simulator():

# new environmentGenerator
    class TestScorer(base.Scorer):
        """Mock Scorer for building the mock Environment."""
        def score(self, state: 'base.State'):
            return 1

    class TestEnvironment(base.Environment):
        """Mock Environment."""
        def __init__(self, initial_state):
            super().__init__(
                initial_state=initial_state,
                scorer=TestScorer(),
                updater=base.Updater())

        def _configure_actions(self):
            self.add_action(base.Action('test', 1))

    class TestEnvironmentGenerator(base.EnvironmentGenerator):
        """Spy EnvironmentGenerator."""
        new_environment_counter = 0

        def new_environment(self):
            self.new_environment_counter += 1
            initial_state = base.State(0)
            return TestEnvironment(initial_state=initial_state)

    class TestAgent(agents.Agent):
        """Spy Agent."""
        choose_action_counter = 0
        learn_counter = 0

        def choose_action(self, state: 'base.State'):
            self.choose_action_counter += 1
            return self._action_list[0]

        def learn(self):
            self.learn_counter += 1

    class TestViewModel(view_models.ViewModel):
        """Spy ViewModel."""
        update_counter = 0

        def update(self, scenario_count: int, step_count: int, environment: 'base.Environment', agent: 'agents.Agent'):
            self.update_counter += 1
 
    environment_generator = TestEnvironmentGenerator()
    environment = environment_generator.new_environment()

    agent = TestAgent(environment.action_list)

    view_model = TestViewModel()

    test_simulator = simulator.Simulator(
        environment_generator=environment_generator,
        agent=agent,
        view_model=view_model,
        num_scenarios=10,
        num_rounds=10)

    assert isinstance(test_simulator, simulator.Simulator)

    test_simulator.run()

    assert environment_generator.new_environment_counter == 10
    assert agent.choose_action_counter == 100
    assert agent.learn_counter == 100
    assert view_model.update_counter == 100