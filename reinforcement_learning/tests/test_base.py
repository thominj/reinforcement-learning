"""Tests of the base classes.
"""
import reinforcement_learning.base as base
import reinforcement_learning.agents as agents

def test_integration():
    """Tests that all base components work together.
    """
    initial_state = base.State(0)

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

    env = TestEnvironment(initial_state=initial_state)
    assert env.state == initial_state

    agent = agents.RandomAgent(action_list=env.action_list)
    rewards = []

    for _ in range(100):
        action = agent.choose_action()
        env.update(action)
        rewards.append(env.reward.value)
        agent.learn()

    # At some point the reward should have changed
    assert len(set(rewards)) > 1
