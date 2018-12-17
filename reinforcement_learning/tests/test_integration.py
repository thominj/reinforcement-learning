from reinforcement_learning.environments.environment import Environment
from reinforcement_learning.actions.action import Action
from reinforcement_learning.states.state import State
from reinforcement_learning.updaters.updater import Updater
from reinforcement_learning.scorers.scorer import Scorer
from reinforcement_learning.agents.agent import Agent

def test_integration():
    initial_state = State(0)
    env = Environment(initial_state=initial_state)
    assert env.state == initial_state

    agent = Agent(state=initial_state, action_list=env.action_list)
    rewards = []

    for _ in range(10):
        action = agent.choose_action(env.state, env.reward)
        env.update(action)
        rewards.append(env.reward)
        agent.learn(env.state, env.reward)        

    # At some point the reward should have changed
    assert len(set(rewards)) > 1