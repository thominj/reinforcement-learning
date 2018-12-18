import reinforcement_learning.base as base

def test_integration():
    initial_state = base.State(0)
    env = base.Environment(initial_state=initial_state)
    assert env.state == initial_state

    agent = base.Agent(state=initial_state, action_list=env.action_list)
    rewards = []

    for _ in range(10):
        action = agent.choose_action(env.state, env.reward)
        env.update(action)
        rewards.append(env.reward)
        agent.learn(env.state, env.reward)        

    # At some point the reward should have changed
    assert len(set(rewards)) > 1