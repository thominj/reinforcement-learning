"""Tests agents
"""
import reinforcement_learning.agents as agents
import reinforcement_learning.base as base

def testLearningRandomAgentLearns():

    action_list = []
    action_list.append(base.Action('1', 1))
    action_list.append(base.Action('2', 2))
    action_list.append(base.Action('3', 3))

    state = base.State(0)

    agent = agents.LearningRandomAgent(action_list)

    rewarded_action_count = 0
    unrewarded_action_count = 0
    for _ in range(100):
        action = agent.choose_action(state)
        if action.name == '1':
            reward = base.Reward(10)
            rewarded_action_count += 1
        else:
            reward = base.Reward(0)
            unrewarded_action_count += 1

        agent.learn(state, reward)

    assert rewarded_action_count - unrewarded_action_count > 0
