import click
import typing

import reinforcement_learning.demos.cipher_puzzle.agents as cp_agents
import reinforcement_learning.demos.cipher_puzzle.base as cp_base
import reinforcement_learning.demos.cipher_puzzle.view_models as cp_view_models
import reinforcement_learning.agents as agents
import reinforcement_learning.simulator as simulator

@click.command()
@click.option('-a', '--agent', help="The agent you want to use: [random (default), learningRandom, cipherPuzzleLearningRandom]", default='random', type=str, )
@click.option('-n', '--num_steps', help="The number of steps to run for each scenario.", default=100, type=int)
@click.option('-s', '--num_scenarios', help="The number of scenarios to run.", default = 10, type=int)
def demo(agent, num_steps, num_scenarios):
    """Demonstrates a scenario using a specified Agent and a Cipher Puzzle"""
    agent_options = {
        'random': agents.RandomAgent,
        'learningRandom': agents.LearningRandomAgent,
        'cipherPuzzleLearningRandom': cp_agents.CipherPuzzleLearningRandomAgent,
    }
    
    words_list = []
    phrases_list = []

    with open('reinforcement_learning/demos/cipher_puzzle/data/word_list.txt', 'r') as words_file:
        for word in words_file:
            words_list.append(word)

    with open('reinforcement_learning/demos/cipher_puzzle/data/famous_quotes.txt', 'r') as quote_file:
        for quote in quote_file:
            phrases_list.append(quote)

    environment_generator = cp_base.CipherPuzzleEnvironmentGenerator(
        words_list=words_list,
        phrases_list=phrases_list)
    environment = environment_generator.new_environment()

    agent = agent_options[agent](environment.action_list)

    view_model = cp_view_models.CipherPuzzlePrintViewModel()

    test_simulator = simulator.Simulator(
        environment_generator=environment_generator,
        agent=agent,
        view_model=view_model,
        num_scenarios=num_scenarios,
        num_steps=num_steps)

    test_simulator.run()