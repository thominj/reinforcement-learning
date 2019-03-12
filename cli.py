import click
import typing

import reinforcement_learning.demos.cipher_puzzle.cipher_puzzle as cipher_puzzle
import reinforcement_learning.agents as agents
import reinforcement_learning.simulator as simulator

@click.command()
def demo():
    """Demonstrates a scenario using a random Agent and a Cipher Puzzle"""
    words_list = []
    phrases_list = []

    with open('reinforcement_learning/demos/cipher_puzzle/data/word_list.txt', 'r') as words_file:
        for word in words_file:
            words_list.append(word)

    with open('reinforcement_learning/demos/cipher_puzzle/data/famous_quotes.txt', 'r') as quote_file:
        for quote in quote_file:
            phrases_list.append(quote)

    num_scenarios = 1
    num_steps = 10

    environment_generator = cipher_puzzle.CipherPuzzleEnvironmentGenerator(
        words_list=words_list,
        phrases_list=phrases_list)
    environment = environment_generator.new_environment()

    agent = cipher_puzzle.CipherPuzzleAgent(environment.action_list)

    view_model = cipher_puzzle.CipherPuzzlePrintViewModel()

    test_simulator = simulator.Simulator(
        environment_generator=environment_generator,
        agent=agent,
        view_model=view_model,
        num_scenarios=num_scenarios,
        num_steps=num_steps)

    test_simulator.run()