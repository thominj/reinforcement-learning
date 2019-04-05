"""Classes needed to build a Ciper Puzzle environment."""
import datetime

import reinforcement_learning.demos.cipher_puzzle as cipher_puzzle
import reinforcement_learning.base as base
import reinforcement_learning.view_models as view_models

class CipherPuzzlePrintViewModel(view_models.ViewModel):

    best_solution = ''
    best_score = 0

    def __init__(self):
        print("Timestamp, Scenario, Step, Puzzle, Solution, Score, Best Solution, Best Score")

    def update(
        self, 
        scenario_count: int,
        step_count: int,
        environment: 'cipher_puzzle.base.CipherPuzzleEnvironment',
        agent: 'cipher_puzzle.agents.CipherPuzzleLearngingRandomAgent'):

        if environment.reward.value > self.best_score:
            self.best_solution = environment.state.current_output
            self.best_score = environment.reward.value

        if step_count % 1 == 0:
            print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                datetime.datetime.now(),
                scenario_count,
                step_count,
                environment.state.puzzle.rstrip(),
                environment.state.current_output,
                environment.reward.value,
                self.best_solution,
                self.best_score,
                min(agent.probability_map),
                max(agent.probability_map)))