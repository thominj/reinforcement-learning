"""Tests for the Cipher Puzzle classes.
"""

import pytest
import reinforcement_learning.agents as agents
import reinforcement_learning.cipher_puzzle as cipher_puzzle

def test_create_cipher():
    cipher = cipher_puzzle.Cipher()
    assert cipher.map is not None
    assert cipher.reverse_map is not None

    good_initial_map = {'a': 'b'}
    cipher = cipher_puzzle.Cipher(initial_map=good_initial_map)
    assert cipher.map == good_initial_map

    bad_initial_map = {'a': 'c', 'b': 'c'}
    with pytest.raises(cipher_puzzle.CipherException):
        cipher = cipher_puzzle.Cipher(bad_initial_map)

def test_encipher_and_decipher():
    cipher = cipher_puzzle.Cipher()
    plain_text = 'This is some plain text.'
    cipher_text = cipher.encipher(plain_text)
    assert cipher_text != plain_text
    deciphered_text = cipher.decipher(cipher_text)
    assert deciphered_text == plain_text

def test_mutate_cipher():
    cipher = cipher_puzzle.Cipher({
        'a': 'z',
        'b': 'x',
        'c': 'y',
        'x': 'c',
        'y': 'a',
        'z': 'b'
    })

    plain_text = 'abc'
    assert cipher.encipher(plain_text) == 'zxy'

    action = cipher_puzzle.CipherMutateAction('a to y', {'a': 'y'})
    cipher.mutate(action)

    assert cipher.encipher(plain_text) == 'yxz'

def test_scorer():
    target_text = 'a very fine day for a test!'
    original_cipher = cipher_puzzle.Cipher()
    puzzle = original_cipher.encipher(target_text)

    words = ['a', 'very', 'fine', 'day', 'for', 'test']
    phrases = ['a very fine day for a test!']
    scorer = cipher_puzzle.RecognizedWordAndPhraseScorer(words=words, phrases=phrases)

    # Score should be 1 with the original cipher
    solved_state = cipher_puzzle.CipherPuzzleState(puzzle=puzzle, cipher=original_cipher)
    assert abs(1.0 - scorer.score(state=solved_state).value) <= 0.0001

    # Score should be close to zero with a random cipher
    random_state = cipher_puzzle.CipherPuzzleState(puzzle=puzzle, cipher=cipher_puzzle.Cipher())
    assert scorer.score(state=random_state).value < 0.1

def test_integration():
    """Tests integration of cipher puzzle components.
    """
    plain_text = 'these are words'
    initial_map = {
        't': 'a',
        'h': 'b',
        'e': 'c',
        's': 'd',
        'a': 'e',
        'r': 'f',
        'w': 'g',
        'o': 'h',
        'd': 'i',
    }
    cipher = cipher_puzzle.Cipher(initial_map=initial_map)
    puzzle = cipher.encipher(plain_text)
    initial_state = cipher_puzzle.CipherPuzzleState(puzzle=puzzle, cipher=cipher)

    words = ['these', 'are', 'words']
    phrases = ['these are words']

    env = cipher_puzzle.CipherPuzzleEnvironment(
        initial_state=initial_state,
        words=words,
        phrases=phrases)
    assert env.state == initial_state

    agent = agents.RandomAgent(action_list=env.action_list)
    rewards = [env.reward.value]

    for _ in range(10):
        action = agent.choose_action()
        env.update(action)
        rewards.append(env.reward.value)

    # At some point the reward should have changed
    assert len(set(rewards)) > 1
