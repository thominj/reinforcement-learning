from reinforcement_learning.cipher_puzzle import Cipher, CipherMutateAction, CipherPuzzleEnvironment, CipherPuzzleState, RecognizedWordAndPhraseScorer, RandomAgent

def test_cipher():
    cipher = Cipher()
    plain_text = 'This is some plain text.'
    cipher_text = cipher.encipher(plain_text)
    assert cipher_text != plain_text
    deciphered_text = cipher.decipher(cipher_text)
    assert deciphered_text == plain_text

def test_mutate_cipher():
    cipher = Cipher({
        'a': 'z',
        'b': 'x',
        'c': 'y',
        'x': 'c',
        'y': 'a',
        'z': 'b'
    })

    plain_text = 'abc'
    assert 'zxy' == cipher.encipher(plain_text)

    action = CipherMutateAction('a to y', {'a': 'y'})
    cipher.mutate(action)

    assert 'yxz' == cipher.encipher(plain_text)

def test_scorer():
    target_text = 'a very fine day for a test!' 
    original_cipher = Cipher()
    puzzle = original_cipher.encipher(target_text)

    words = ['a','very','fine','day','for','test']
    phrases = ['a very fine day for a test!']
    scorer = RecognizedWordAndPhraseScorer(words=words, phrases=phrases)

    # Score should be 1 with the original cipher
    solved_state = CipherPuzzleState(puzzle=puzzle, cipher=original_cipher)
    assert abs(1.0 - scorer.get_reward(state=solved_state).value) <= 0.0001

    # Score should be close to zero with a random cipher
    random_state = CipherPuzzleState(puzzle=puzzle, cipher=Cipher())
    assert 0.1 > scorer.get_reward(state=random_state).value

def test_integration():
    plain_text = 'these are words'
    '''
    t = a
    h = b
    e = c
    s = d
    a = e
    r = f
    w = g
    o = h
    d = i
    s = j
    '''
    puzzle = 'abcdc efc ghfij'
    initial_state = CipherPuzzleState(puzzle)

    words = ['these', 'are', 'words']
    phrases = ['these are words']

    env = CipherPuzzleEnvironment(initial_state=initial_state, words=words, phrases=phrases)
    assert env.state == initial_state

    agent = RandomAgent(state=initial_state, action_list=env.action_list)
    rewards = []

    # @TODO(jdt) this works but it is slow - is there a way to ensure a reward change in less
    # steps? Maybe we make the puzzle with a cipher, then initialize with that same cipher, and
    # expect reward to start at 1.0 and then change to something less than that within a few steps.
    for _ in range(100000):
        action = agent.choose_action(env.state, env.reward)
        env.update(action)
        rewards.append(env.reward.value)
        agent.learn(env.state, env.reward)        

    # At some point the reward should have changed
    assert len(set(rewards)) > 1
