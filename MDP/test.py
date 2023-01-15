from mdp import MDP


board = 'board'
terminal_states = 'terminal_states'
transition_function = 'transition_function'

board_env = []
with open(board, 'r') as f:
    for line in f.readlines():
        row = line[:-1].split(',')
        board_env.append(row)

terminal_states_env = []
with open(terminal_states, 'r') as f:
    for line in f.readlines():
        row = line[:-1].split(',')
        terminal_states_env.append(tuple(map(int, row)))

transition_function_env = {}
with open(transition_function, 'r') as f:
    for line in f.readlines():
        action, prob = line[:-1].split(':')
        prob = prob.split(',')
        transition_function_env[action] = tuple(map(float, prob))

# initialising the env
mdp = MDP(board=board_env,
            terminal_states=terminal_states_env,
            transition_function=transition_function_env,
            gamma=0.9)

from pprint import pprint

# print(mdp.board[1][1])
# print(mdp.num_col)
# mdp.print_rewards()
# print(mdp.transition_function)
# print(mdp.stp())
print(mdp.terminal_states)
# # mdp.print_policy()
# mdp.print_rewards()
# # mdp.print_utility()