from copy import deepcopy
import random
import numpy as np
import itertools
from time import time
import math

from mdp import MDP

def value_iteration(mdp: MDP, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    current_U = deepcopy(U_init)
    while True:
        new_U = deepcopy(current_U)
        delta = 0
        for state in list(itertools.product(range(mdp.num_row), range(mdp.num_col))):
            if mdp.board[state[0]][state[1]] == "WALL":
                continue
            if state in mdp.terminal_states:
                new_U[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])
                continue
            # new_U[state[0], state[1]] = mdp.board[state[0]][state[1]] + mdp.gamma * max([sum([...]) for action in mdp.actions])
            action_expected_values = {action: 0 for action in mdp.actions}
            for action in mdp.actions: # trying to take an action

                # Calculate expected value if action is taken
                for index, actual_action in enumerate(mdp.actions): # result from trying to take that action
                    next_state = mdp.step(state, actual_action)
                    prob_actual_action = mdp.transition_function[action][index]
                    action_expected_values[action] += prob_actual_action * current_U[next_state[0]][next_state[1]]

            # Update new utility according to Bellman
            best_action = max(action_expected_values, key=action_expected_values.get)
            current_reward = float(mdp.board[state[0]][state[1]])
            new_U[state[0]][state[1]] = current_reward + mdp.gamma * action_expected_values[best_action]
            delta = max(np.abs(new_U[state[0]][state[1]] - current_U[state[0]][state[1]]), delta)
        current_U = deepcopy(new_U)
        if delta * mdp.gamma < epsilon * (1 - mdp.gamma):
            return current_U


def get_policy(mdp: MDP, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    policy = deepcopy(U)
    
    for state in list(itertools.product(range(mdp.num_row), range(mdp.num_col))):
        if state in mdp.terminal_states or mdp.board[state[0]][state[1]] == "WALL":
            continue
        
        # Find action's values
        action_expected_values = {action: 0 for action in mdp.actions}
        for action in mdp.actions:
            for index, actual_action in enumerate(mdp.actions): # result from trying to take that action
                next_state = mdp.step(state, actual_action)
                prob_actual_action = mdp.transition_function[action][index]
                # current_reward = float(mdp.board[next_state[0]][next_state[1]])
                current_reward = float(mdp.board[state[0]][state[1]])
                action_expected_values[action] += prob_actual_action * (mdp.gamma * U[next_state[0]][next_state[1]] + current_reward)
        
        # Update policy to do the best action
        policy[state[0]][state[1]] = max(action_expected_values, key=action_expected_values.get)
    return policy


# Given an MDP, state and wanted action - it attempts to do the action and returns a bunch of useless info
def q_learning_step(mdp: MDP, state: tuple, action: str, probability_threshold: float) -> tuple[tuple, float, bool]:
    
    # Randomize the action
    random_num = random.random()
    for index, threshold in enumerate(probability_threshold[action]):
        if random_num < threshold:
            # random_action = list(mdp.actions.items())[index][0]
            random_action = list(mdp.actions)[index]
            break
    
    # Return info
    next_state = mdp.step(state, random_action)
    next_state_reward = float(mdp.board[next_state[0]][next_state[1]])
    next_is_terminal = next_state in mdp.terminal_states
    return next_state, next_state_reward, next_is_terminal


# Returns the upper bound of each action
# e.g. if the transition function is [0.1 0.0 0.1 0.8], it returns:
#                                    [0.1 0.1 0.9 1.0]
def generate_probability_threshold(transition_function: dict):
    threshold = {} # action: list
    for action in transition_function:
        prob_arr = []
        for actual_action_index in range(len(transition_function[action])):
            if actual_action_index == 0:
                prob_arr.append(transition_function[action][actual_action_index])
            else:
                prob_arr.append(prob_arr[actual_action_index-1] + transition_function[action][actual_action_index])
        threshold[action] = deepcopy(prob_arr)
    return threshold

def q_learning(mdp: MDP, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #
    # q_table = np.zeros((mdp.num_row, mdp.num_col, len(mdp.actions)))
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    is_state_legal = lambda state: mdp.board[state[0]][state[1]] != "WALL"
    states = list(filter(is_state_legal, states))
    state_to_index = {state: index for index, state in enumerate(states)}
    num_states = len(states)
    q_table = np.zeros((num_states, len(mdp.actions)))
    for state in mdp.terminal_states:
        q_table[state_to_index[state]] = mdp.board[state[0]][state[1]]

    probability_threshold = generate_probability_threshold(mdp.transition_function)
    
    for episode in range(total_episodes): # episodes
        state = init_state
        for _ in range(max_steps):
            explore_exploit_threshold = random.uniform(0,1)
            
            # Take action (random / best) according to explore/exploit
            if explore_exploit_threshold > epsilon:
                action_index = np.argmax(q_table[state_to_index[state]])
            else:
                action_index = random.randint(0,3)
            
            # Take action
            action = list(mdp.actions)[action_index]
            new_state, _, is_final = q_learning_step(mdp, state, action, probability_threshold)
            reward = float(mdp.board[state[0]][state[1]])
            # Update Q(s,a)
            learning_value = reward + mdp.gamma * np.max(q_table[state_to_index[new_state]]) - q_table[state_to_index[state]][action_index]
            q_table[state_to_index[state], action_index] = q_table[state_to_index[state]][action_index] + learning_rate * learning_value
            state = new_state
            if is_final:
                break
            
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
    return q_table


def q_table_policy_extraction(mdp: MDP, qtable):
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    is_state_legal = lambda state: mdp.board[state[0]][state[1]] != "WALL"
    states = list(filter(is_state_legal, states))
    state_to_index = {state: index for index, state in enumerate(states)}
    
    policy = [["none" for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    # policy = np.zeros((mdp.num_row, mdp.num_col), dtype=object)
    # For each state, extract its policy from the given qtable
    for state in states:
        if state in mdp.terminal_states or mdp.board[state[0]][state[1]] == "WALL":
            continue
        
        action_index = np.argmax(qtable[state_to_index[state]])
        action = list(mdp.actions)[action_index]
        policy[state[0]][state[1]] = action
    return policy


def policy_evaluation(mdp: MDP, policy):
    policy = np.array(policy)
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    is_state_legal = lambda state: mdp.board[state[0]][state[1]] != "WALL"
    states = list(filter(is_state_legal, states))
    state_to_index = {state: index for index, state in enumerate(states)}
    num_states = len(states)
    I = np.identity(num_states)
    utility_value = np.zeros((num_states))
    probability_matrix = np.zeros((num_states, num_states))
    for state in states:
        if state in mdp.terminal_states or mdp.board[state[0]][state[1]] == "WALL":
            continue
        for index, action in enumerate(mdp.actions.keys()):
            next_state = mdp.step(state, action)
            next_state_probability = mdp.transition_function[policy[state]][index]
            probability_matrix[state_to_index[state]][state_to_index[next_state]] += next_state_probability
    reward_vector = np.array([float(mdp.board[state[0]][state[1]]) for state in states])
    utility_value = np.dot(np.linalg.inv(I - np.dot(mdp.gamma, probability_matrix)), reward_vector)
    utility = np.zeros((mdp.num_row, mdp.num_col))
    for state in states:
        utility[state[0]][state[1]] = utility_value[state_to_index[state]]
    return utility


def policy_iteration(mdp: MDP, policy_init):
    changed = True
    policy = np.array(policy_init)
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    is_state_legal = lambda state: mdp.board[state[0]][state[1]] != "WALL"
    states = list(filter(is_state_legal, states))
    while changed:
        changed = False
        utility = np.array(policy_evaluation(mdp, policy))
        # For each state, check if there's a better action for the current policy
        for state in states:
            if state in mdp.terminal_states or mdp.board[state[0]][state[1]] == "WALL":
                continue
            current_expected_value = 0
            policy_action = policy[state]
            
            for real_action_index, real_action in enumerate(mdp.actions):
                current_expected_value += mdp.transition_function[policy_action][real_action_index] * utility[mdp.step(state, real_action)]
            max_expected_value = -math.inf
            best_action = 0
            
            # Calculate action expected value and update best action if needed
            for action in mdp.actions:
                action_expected_value = 0
                for real_action_index, real_action in enumerate(mdp.actions):
                    action_expected_value += mdp.transition_function[action][real_action_index] * utility[mdp.step(state, real_action)]
                if action_expected_value > max_expected_value:
                    max_expected_value = action_expected_value
                    best_action = action
            # Update state's policy
            if max_expected_value > current_expected_value:
                changed = True                
                policy[state] = best_action
    return policy
