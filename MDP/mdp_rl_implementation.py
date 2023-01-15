from copy import deepcopy
import random
import numpy as np
import itertools
from time import time
# import math

# itertools.product()  # TODO


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
                current_reward = float(mdp.board[next_state[0]][next_state[1]])
                action_expected_values[action] += prob_actual_action * (mdp.gamma * U[next_state[0]][next_state[1]] + current_reward)
        
        # Update policy to do the best action
        policy[state[0]][state[1]] = max(action_expected_values, key=action_expected_values.get)
    return policy


def q_learning(mdp: MDP, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
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
    q_table = np.zeros((mdp.num_row, mdp.num_col, len(mdp.actions)))
    # Loop for each episode:
    #   Initialize S
    #   Loop for each step of episode:
    #       Choose A from S using policy derived from Q (e.g. epsilon-greedy)
    #       Take action A, observe R, S'
    #       Q(S,A) <- Q(S,A) (old) + alpha(R+ gamma * max (Q(S', a) - Q(S,A) (old)))
    #       S <- S'
    #   until S is terminal    
    state = init_state
    for episode in range(total_episodes): # episodes
        episode_start = time()
        for _ in range(max_steps):
            explore_exploit_threshold = random.uniform(0,1)
            
            if explore_exploit_threshold > epsilon:
                action_index = np.argmax(q_table[state])
            else:
                action_index = random.randint(0,3)
            
            # Take action
            action = list(mdp.actions)[action_index]
            random_action = np.random.choice(list(mdp.actions), p=mdp.transition_function[action])
            # random_action = random.choice(list(mdp.actions)) #### TODO FIX MEEEEEEE
            new_state = mdp.step(state, random_action)
            reward = float(mdp.board[new_state[0]][new_state[1]])
            is_final = new_state in mdp.terminal_states

            # Update Q(s,a)
            learning_value = reward + mdp.gamma * np.max(q_table[new_state]) - q_table[state][action_index]
            q_table[new_state][action_index] = q_table[state][action_index] + learning_rate * learning_value
            
            state = new_state
            
            if is_final:
                break
        episode_end = time()
        print(episode_end - episode_start)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
    return q_table


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
