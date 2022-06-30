import pdb
from copy import deepcopy
import numpy as np

def calcU(mdp,U,state_row,state_col,action):
    #Up
    val = 0.0
    i=0
    actions_list = list(mdp.actions.keys())
    for item in actions_list:
        row, col = mdp.step((state_row, state_col), item)
        val += U[row][col] * mdp.transition_function[action][i]
        i+=1
    return val



def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U_tag = deepcopy(mdp.board)
    for i in range(0, np.array(U_tag).shape[0]):
        for j in range(0, np.array(U_tag).shape[1]):
            if U_tag[i][j] != "WALL":
                U_tag[i][j] = float(U_tag[i][j])
    U = deepcopy(U_init)
    while True:
        delta = 0
        for state_row in range(0,np.array(mdp.board).shape[0]):
            for state_col in range(0,np.array(mdp.board).shape[1]):
                if mdp.board[state_row][state_col] == "WALL" :
                    continue
                if (state_row, state_col) in mdp.terminal_states:
                    U_tag[state_row][state_col] = float(mdp.board[state_row][state_col])
                else:
                    U_tag[state_row][state_col] = float(mdp.board[state_row][state_col])
                    U_tag[state_row][state_col] += mdp.gamma* max([calcU(mdp,U,state_row,state_col,s) for s in mdp.actions])
                delta = max(delta, abs(U_tag[state_row][state_col] - U[state_row][state_col]))
        U = deepcopy(U_tag)
        if mdp.gamma == 1:
            if delta == 0:
                return U
        else:
            if (delta < (epsilon*(1-mdp.gamma)/mdp.gamma)):
                return U

    # ========================

def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(U)
    for row in range(0, np.array(U).shape[0]):
        for col in range(0, np.array(U).shape[1]):
            policy[row][col] = -1
    for row in range(0, np.array(U).shape[0]):
        for col in range(0, np.array(U).shape[1]):
            if U[row][col] =="WALL" or (row, col) in mdp.terminal_states:
                continue
            max_action , max_u = None, float("-inf")
            for action in mdp.actions:
                u = calcU(mdp,U,row,col,action)
                if u > max_u:
                    maxAction, max_u = action, u
            policy[row][col] = maxAction
    return policy
    # ========================

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    columns = len(mdp.board[0])
    rows = len(mdp.board)
    matrix = np.zeros((int(rows*columns), int(rows*columns)))
    for row in range(0,np.array(mdp.board).shape[0]):
        for col in range(0,np.array(mdp.board).shape[1]):
            if mdp.board[row][col] == "WALL":
                continue
            if(row, col) in mdp.terminal_states:
                continue
            else:
                i = 0
                actions_list = list(mdp.actions.keys())
                for item in actions_list:
                    curr_row, curr_col = mdp.step((row, col), item)
                    matrix[col + (row * columns)][curr_col + (curr_row * columns)] += mdp.gamma*mdp.transition_function[policy[row][col]][i]
                    i+=1

    n_list = []
    for row in range(0,np.array(mdp.board).shape[0]):
        for col in range(0,np.array(mdp.board).shape[1]):
            if mdp.board[row][col] != 'WALL':
                n_list.append(float(mdp.board[row][col]))
            else:
                n_list.append(0.0)
    inverted = np.linalg.inv(np.subtract(np.identity(len(n_list)),matrix)).dot(n_list)

    U = np.zeros((rows, columns))
    for row in range(0,np.array(mdp.board).shape[0]):
        for col in range(0,np.array(mdp.board).shape[1]):
            if mdp.board[row][col] != 'WALL':
                U[row][col] = inverted[col + (row * columns)]
            else:
                U[row][col] = 0.0
    return U
    # ========================

def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    policy = deepcopy(policy_init)
    while True:
        unchanged = True
        U = policy_evaluation(mdp,policy)
        for state_row in range(0,np.array(mdp.board).shape[0]):
            for state_col in range(0,np.array(mdp.board).shape[1]):
                if mdp.board[state_row][state_col] == "WALL" :
                    continue
                if (state_row, state_col) in mdp.terminal_states:
                    continue
                else:
                    orig_val, max_action = float("-inf"), None
                    for action in mdp.actions:
                        val = calcU(mdp, U, state_row, state_col, action)
                        if val > orig_val:
                            max_action = action
                            orig_val = val
                    original_action = policy[state_row][state_col]
                    if max_action != original_action:
                        policy[state_row][state_col] = max_action
                        unchanged = False
        if unchanged:
            return policy
    # ========================
