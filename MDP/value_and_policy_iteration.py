import pdb
from copy import deepcopy
import numpy as np

def calcU(mdp,U,state_row,state_col,action):
    # print("-"*30 + f"{state_row, state_col, action}" + "-"*30)
    #Up
    val = 0.0
    # for action in mdp.transition_function[action]:
    #     print(action)
    row,col = mdp.step((state_row, state_col), "UP")
    val+= U[row][col] * mdp.transition_function[action][0]
    #down
    row, col = mdp.step((state_row, state_col), "DOWN")
    val += U[row][col] * mdp.transition_function[action][1]
    #left
    row, col = mdp.step((state_row, state_col), "RIGHT")
    val += U[row][col] * mdp.transition_function[action][2]
    #right
    row, col = mdp.step((state_row, state_col), "LEFT")
    val += U[row][col] * mdp.transition_function[action][3]
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
                delta = max(delta, round(float(abs(U_tag[state_row][state_col] - U[state_row][state_col])),4))
        U = deepcopy(U_tag)
        if not mdp.gamma == 1:
            if (delta < (epsilon*(1-mdp.gamma)/mdp.gamma)):
                return U
        else:
            if delta == 0:
                return U
    # ========================

def getValue(mdp,U,row,col,action,max_row,max_col):
    next_state = tuple(map(sum, zip((row,col), mdp.actions[action])))
    # collide with a wall
    if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= max_row or next_state[1] >= max_col or \
            U[next_state[0]][next_state[1]] == "WALL":
        next_state = (row,col)
    return next_state

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
            if U[row][col] =="WALL":
                continue
            max_action , max_u = None, float("-inf")
            for action in mdp.actions:
                ret_row,ret_col = getValue(mdp,U,row,col,action,np.array(U).shape[0],np.array(U).shape[1])
                u = U[ret_row][ret_col]
                if u>max_u:
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
    matrix = np.zeros((rows*columns), (rows*columns))
    for row in range(0,np.array(mdp.board).shape[0]):
        for col in range(0,np.array(mdp.board).shape[1]):
            if not mdp.board[row][col] == "WALL" and not (row, col) in mdp.terminal_states:
                pass
            elif mdp.board[row][col] == "WALL":
                pass

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
                    orig_val, max_action= float("-inf"), None
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
