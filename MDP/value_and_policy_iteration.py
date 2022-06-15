import pdb
from copy import deepcopy
import numpy as np

def calcU(mdp,U,board,state_row,state_col,action):
    if (state_row,state_col) in mdp.terminal_states:
        return int(mdp.board[state_row][state_col])
    val = board[state_row][state_col]
    print(board[state_row][state_col],state_row,state_col,val,board,U)
    # print(val)
    #up,down,left,right
    #Up
    row,col = mdp.step((state_row, state_col), "UP")
    # print(mdp.transition_function[action][0],U[row][col],row,col)
    val+=mdp.gamma * U[row][col] * mdp.transition_function[action][0]
    #down
    row, col = mdp.step((state_row, state_col), "DOWN")
    # print(mdp.transition_function[action][1], U[row][col], row, col)
    val += mdp.gamma * U[row][col] * mdp.transition_function[action][1]
    #left
    row, col = mdp.step((state_row, state_col), "LEFT")
    # print(mdp.transition_function[action][2], U[row][col], row, col)
    val += mdp.gamma * U[row][col] * mdp.transition_function[action][2]
    #right
    row, col = mdp.step((state_row, state_col), "RIGHT")
    # print(mdp.transition_function[action][3], U[row][col], row, col)
    val += mdp.gamma * U[row][col] * mdp.transition_function[action][3]
    # print(val)
    print(val)
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
                U_tag[i][j] = int(U_tag[i][j])
    board = deepcopy(U_tag)
    print(board,U_tag)
    U = deepcopy(U_init)
    while True:
        delta = 0
        for state_row in range(0,np.array(mdp.board).shape[0]):
            for state_col in range(0,np.array(mdp.board).shape[1]):
                if mdp.board[state_row][state_col] == "WALL" :
                    continue
                # print(f'tag: {U_tag}')
                # print(f'reg: {U}')
                # print(f'board: {board}')
                U_tag[state_row][state_col] = max([calcU(mdp,U,board,state_row,state_col,s) for s in mdp.actions])
                # print(f'tag: {U_tag}')
                # print(f'reg: {U}')
                # print(f'board: {board}')
                # print(U[state_row][state_col],U_tag[state_row][state_col])
                # print(delta,f'{abs((U_tag[state_row][state_col] - U[state_row][state_col]))}')
                delta = max(delta, abs(U_tag[state_row][state_col] - U[state_row][state_col]))
                # print(f'delta {delta}')
        board = deepcopy(U_tag)
        U = deepcopy(U_tag)
        if not mdp.gamma == 1:
            if delta < epsilon*(1-mdp.gamma)/mdp.gamma:
                break
        else:
            if delta == 0:
                break
    return U
    # ========================

def getValue(mdp,U,row,col,action,max_row,max_col):
    next_state = tuple(map(sum, zip((row,col), mdp.actions[action])))
    # collide with a wall
    if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= max_row or next_state[1] >= max_col or \
            U[next_state[0]][next_state[1]] == None:
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
            if U[row][col] is None:
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
