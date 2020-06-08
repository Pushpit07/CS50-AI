"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board == initial_state:
        return X

    xcounter = 0
    ocounter = 0
    for row in board:
        xcounter += row.count(X)
        ocounter += row.count(O)

    if xcounter == ocounter:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_moves.append([i, j])
    return possible_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    boardcopy = copy.deepcopy(board)
    try:
        if boardcopy[action[0]][action[1]] != EMPTY:
            raise IndexError
        else:
            boardcopy[action[0]][action[1]] = player(boardcopy)
            return boardcopy
    except IndexError:
        print('Spot already occupied')


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    columns = []
    # Checks rows
    for row in board:
        xcounter = row.count(X)
        ocounter = row.count(O)
        if xcounter == 3:
            return X
        if ocounter == 3:
            return O

    # Checks columns
    for j in range(len(board)):
        column = [row[j] for row in board]
        columns.append(column)

    for j in columns:
        xcounter = j.count(X)
        ocounter = j.count(O)
        if xcounter == 3:
            return X
        if ocounter == 3:
            return O

    # Checks diagonals
    if board[0][0] == O and board[1][1] == O and board[2][2] == O:
        return O
    if board[0][0] == X and board[1][1] == X and board[2][2] == X:
        return X
    if board[0][2] == O and board[1][1] == O and board[2][0] == O:
        return O
    if board[0][2] == X and board[1][1] == X and board[2][0] == X:
        return X

    # No winner/tie
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    empty_counter = 0
    for row in board:
        empty_counter += row.count(EMPTY)
    if empty_counter == 0:
        return True
    elif winner(board) is not None:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    if player(board) == X:
        return max_value_alpha_beta(board, float("-inf"), float("inf"))[1]
    elif player(board) == O:
        return min_value_alpha_beta(board, float("-inf"), float("inf"))[1]
    else:
        raise Exception("bug in minimax algorithm")


def max_value_alpha_beta(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    v = float("-inf")
    best = None
    for action in actions(board):
        min_v = min_value_alpha_beta(result(board, action), alpha, beta)[0]
        if min_v > v:
            v = min_v
            best = action
        alpha = max(alpha, v)
        if beta <= alpha:
            break
    return v, best

def min_value_alpha_beta(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    v = float("inf")
    best = None
    for action in actions(board):
        max_v = max_value_alpha_beta(result(board, action), alpha, beta)[0]
        if max_v < v:
            v = max_v
            best = action
        beta = min(beta, v)
        if beta <= alpha:
            break
    return v, best
