# Programm zur Lösung eines Sudoku-Raetsels
# Autor: Hossein Omid Beiki
# Basierend auf: https://www.techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/
# Stand: 12.07.2021

import numpy as np

board = [
    [7, 8, 0, 4, 0, 0, 1, 2, 0],
    [6, 0, 0, 0, 7, 5, 0, 0, 9],
    [0, 0, 0, 6, 0, 1, 0, 7, 8],
    [0, 0, 7, 0, 4, 0, 2, 6, 0],
    [0, 0, 1, 0, 5, 0, 9, 3, 0],
    [9, 0, 4, 0, 6, 0, 0, 0, 5],
    [0, 7, 0, 3, 0, 0, 0, 1, 2],
    [1, 2, 0, 0, 0, 7, 4, 0, 0],
    [0, 4, 9, 2, 0, 6, 0, 0, 7]
]
board_np = np.array(board)


def solve_sudoku(board):
    solve(board, print=False)
    return board


def print_board(bo):
    for i in range(bo.shape[0]):
        if i // 3 == i / 3 and i != 0:
            print('- - - - - - - - - - - - - - -')
        for j in range(bo.shape[1]):
            if j // 3 == j / 3 and j != 0:
                print('| ', end='')

            print(bo[i][j].astype(int), ' ', end='')

            if j == bo.shape[1] - 1:
                print('')
    return None


def find_empty(bo):
    for i in range(bo.shape[0]):
        for j in range(bo.shape[1]):
            if bo[i, j] == 0:
                return i, j
    return None, None


def check_valid(bo, r, c):
    # check the row
    for i in range(bo.shape[0]):
        if bo[r][c] == bo[i][c] and r != i:
            return False
    # check the column
    for j in range(bo.shape[1]):
        if bo[r][c] == bo[r][j] and c != j:
            return False
    # check the square
    box_x = r // 3
    box_y = c // 3
    for i in range(box_x * 3, box_x * 3 + 3):
        for j in range(box_y * 3, box_y * 3 + 3):
            if bo[r][c] == bo[i][j] and c != j and r != i:
                return False
    return True


def solve(bo, **kwargs):
    if 'print' in kwargs:
        if kwargs['print']:
            print("")
            print_board(bo)
            print("")
    row, column = find_empty(bo)
    if row is None:
        # no empty cell anymore --> solved
        return True
    else:
        for i in range(1, 10):
            bo[row][column] = i
            if not check_valid(bo, row, column):
                bo[row][column] = 0
            else:
                if solve(bo):
                    return True
                bo[row][column] = 0
    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_board(board_np)
    # solve_sudoku(board_np, print=True)

    bo2 = solve_sudoku(board_np)
    print("")
    print("")
    print_board(bo2)

    print("")
    print("")
    print_board(board_np)
