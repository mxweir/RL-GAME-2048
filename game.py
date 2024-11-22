import numpy as np
import random

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x][y] = 2 if random.random() < 0.9 else 4

    def slide_and_merge_row(self, row):
        new_row = [i for i in row if i != 0]
        merged_row = []
        skip = False
        for i in range(len(new_row)):
            if skip:
                skip = False
                continue
            if i + 1 < len(new_row) and new_row[i] == new_row[i + 1]:
                merged_row.append(new_row[i] * 2)
                skip = True
            else:
                merged_row.append(new_row[i])
        return merged_row + [0] * (len(row) - len(merged_row))

    def move_left(self):
        for i in range(4):
            self.board[i] = self.slide_and_merge_row(self.board[i])

    def move_right(self):
        for i in range(4):
            self.board[i] = self.slide_and_merge_row(self.board[i][::-1])[::-1]

    def move_up(self):
        self.board = self.board.T
        self.move_left()
        self.board = self.board.T

    def move_down(self):
        self.board = self.board.T
        self.move_right()
        self.board = self.board.T

    def move(self, direction):
        original_board = self.board.copy()
        if direction == 'up':
            self.move_up()
        elif direction == 'down':
            self.move_down()
        elif direction == 'left':
            self.move_left()
        elif direction == 'right':
            self.move_right()

        if not np.array_equal(original_board, self.board):
            self.add_new_tile()

    def is_game_over(self):
        for move in ['up', 'down', 'left', 'right']:
            test_game = Game2048()
            test_game.board = self.board.copy()
            test_game.move(move)
            if not np.array_equal(self.board, test_game.board):
                return False
        return True

    def get_score(self):
        return np.sum(self.board)

    def print_board(self):
        print(self.board)
