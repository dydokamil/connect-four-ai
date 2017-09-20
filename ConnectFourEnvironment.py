import numpy as np


class ConnectFourEnvironment:
    def __init__(self):
        self.__grid__ = np.zeros((6, 7))
        self.__needs_reset__ = True
        self.__reds_turn__ = True

    def render(self):
        for row in self.__grid__:
            for element in row:
                if element == .0:
                    element = '#'
                elif element == 1.:
                    element = 'o'
                elif element == 2.:
                    element = 'x'
                element = '#' if element == 0.0 else element
                print(element, '', end='')
            print()

    def reset(self):
        self.__needs_reset__ = False
        return np.copy(self.__grid__)

    def step(self, action):
        if self.__needs_reset__:
            raise RuntimeError("Call reset function first.")
        if np.any(self.__grid__[:, action] == 0):
            free_rows = np.where(self.__grid__[:, action] == 0)[0]
            lowest_row = free_rows[-1]
            self.__grid__[lowest_row, action] = 1 if self.__reds_turn__ else 2

        self.__reds_turn__ = not self.__reds_turn__
        winner = self.__detect_termination__()

        terminated = False
        if winner != 0:
            self.__needs_reset__ = True
            terminated = True

        return np.copy(self.__grid__), winner, terminated

    def __detect_termination__(self):
        def rolling_window(arr, start_idx, end_idx, size):
            for i in range(end_idx + 1 - start_idx - size):
                for row in arr:
                    yield row[start_idx + i:size + i]

        def rolling_square(arr, start_x, end_x, start_y, end_y, size):
            for i in range(end_x + 1 - start_x - size):
                for j in range(end_y + 1 - start_y - size):
                    yield arr[start_x + i:size + i, start_y + j:size + j]

        # detect column-wise and row-wise
        for i in range(2):
            grid = self.__grid__
            if i == 1:
                grid = np.rot90(grid)

            gen = rolling_window(grid, 0, grid.shape[1], 4)

            while True:
                try:
                    window = next(gen)
                    unique = np.unique(window)
                    if len(unique) == 1:
                        if unique == 1.:
                            print("P1 won.")
                            return 5.
                        elif unique == 2.:
                            print("P2 won.")
                            return -5.
                except StopIteration:
                    break

        # detect diagonals
        for i in range(2):
            grid = self.__grid__
            if i == 1:
                grid = np.rot90(grid)

            gen = rolling_square(grid, 0, grid.shape[0], 0, grid.shape[1], 4)

            while True:
                try:
                    window = next(gen)
                    unique = np.unique(np.diag(window))
                    if len(unique) == 1:
                        if unique == 1.:
                            print("P1 won!")
                            return 5.
                        elif unique == 2.:
                            print("P2 won!")
                            return -5.
                except StopIteration:
                    break

        if np.all(self.__grid__ != 0):
            print("Draw.")
            return -1

        return 0
