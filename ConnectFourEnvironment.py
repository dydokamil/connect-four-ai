import numpy as np


class ConnectFourEnvironment:
    def __init__(self):
        self.__grid__ = np.zeros((6, 7))
        self.__needs_reset__ = True
        self.__yellows_turn__ = True
        self.__reward__ = 0

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
        if self.__reward__ != 0:
            raise RuntimeError("Call get_reward function first.")
        self.__needs_reset__ = False
        return np.copy(self.__grid__)

    def yellows_turn(self):
        return self.__yellows_turn__

    def get_random_action(self):
        if not self.__needs_reset__:
            free_cols = np.unique(np.where(self.__grid__ == 0)[1])
            return np.random.choice(free_cols)

    def get_state(self):
        return np.copy(self.__grid__)

    def get_reward(self):
        r = self.__reward__
        self.__reward__ = 0
        return r

    def step(self, action):
        if self.__needs_reset__:
            raise RuntimeError("Call reset function first.")
        if np.any(self.__grid__[:, action] == 0):
            free_rows = np.where(self.__grid__[:, action] == 0)[0]
            lowest_row = free_rows[-1]
            self.__grid__[lowest_row, action] = \
                1 if self.__yellows_turn__ else 2
        else:  # prohibited move
            self.__needs_reset__ = True
            print("Move prohibited!")
            self.__reward__ = -500
            self.__needs_reset__ = True

        self.__yellows_turn__ = not self.__yellows_turn__
        if self.__reward__ == 0:
            self.__reward__ = self.__detect_termination__()

        if self.__reward__ != 0:
            self.__needs_reset__ = True

    def is_finished(self):
        return self.__needs_reset__

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
                            print("P1 won.")
                            return 5.
                        elif unique == 2.:
                            print("P2 won.")
                            return -5.
                except StopIteration:
                    break

        if np.all(self.__grid__ != 0):
            print("Draw.")
            return -1

        return 0
