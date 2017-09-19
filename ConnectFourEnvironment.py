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
        if self.__detect_termination__():
            self.__needs_reset__ = True
            return np.copy(self.__grid__), 0, False

    def __detect_termination__(self):
        for i in range(7):
            shifted_cols = np.roll(self.__grid__, i, axis=1)
            windows = shifted_cols[:, :4]
            if len(np.unique(shifted_cols[:, :4])) == 1:
                print("Somebody won.")

        pass
