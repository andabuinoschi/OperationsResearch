import math

import numpy as np

from InputParser import InputParser

np.random.seed(42)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PrimalDualInteriorPoint:

    def __init__(self, input):
        p = 18

        input_parser = InputParser(input)
        self.tableau = input_parser.get_tableau()
        self.all_variables = input_parser.get_all_variables()
        self.basic_variables = input_parser.get_basic_variables()
        self.m = self.tableau.shape[0] - 1
        self.n = self.tableau.shape[1] - 1

        self.epsilon = 10 ** (-1 * p)
        self.k = 0
        self.q = 6

        if self.n < 13:
            self.theta = 0.5
        else:
            self.theta = 1 - 3.5 / math.sqrt(self.n)

        self.MAX_ITER = 100

    @staticmethod
    def matmul3(a, b, c):
        return np.matmul(np.matmul(a, b), c)

    @staticmethod
    def matrix_euclidean_distance(m1, m2):
        s = 0
        for m1array, m2array in zip(m1, m2):
            for m1element, m2element in zip(m1array, m2array):
                s += (m1element - m2element) ** 2

        return math.sqrt(s)

    def run(self):
        x = np.random.rand(self.n) + 0.1
        s = np.random.rand(self.n) + 0.1
        y = np.random.rand(self.m) + 0.1
        niu = 10
        A = self.tableau[:-1, :-1]
        Atranspose = np.transpose(A)
        it = 0

        while True:
            sk = np.diag(s)
            dk = np.diag(x / s)

            rop = self.tableau[:self.m, self.n] - np.matmul(A, x)
            rod = self.tableau[self.m, :self.n] - np.matmul(Atranspose, y) - s
            vk = niu - np.multiply(x, s)

            deltaY = np.matmul(np.linalg.inv(-1 * (self.matmul3(A, dk, Atranspose))),
                               (self.matmul3(A, np.linalg.inv(sk), vk) - self.matmul3(A, dk, rod) - rop))

            deltaS = -1 * np.matmul(Atranspose, deltaY) + rod
            deltaX = np.matmul(np.linalg.inv(sk), vk) - np.matmul(dk, deltaS)

            dxarr = [-1 * x_ / diag for x_, diag in zip(x, deltaX) if diag < 0]

            if len(dxarr) == 0:
                alphaX = 1
            else:
                alphaX = np.min(dxarr)

            dsarr = [-1 * s_ / diag for s_, diag in zip(s, deltaS) if diag < 0]

            if len(dsarr) == 0:
                alphaS = 1
            else:
                alphaS = np.min(dsarr)

            alphaMax = min(alphaX, alphaS)
            alpha = 0.999999 * alphaMax

            new_x = x + alpha * deltaX
            new_y = y + alpha * deltaY
            new_s = s + alpha * deltaS
            niu = niu * self.theta
            it += 1

            tmp_arr_new = np.asarray([new_x, new_y, new_s])
            tmp_arr_old = np.asarray([np.array(x), np.array(y), np.array(s)])

            if (it > self.MAX_ITER or np.matmul(np.transpose(new_x),
                                                new_s) <= self.epsilon or self.matrix_euclidean_distance(tmp_arr_new,
                                                                                                         tmp_arr_old) >= 10 ** self.q):
                x = new_x
                y = new_y
                s = new_s
                break

            x = new_x
            y = new_y
            s = new_s

        xTs = np.matmul(np.transpose(x), s)
        if xTs <= self.epsilon:
            print(f"Algorithm finished at iteration {it}")
            print(f"{bcolors.OKBLUE}The values for x are {x}{bcolors.ENDC}")
            print(f"The values for y are {y}")
            print(f"The values for s are {s}")
            print(f"Duality gap is: {xTs}")
            print(f"{bcolors.OKGREEN}Function value is {np.dot(self.tableau[self.m][:self.n], x)}{bcolors.ENDC}")
            for row in self.tableau[:self.m, :]:
                print(
                    f"Constraint {row[:self.n]}, expected value: {row[self.n]}, actual value: {np.dot(row[:self.n], x)}")


if __name__ == '__main__':
    # hw a
    input = [[2, 1, 3, 0, '=', 35], [-1, 1, 0, 2, '=', 12], ['min', 1, 3, 1, 0]]
    # hw b
    # input = [[2, 1, 2, 0, 0, 0, '=', 4], [2, 3, 0, 1, 0, 0, '=', 3], [4, 1, 0, 0, 3, 0, '=', 5],
    #          [1, 5, 0, 0, 0, 1, '=', 2], ['min', -2, -1, 0, 0, 0, 0]]
    # example course
    # input = [[-2, 1, '<=', 2], [-1, 2, '<=', 7], [1, 2, '<=', 3], ['min', -1, -2]]

    pdip = PrimalDualInteriorPoint(input)
    pdip.run()
