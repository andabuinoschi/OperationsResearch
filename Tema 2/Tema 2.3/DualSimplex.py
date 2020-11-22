import numpy as np
from InputParser import InputParser


class DualSimplex:
    """
    Dual Simplex algorithm
    The constructor of the object calls the InputParser in order to transform the input into
    standard form
    """

    def __init__(self, input: np.array = None, tableau=None, all_variables=None, basic_variables=None):
        if input:
            input_parser = InputParser(input)
            self.tableau = input_parser.get_tableau()
            self.all_variables = input_parser.get_all_variables()
            self.basic_variables = input_parser.get_basic_variables()
        elif tableau is not None and all_variables is not None and basic_variables is not None:
            self.tableau = tableau
            self.all_variables = all_variables
            self.basic_variables = basic_variables

        self.m = self.tableau.shape[0] - 1
        self.n = self.tableau.shape[1] - 1

    def solve_dual_simplex(self):
        print("First tableau of Dual Simplex is: \n{0}".format(self.tableau))
        print("The first basic variables are: {0} = {1}".format(
            self.basic_variables, self.tableau[:self.m, self.n]))
        print("------------------------ DUAL SIMPLEX ------------------------")
        while np.any(self.tableau[:self.m, self.n] < 0):
            # Get the leaving variable in basic variables according to Bland's rule
            labels_basic_variables = [int(basic_variable[1:]) for basic_variable in self.basic_variables]
            basic_variables_values = self.tableau[:self.m, self.n]
            mask_leaving_variables = np.ma.MaskedArray(labels_basic_variables, basic_variables_values >= 0)
            pivot_index_row = mask_leaving_variables.argmin()  # k
            # Check for infeasibility, the unboundness of the dual problem
            if np.all(self.tableau[pivot_index_row, :self.n] >= 0):
                return "The problem is infeasible and the dual problem is unbounded."
            pivot_row = self.tableau[pivot_index_row, :self.n]
            # Compute ratios to compare them for the entering variable in the basis
            valid_pivot_row = np.ma.MaskedArray(pivot_row, pivot_row >= 0)
            ratios = np.abs(self.tableau[self.m, :self.n] / valid_pivot_row)
            # Choose the entering variable according to Bland's rule
            # If multiple variables have the minimum ratios, then choose the one with smallest index.
            min_ratio = ratios.min()
            # Since in all_variables we have the label in the right order, we do not need masked arrays anymore.
            pivot_index_column = np.min(np.where(ratios == min_ratio))  # l
            entering_variable = self.all_variables[pivot_index_column]
            leaving_variable = self.basic_variables[pivot_index_row]
            self.basic_variables[pivot_index_row] = entering_variable
            # Pivoting rule
            for index_row in range(0, self.m + 1):
                for index_column in range(0, self.n + 1):
                    if index_row != pivot_index_row and index_column != pivot_index_column:
                        self.tableau[index_row, index_column] = (self.tableau[index_row, index_column] *
                                                                 self.tableau[
                                                                     pivot_index_row, pivot_index_column] -
                                                                 self.tableau[index_row, pivot_index_column] *
                                                                 self.tableau[pivot_index_row, index_column]) / \
                                                                self.tableau[pivot_index_row, pivot_index_column]
            for index_row in range(0, self.m + 1):
                if index_row != pivot_index_row:
                    self.tableau[index_row, pivot_index_column] = 0
            for index_column in range(0, self.n + 1):
                if index_column != pivot_index_column:
                    self.tableau[pivot_index_row, index_column] /= self.tableau[
                        pivot_index_row, pivot_index_column]
            self.tableau[pivot_index_row, pivot_index_column] = 1
            print("The new tableau of Dual Simplex is: \n{0}".format(np.round(self.tableau, 3)))
            print("The new basic variables are the following: {0} = {1}".format(self.basic_variables,
                                                                        np.round(self.tableau[:self.m, self.n], 3).tolist()))
        print("The solution is {0} with basic variables {1} = {2}".format(
            np.round(-self.tableau[self.m, self.n], 3),
            self.basic_variables,
            np.round(self.tableau[:self.m, self.n], 3).tolist()))

        print("---------------------------------------------------------")
        return self.tableau, self.basic_variables


if __name__ == '__main__':
    # The fist rows in input are constraints and the last line is the objective
    # Cursul 5 - example -> slide 37
    # input = [[-3, 2, '<=', -4], [-1, -2, '<=', -3], ['min', 2, 3, 0]]

    # Seminar 6
    # input = [[2, -3, 2, '>=', 3], [-1, 1, 1, '>=', 5], ['min', 5, 2, 8, 0]]

    # Tema 2.3 - ex. 1
    # subpunctul a)
    # input = [[1, 2, 3, '<=', 4], [2, 2, 1, '>=', 5], ['min', 3, 4, 2, 0]]

    # subpunctul b)
    input = [[4, 2, 1, 1, '>=', 4], [5, 1, -1, 0, '>=', 5], ['min', 24, 6, 1, 1, 0]]

    dual_simplex = DualSimplex(input)
    dual_simplex.solve_dual_simplex()
