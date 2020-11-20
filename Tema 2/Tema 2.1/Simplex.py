import numpy as np
from InputParser import InputParser


class Simplex:
    """
    Simplex algorithm
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

    def solve_simplex(self, additional_optimal_solution=False):
        print("First tableau of Simplex is: \n{0}".format(self.tableau))
        print("The first basic variables are: {0} = {1}".format(
            self.basic_variables, self.tableau[:self.m, self.n]))
        print("------------------------ SIMPLEX ------------------------")
        while np.any(self.tableau[self.m, :self.n] < 0):
            # Get the entering variable in basic variables according to Bland's rule
            pivot_index_column = np.min(np.where(self.tableau[self.m, :self.n] < 0))  # l
            # Check the unboundness problem
            if np.all(self.tableau[:self.m, pivot_index_column] <= 0):
                return "The problem is unbounded"
            pivot_column = self.tableau[:self.m, pivot_index_column]
            # Compute ratios
            valid_pivot_column = np.ma.MaskedArray(pivot_column, pivot_column <= 0)
            ratios = self.tableau[:self.m, self.n] / valid_pivot_column
            # Choose the leaving variable according to Bland's rule
            # If multiple variables have the minimum ratios, then choose the one with smallest index.
            min_ratio = ratios.min()
            # Get the label number of the variable to apply Bland's rule
            labels_basic_variables = [int(basic_variable[1:]) for basic_variable in self.basic_variables]
            mask_leaving_variables = np.ma.MaskedArray(labels_basic_variables, ratios != min_ratio)
            leaving_variable_index = mask_leaving_variables.argmin()  # k
            entering_variable = self.all_variables[pivot_index_column]
            leaving_variable = self.basic_variables[leaving_variable_index]
            self.basic_variables[leaving_variable_index] = entering_variable
            # Pivoting rule
            for index_row in range(0, self.m + 1):
                for index_column in range(0, self.n + 1):
                    if index_row != leaving_variable_index and index_column != pivot_index_column:
                        self.tableau[index_row, index_column] = (self.tableau[index_row, index_column] *
                                                                 self.tableau[
                                                                     leaving_variable_index, pivot_index_column] -
                                                                 self.tableau[index_row, pivot_index_column] *
                                                                 self.tableau[leaving_variable_index, index_column]) / \
                                                                self.tableau[leaving_variable_index, pivot_index_column]
            for index_row in range(0, self.m + 1):
                if index_row != leaving_variable_index:
                    self.tableau[index_row, pivot_index_column] = 0
            for index_column in range(0, self.n + 1):
                if index_column != pivot_index_column:
                    self.tableau[leaving_variable_index, index_column] /= self.tableau[
                        leaving_variable_index, pivot_index_column]
            self.tableau[leaving_variable_index, pivot_index_column] = 1
            print("The new tableau of Simplex is: \n{0}".format(np.round(self.tableau, 3)))
            print("The new basic variables are the following: {0} = {1}".format(self.basic_variables,
                                                                        np.round(self.tableau[:self.m, self.n], 3).tolist()))
        print("The solution is {0} with basic variables {1} = {2}".format(
            -self.tableau[self.m, self.n].astype(np.float32),
            self.basic_variables,
            np.round(self.tableau[:self.m, self.n], 3).tolist()))

        print("---------------------------------------------------------")

        # Multiple optimal solutions
        if additional_optimal_solution:
            # Get the non-basic variables using MaskedArray.
            non_basic_variables = np.ma.MaskedArray(self.all_variables, np.isin(self.all_variables, self.basic_variables))
            # Use MaskedArray to store the variables that have 0 reduced cost out of the non-basic variables.
            zero_reduced_cost_non_basic_variables = np.ma.MaskedArray(non_basic_variables, self.tableau[self.m, :-1] != 0)
            # Check if there exists such a variable, we use logical not since the getmaskarray returns True
            # on the masked values which are either basic variables, either have a reduced cost > 0.
            zero_reduced_cost_non_basic_variables = np.logical_not(
                np.ma.getmaskarray(zero_reduced_cost_non_basic_variables))
            if np.any(zero_reduced_cost_non_basic_variables):
                print("There is an additional optimal solution.")
                # Choose another entering variable in the basis.
                pivot_index_column = np.min(np.where(zero_reduced_cost_non_basic_variables))
                # Obtain the pivot column
                pivot_column = self.tableau[:self.m, pivot_index_column]
                # Compute ratios
                valid_pivot_column = np.ma.MaskedArray(pivot_column, pivot_column <= 0)
                ratios = self.tableau[:self.m, self.n] / valid_pivot_column
                # Choose the leaving variable according to Bland's rule
                # If multiple variables have the minimum ratios, then choose the one with smallest index.
                min_ratio = ratios.min()
                # Get the label number of the variable to apply Bland's rule
                labels_basic_variables = [int(basic_variable[1:]) for basic_variable in self.basic_variables]
                mask_leaving_variables = np.ma.MaskedArray(labels_basic_variables, ratios != min_ratio)
                leaving_variable_index = mask_leaving_variables.argmin()  # k
                entering_variable = self.all_variables[pivot_index_column]
                leaving_variable = self.basic_variables[leaving_variable_index]
                self.basic_variables[leaving_variable_index] = entering_variable
                print("The entering variable is {0} and the leaving variable is {1}.".format(entering_variable,
                                                                                             leaving_variable))
                # Pivoting rule
                for index_row in range(0, self.m + 1):
                    for index_column in range(0, self.n + 1):
                        if index_row != leaving_variable_index and index_column != pivot_index_column:
                            self.tableau[index_row, index_column] = (self.tableau[index_row, index_column] *
                                                                     self.tableau[
                                                                         leaving_variable_index, pivot_index_column] -
                                                                     self.tableau[index_row, pivot_index_column] *
                                                                     self.tableau[leaving_variable_index, index_column]) / \
                                                                    self.tableau[leaving_variable_index, pivot_index_column]
                for index_row in range(0, self.m + 1):
                    if index_row != leaving_variable_index:
                        self.tableau[index_row, pivot_index_column] = 0
                for index_column in range(0, self.n + 1):
                    if index_column != pivot_index_column:
                        self.tableau[leaving_variable_index, index_column] /= self.tableau[
                            leaving_variable_index, pivot_index_column]
                self.tableau[leaving_variable_index, pivot_index_column] = 1
                print("The tableau corresponding to the additional optimal solution is: \n{0}".format(
                    np.round(self.tableau, 3)))
                print("The additional optimal solution is the following: {0} = {1}".format(self.basic_variables,
                    np.round(self.tableau[:self.m, self.n], 3).tolist()))
            else:
                print("The above result is the unique optimal solution of the problem.")
        return self.tableau, self.basic_variables


if __name__ == '__main__':
    # The fist rows in input are constraints and the last line is the objective
    # Seminar
    # input = [[0.5, -5.5, -2.5, 9, '<=', 0], [0.5, -1.5, -0.5, 1, '<=', 0], [1, 0, 0, 0, '<=', 1],
    #          ['max', 10, -57, -9, -24, 2]]

    # Homework - Ex. 4
    # subpunctul a
    # input = [[-2, 1, '<=', 2], [-1, 2, '<=', 7], [1, 0, '<=', 3], ['min', -1, -1, 0]]
    # subpunctul b
    # input = [[-1, 1, '<=', 2], [-2, 1, '<=', 1], ['min', -1, -2, 0]]
    # subpunctul c
    # input = [[4, 5, -2, '<=', 22], [1, -2, 1, '<=', 30], ['min', 3, -2, -4, 0]]

    # simplex = Simplex(input)
    # simplex.solve_simplex()

    # Multiple optimal solutions in Simple (course 4)
    tableau = np.array([[1.0, 3.0, 0.0, 3.0, 0.0, 1.0], [0.0, 1.0, 0.0, -1.0, 1.0, 5.0],
                        [0.0, 2.0, 1.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, 0.0, 0.0, 12.0]])
    simplex = Simplex(tableau=tableau, all_variables=np.array(['x' + str(index) for index in range(1, 6)]),
                      basic_variables=['x1', 'x5', 'x3'])
    simplex.solve_simplex(additional_optimal_solution=True)
