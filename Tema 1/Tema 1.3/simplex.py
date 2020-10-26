import numpy as np


class Simplex:
    def __init__(self, input: np.array):
        self.input = input
        self.m = self.input.shape[0] - 1  # last line is the objective in input
        self.n = self.input.shape[1] - 1  # last column is the rhs column in input
        self.rhs_column = self.input[:, self.n]
        self.constraints = self.input[0: self.m]
        self.constraints = np.delete(self.constraints, [self.n], axis=1)  # remove rhs column
        self.objective = self.input[self.m, 0:self.n]  # The coefficients for the objective is for the variables

        # Adding the slack variables
        self.all_variables = [index for index in range(1, self.n + self.m + 1)]
        self.basic_variables = [index for index in range(self.n + 1, self.n + self.m + 1)]
        self.n = self.n + self.m  # Update number of variables adding the slack variables
        slack_variables = np.identity(n=self.m, dtype=np.float32)
        self.constraints = np.append(self.constraints, slack_variables, axis=1)
        slack_variables_cost = np.zeros(shape=self.m)
        self.objective = np.append(self.objective, slack_variables_cost)

        # Preparing the tableau for solving
        # Expanding dimensions from 1D to 2D
        self.objective = np.expand_dims(self.objective, axis=0)
        self.rhs_column = np.expand_dims(self.rhs_column, axis=1)
        # Append the reduced cost row
        self.tableau = np.append(self.constraints, self.objective, axis=0)
        # Append the RHS column
        self.tableau = np.append(self.tableau, self.rhs_column, axis=1)

    def solve_simplex(self):
        print("First tableau of Simplex is: \n{0}".format(self.tableau))
        while np.any(self.tableau[self.m] < 0):
            # Get the entering variable in basic variables
            pivot_index_column = np.min(np.where(self.tableau[self.m] < 0))  # l
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
            mask_leaving_variables = np.ma.MaskedArray(self.basic_variables, ratios != min_ratio)
            leaving_variable_index = mask_leaving_variables.argmin()  # k
            entering_variable = self.all_variables[pivot_index_column]
            leaving_variable = self.basic_variables[leaving_variable_index]
            self.basic_variables[leaving_variable_index] = entering_variable
            # Pivoting rule
            for index_row in range(0, self.m + 1):
                for index_column in range (0, self.n + 1):
                    if index_row != leaving_variable_index and index_column != pivot_index_column:
                        self.tableau[index_row, index_column] = (self.tableau[index_row, index_column] *
                                                                self.tableau[leaving_variable_index, pivot_index_column] -
                                                                self.tableau[index_row, pivot_index_column] *
                                                                self.tableau[leaving_variable_index, index_column]) / \
                                                                self.tableau[leaving_variable_index, pivot_index_column]
            for index_row in range(0, self.m + 1):
                if index_row != leaving_variable_index:
                    self.tableau[index_row, pivot_index_column] = 0
            for index_column in range(0, self.n + 1):
                if index_column != pivot_index_column:
                    self.tableau[leaving_variable_index, index_column] /= self.tableau[leaving_variable_index, pivot_index_column]
            self.tableau[leaving_variable_index, pivot_index_column] = 1
            print("The new tableau of Simplex is: \n{0}".format(self.tableau))
            print("The new basic variables are the following: {0}".format(self.basic_variables))
        return "The solution is {0} with basic variables {1} = {2}".format(-self.tableau[self.m, self.n],
                                                                           self.basic_variables,
                                                                           self.tableau[:self.m, self.n])


if __name__ == '__main__':
    # The fist rows in input are constraints and the last line is the objective

    # Ex. 4
    # subpunctul a
    # input = [[-2, 1, 2], [-1, 2, 7], [1, 0, 3], [-1, -1, 0]]
    # subpunctul b
    # input = [[-1, 1, 2], [-2, 1, 1], [-1, -2, 0]]
    # subpunctul c
    input = [[4, 5, -2, 22], [1, -2, 1, 30], [3, -2, -4, 0]]
    input = np.array(input, dtype=np.float32)
    simplex = Simplex(input)
    solution = simplex.solve_simplex()
    print(solution)
