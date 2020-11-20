import numpy as np
from InputParser import InputParser
from Simplex import Simplex


class TwoPhase:
    """
    This is the Two-Phase implementation algorithm to solve LP problems.
    The constructor calls InputParser to get the problem into standard form
    and then solve the problem according to the Two-Phase method.
    The First Phase problem consists in adding artificial variables
    wherever it is needed and solve with Simplex method the new problem.
    The Second Phase represents in changing the objective according to the original
    problem and removing the artificial variables columns. Now, solve it again using
    Simplex method.

    Example: For the problem max 2x_1 - x_2 + 2x_3
                             s.t. x_1 + x_2 +  x_3 <=  6
                                 -x_1 + x_2        <= -1
                                 -x_1       +  x_3 <= -1
                                 x_1, x_2, x_3 >=0
            we get the following standard form from InputParser:
                             min z = -2x_1 + x_2 -2x_3
                             s.t.  x_1 + x_2 + x_3 + x_4             = 6
                                   x_1 - x_2             - x_5       = 1
                                   x_1       - x_3             - x_6 = 1
                                   x_1, x_2, x_3, x_4, x_5, x_6 >= 0
                            The new tableau from the InputParser is:
                            [[ 1.  1.  1.  1.  0.  0.  6.]
                             [ 1. -1.  0.  0. -1.  0.  1.]
                             [ 1.  0. -1.  0.  0. -1.  1.]
                             [-2.  1. -2.  0.  0.  0.  0.]]
                             The slack variables in the InputParser are: ['x4' 'x5' 'x6']
                             The basic variables in the InputParser are: ['x4', -1.0, -1.0]
                             where -1 represents that there is no variable that we can use for
                             the basic variables (here we would need artificial variables).

            Phase 1: We need to add artificial variables for the 2nd and the 3rd constraint where
            we lack a basic feasible solution. So, we should define artificial variables y_1 and y_2.
            The new problem becomes:
                             min  z' = y_1 + y_2
                             s.t.  x_1 + x_2 + x_3 + x_4                         = 6
                                   x_1 - x_2             - x_5       + y_1       = 1
                                   x_1       - x_3             - x_6       + y_2 = 1
                                   x_1, x_2, x_3, x_4, x_5, x_6, y_1, y_2 >= 0
            In order not to change the original problem, the values for y_1 and y_2 should become 0
            and the objective 0 after the first phase of Simplex algorithm. The new basic initial
            solution is [x_4, y_1, y_2] = [6, 1, 1].
            We have to rewrite the new objective (z') in terms of non-basic variables, so:
                             y_1 = 1 - x_1 + x_2 + x_5
                             y_2 = 1 - x_1 + x_3 + x_6
                             z' = -2x_1 + x_2 + x_3 + x_5 + x_6 + 2
            So the problem that we need to pass to Simplex for the first phase is:
                             min z' = -2x_1 + x_2 + x_3 + x_5 + x_6 + 2
                             s.t.  x_1 + x_2 + x_3 + x_4                         = 6
                                   x_1 - x_2             - x_5       + y_1       = 1
                                   x_1       - x_3             - x_6       + y_2 = 1
                                   x_1, x_2, x_3, x_4, x_5, x_6, y_1, y_2 >= 0
            The new variables are grouped as the following:
            - the variables: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1', 'y2']
            - the slack variables: ['x4', 'x5', 'x6']
            - the artificial variables: ['y1', 'y2']
            Solving with Simplex the first phase problem, we should have the objective z' = 0
            and the artificial variables as non-basic (also having values of 0) if the original
            problem is feasible. For our example, the solution to the Simplex algorithm is:
            - the tableau:   [[0, 0,  3, 1, -1,  2, 1, -2, 5]
                              [1, 0, -1, 0,  0, -1, 0,  1, 1]
                              [0, 1, -1, 0,  1, -1, 0,  1, 0]
                              [0, 0,  0, 0,  0,  0, 1,  1, 0]]
            - the basic variables: ['x4', 'x1', 'x2']

            Phase 2: For the phase 2 problem, we need to remove the columns corresponding to the
            artificial variables and put the original objective function into the tableau.
            So we have:
                             [[ 0, 0,  3, 1, -1,  2, 5]
                              [ 1, 0, -1, 0,  0, -1, 1]
                              [ 0, 1, -1, 0,  1, -1, 0]
                              [-2, 1, -2, 0,  0,  0, 0]]
            Since the reduced cost of the basic variables are not 0, the tableau is not in proper
            manner. We have to rewrite it.
                             x_1 = x_3 + x_6 + 1
                             x_2 = x_3 - x_5 + x_6
                             z = -2x_1 + x_2 -2_x3 = -2x_3 - 2x_6 -2 + x_3 - x_5 + x_6 - 2x_3
                               = -3x_3 - x_5 - x_6 - 2
            The new tableau is:
                             [[ 0, 0,  3, 1, -1,  2, 5]
                              [ 1, 0, -1, 0,  0, -1, 1]
                              [ 0, 1, -1, 0,  1, -1, 0]
                              [ 0, 0, -3, 0, -1, -1, 2]]
            Solving the Phase 2 Simplex problem, we get the last tableau:
                             [[ 0, 0.5, 1, 0.5, 0,  0.5, 2.5]
                              [ 1, 0.5, 0, 0.5, 0, -0.5, 3.5]
                              [ 0, 1.5, 0, 0.5, 1, -0.5, 2.5]
                              [ 0,   3, 0,   2, 0,    0, 12]]
            with basic variables ['x3', 'x1', 'x5']. So the solution of the original problem is
            [x_1, x_2, x_3, x_4, x_5, x_6] = [3.5, 0, 2.5, 0, 2,5, 0] with the objective z = -12.
            Do not forget to change the sign.
    """

    def __init__(self, input):
        parser = InputParser(input)
        self.tableau = parser.get_tableau()
        self.RHS = np.expand_dims(self.tableau[:, -1], axis=1)
        self.m = self.tableau.shape[0] - 1  # the number of constraints
        self.n = self.tableau.shape[1] - 1  # the number of variables
        self.original_objective = np.copy(
            self.tableau[self.m])  # the original objective function to restate it in phase 2
        self.all_variables = parser.get_all_variables()
        self.slack_variables = parser.get_slack_variables()
        self.basic_variables = parser.get_basic_variables()  # this is a list
        self.artificial_variables = []
        self.artificial_variable_label = 1

    def solve_two_phase_problem(self):
        self.solve_phase_1_problem()
        self.solve_phase_2_problem()

    # The first phase solver.
    # The method below adds artificial variables where are needed, rewrite the objective according
    # to the non-basic variables and call the Simplex solver for the phase 1 tableau problem.
    def solve_phase_1_problem(self):
        print("---------------------------------------------------------")
        print("------------------------ PHASE 1 ------------------------")
        # Remove RHS column for the moment
        self.tableau = self.tableau[:, :-1]
        # Check if we need artificial variables. If so, we change the objective.
        if -1 in self.basic_variables:
            self.tableau[self.m] = np.zeros(shape=(1, self.n))
        # Check where do we need artificial variables and create them
        artificial_variable_index_in_basic = []
        for index in range(0, len(self.basic_variables)):
            variable = self.basic_variables[index]
            if variable == -1:  # this means we need an artificial variable here
                artificial_variable_index_in_basic.append(index)
                # We need to create a new column for our tableau
                # (the coefficients of A matrix and the value for the objective function).
                artificial_variable_column = np.zeros(shape=(len(self.basic_variables) + 1, 1))
                artificial_variable_column[index, 0] = 1
                # Add the artificial variable into the objective now
                artificial_variable_column[-1, 0] = 1
                new_artificial_variable_label = 'y' + str(self.artificial_variable_label)
                self.artificial_variable_label += 1
                self.basic_variables[index] = new_artificial_variable_label
                # Add the column to the tableau
                self.tableau = np.append(self.tableau, artificial_variable_column, axis=1)
                self.artificial_variables.append(new_artificial_variable_label)
        # Restore the RHS column in the tableau
        self.tableau = np.append(self.tableau, self.RHS, axis=1)
        # Add artificial variable in all variables
        self.artificial_variables = np.array(self.artificial_variables)
        self.all_variables = np.append(self.all_variables, self.artificial_variables)
        # Change number of variables now
        self.n = len(self.all_variables)
        # Make sure the RHS of the objective is 0
        self.tableau[self.m, self.n] = 0
        # We have to rewrite the objective function according to the non-basic variables
        number_non_basic_variables = self.n - self.artificial_variables.shape[0]
        new_objective = np.expand_dims(
            np.sum(-self.tableau[artificial_variable_index_in_basic, :number_non_basic_variables], axis=0), axis=0)
        objective_artificial_variables = np.zeros(shape=(1, self.artificial_variables.shape[0]))
        new_objective = np.append(new_objective, objective_artificial_variables)
        self.tableau[self.m, :-1] = new_objective
        # Also change the RHS of the objective
        self.tableau[self.m, self.n] = np.sum(-self.tableau[artificial_variable_index_in_basic, -1], axis=0)
        # Call Simplex Solver with our new tableau
        simplex = Simplex(tableau=self.tableau, all_variables=self.all_variables, basic_variables=self.basic_variables)
        self.tableau, self.basic_variables = simplex.solve_simplex()

    # The second phase solver.
    # The method below preprocesses the tableau from the first phase solution to restate
    # the original problem and after this step, calls the Simplex solver.
    def solve_phase_2_problem(self):
        print("---------------------------------------------------------")
        print("------------------------ PHASE 2 ------------------------")
        # If the objective of the Simplex solution is not 0, then this means that the original problem is not
        # feasible.
        if self.tableau[self.m, self.n] != 0:
            print("The original problem is not feasible.")
        else:
            # Check if we still have artificial variables in the basis.
            if np.any(np.isin(self.basic_variables, self.artificial_variables)):
                # We have artificial variables in the basis. Remove it according to the following rule:
                # If there exists an artificial variable in the basis y_k on row i then look for another variable
                # x_j to enter the basis for which coefficient tableau[i][j] != 0.
                # If no such variable exists and it holds for artificial variables, then the constraint corresponding
                # to row i is redundant (it is liniarly dependent with other constraint) and can be removed
                # from tableau.
                # We have to replace this artificial variable
                # First get the indexes in all_variables where we do not have artificial variables.
                index_in_all_original_variables = np.where(np.logical_not(np.isin(self.all_variables, self.artificial_variables)))
                # Get the index where we have artificial variables in the basis.
                index_in_basic_artificial_variables = np.where(np.isin(self.basic_variables, self.artificial_variables))
                if np.any(self.tableau[index_in_basic_artificial_variables[0], index_in_all_original_variables[0]] != 0):
                    # Then we can replace the artificial variable with other non-basic original variable
                    indexes_valid_pivot = np.where(self.tableau[index_in_basic_artificial_variables[0], index_in_all_original_variables[0]] != 0)
                    pivot_index_column = np.min(indexes_valid_pivot)
                    entering_variable = self.all_variables[pivot_index_column]
                    leaving_variable_index = np.min(index_in_basic_artificial_variables)
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
                                                                         self.tableau[
                                                                             leaving_variable_index, index_column]) / \
                                                                        self.tableau[
                                                                            leaving_variable_index, pivot_index_column]
                    for index_row in range(0, self.m + 1):
                        if index_row != leaving_variable_index:
                            self.tableau[index_row, pivot_index_column] = 0
                    for index_column in range(0, self.n + 1):
                        if index_column != pivot_index_column:
                            self.tableau[leaving_variable_index, index_column] /= self.tableau[
                                leaving_variable_index, pivot_index_column]
                    self.tableau[leaving_variable_index, pivot_index_column] = 1
                else:
                    # We can remove row. It is redundant.
                    self.tableau = np.delete(self.tableau, index_in_basic_artificial_variables[0], axis=0)
                    self.m -= len(index_in_basic_artificial_variables[0])
                    self.basic_variables = np.delete(self.basic_variables, index_in_basic_artificial_variables)
            # Remove columns containing artificial variables
            index_columns_artificial_variables = np.where(np.isin(self.all_variables, self.artificial_variables))
            self.all_variables = np.delete(self.all_variables, index_columns_artificial_variables, axis=0)
            self.tableau = np.delete(self.tableau, index_columns_artificial_variables, axis=1)
            self.n -= len(self.artificial_variables)
            # Restate the original objective function
            self.tableau[self.m] = self.original_objective
            # Check if there are reduced cost != 0 for the basic variables
            index_basic_variables_in_all = np.where(np.isin(self.all_variables, self.basic_variables))
            reduced_costs_basic_variables = self.tableau[self.m, index_basic_variables_in_all][0]
            # We have to rewrite the basic variables in terms of non-basic variables
            if np.any(reduced_costs_basic_variables != 0):
                # Check the basic variables which we need to rewrite
                # Also pay attention by their order in the basis.
                basic_variables_to_rewrite_in_all = self.all_variables[np.where(reduced_costs_basic_variables != 0)]
                # Store indexes which we need in rewriting the basic variables
                index_in_all_variables_to_rewrite = []
                for basic_variable in self.basic_variables:
                    if basic_variable in basic_variables_to_rewrite_in_all:
                        index_in_all_variables_to_rewrite.append(np.min(np.where(self.all_variables == basic_variable)))
                # Prepare a new matrix of shape (number_basic_variables_to_rewrite, tableau_columns) with elements 0
                rewritten_cost_basic_variables = np.zeros(
                    shape=(len(basic_variables_to_rewrite_in_all), self.tableau.shape[1]))
                index_in_all_variables_in_terms_of = np.where(
                    np.logical_not(np.isin(self.all_variables, basic_variables_to_rewrite_in_all)))
                index_in_basic_variables_to_rewrite = np.where(
                    np.isin(self.basic_variables, basic_variables_to_rewrite_in_all))
                cost_basic_variables_to_rewrite = np.expand_dims(
                    self.tableau[self.m, index_in_all_variables_to_rewrite], axis=0)
                # Compute the new reduced cost array in terms of all the other variables which do not need change
                rewritten_cost_basic_variables[:, index_in_all_variables_in_terms_of[0]] = \
                    -self.tableau[index_in_basic_variables_to_rewrite[0], :][:,
                     index_in_all_variables_in_terms_of[0]]
                rewritten_cost_basic_variables[:, self.n] = self.tableau[
                    index_in_basic_variables_to_rewrite[0], self.n]
                rewritten_cost = np.dot(cost_basic_variables_to_rewrite, rewritten_cost_basic_variables)
                # Do not forget to add the previous cost of the non-basic variables into the new reduced cost
                previous_cost_reduced_non_basic_variables = self.tableau[self.m, :][
                    index_in_all_variables_in_terms_of[0]]
                rewritten_cost[:,
                index_in_all_variables_in_terms_of[0]] += previous_cost_reduced_non_basic_variables
                # Change RHS of the objective sign
                rewritten_cost[0, self.n] = -rewritten_cost[0, self.n]
                # Restate the new objective in the tableau
                self.tableau[self.m] = rewritten_cost
                simplex_solver = Simplex(tableau=self.tableau, basic_variables=self.basic_variables,
                                         all_variables=self.all_variables)
                simplex_solver.solve_simplex()


if __name__ == '__main__':
    # Course 4 - example left as exercise -> slide 20
    # input = [[3, 2, '=', 14], [2, -4, '>=', 2], [4, 3, '<=', 19], ['min', 2, 3, 0]]

    # Course 4 - example -> slide 30
    # input = [[1, 1, 2, '=', 2], [2, 1, 1, '=', 4], ['min', 1, 1, 0, 0]]

    # Course 4 - example -> slide 33
    # input = [[1, 1, '=', 2], [2, 2, '=', 4], ['min', 1, 2, 0]]

    # Seminar 4 - example
    # input = [[1, 1, 1, '<=', 6], [-1, 1, 0, '<=', -1], [-1, 0, 1, '<=', -1], ['max', 2, -1, 2, 0]]

    # HOMEWORK 2.1 - ex. 2
    # Subpunctul a)
    # input = [[1, 1, '>=', 6], [2, 3, '<=', 4], ['min', -1, 0, 0]]

    # Subpunctul b)
    # input = [[2, 1, 1, '=', 4], [1, 1, 2, '=', 2], ['min', 1, 1, 0, 0]]

    # Subpunctul c)
    # input = [[1, 2, -1, 1, '=', 0], [2, -2, 3, 3, '=', 9], [1, -1, 2, -1, '=', 6], ['min', -3, 1, 3, -1, 0]]

    # Subpunctul d)
    input = [[1, 1, '=', 2], [2, 2, '=', 4], ['min', 1, 2, 0]]

    two_phase = TwoPhase(input)
    two_phase.solve_two_phase_problem()
