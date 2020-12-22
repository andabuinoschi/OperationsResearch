import numpy as np
from TwoPhase import TwoPhase


class BranchAndBound:
    """
    Branch and Bound algorithm for solving ILP problems that need relaxation.
    """

    def __init__(self, input):
        self.stack = []
        self.P = input
        self.stack.append(self.P)
        self.x_best = None
        self.best_solution = None
        self.z_best = -np.inf
        self.number_relaxed_problems = 1

    def solve_problem(self):
        while self.stack:
            print("---------------------------------------------------------")
            print("------------------- Relaxed problem R{0} ------------------".format(
                self.number_relaxed_problems))
            self.number_relaxed_problems += 1
            problem = self.stack.pop()
            two_phase = TwoPhase(problem)
            print("The new relaxed problem is {0}".format(problem))
            if two_phase.solve_phase_1_problem() == "The problem is unbounded":
                print("The problem is unbounded.")
            two_phase_result = two_phase.solve_phase_2_problem()
            if two_phase_result != "The original problem is not feasible.":
                # The problem is feasible, so we retrieve the solution of the problem
                # using the two phase method.
                tableau, basic_variables, m, n = two_phase_result
                number_original_variables = len(input[0][:-2])
                variables_two_phase = ['x' + str(index) for index in range(1, n + 1)]
                index_basic_variables_in_all_array = []
                for basic_variable in basic_variables:
                    index_basic_variables_in_all_array.append(variables_two_phase.index(basic_variable))
                solution_zero = tableau[:m, n]
                solution_two_phase = np.zeros(shape=n)
                solution_two_phase[index_basic_variables_in_all_array] = solution_zero
                solution_zero = solution_two_phase[:number_original_variables]
                x_zero = np.array(['x' + str(index) for index in range(1, number_original_variables + 1)])
                objective_zero = tableau[m, n]
                print('Solution of TwoPhase is {0} = {1} with objective = {2}'.format(x_zero,
                                                                                      np.round(solution_zero,
                                                                                               3).tolist(),
                                                                                      np.round(objective_zero, 3)))
                # Check if the new objective found is greater than what we found so far.
                if objective_zero > self.z_best:
                    # If the objective is greater check if the solution is fathomed by integrality.
                    # If so, update the best objective and the solution under integer set.
                    if np.all(solution_zero == solution_zero.astype('int')):
                        print("The solution is fathomed by integrality. We update the best so-far solution {0} = {1},"
                              "having objective {2}.".format(x_zero, solution_zero, np.round(objective_zero, 3)))
                        self.x_best = x_zero
                        self.best_solution = solution_zero
                        self.z_best = objective_zero
                    else:
                        # If the solution is not fathomed by integrality, then choose the variable with
                        # the largest fractional part (if we have multiple such variables, then choose
                        # the first one) and add 2 new problems into the stack. We create 2 constraints
                        # x_j <= [x_j] and x_j >= [x_j] + 1 where x_j is the chosen variable and [x_j]
                        # represents the largest integer i such that i <= x_j. Each of these 2 new problems
                        # contains the original restrictions and either the first constraint created above,
                        # either the second one.
                        fractional_part = solution_zero - solution_zero.astype('int')
                        index_for_j_variable = np.where(fractional_part == np.max(fractional_part))
                        # Choose the index of the variable on which we create the new constraints.
                        if index_for_j_variable[0].size > 1:
                            index_for_j_variable = np.array([index_for_j_variable[0][0]])
                        j_variable = x_zero[index_for_j_variable]
                        # Create the new constraints new_constraint_leq and new_constraint_geq.
                        new_constraint = np.zeros(shape=number_original_variables)
                        new_constraint[index_for_j_variable] = 1
                        new_constraint_leq = new_constraint.tolist()
                        new_constraint_leq.extend(['<=', solution_zero[index_for_j_variable].astype('int').tolist()[0]])
                        new_constraint_geq = new_constraint.tolist()
                        new_constraint_geq.extend(
                            ['>=', solution_zero[index_for_j_variable].astype('int').tolist()[0] + 1])
                        # Add each of these 2 new constraints to the original restrictions of the parent
                        # problem, creating 2 new problems.
                        constraints = problem[:-1].copy()
                        new_relaxed_problem_1 = constraints.copy()
                        new_relaxed_problem_1.append(new_constraint_leq)
                        new_relaxed_problem_2 = constraints.copy()
                        new_relaxed_problem_2.append(new_constraint_geq)
                        new_relaxed_problem_1.append(problem[-1])
                        new_relaxed_problem_2.append(problem[-1])
                        # Add the 2 new problems into the stack in order to be solved.
                        self.stack.append(new_relaxed_problem_1)
                        self.stack.append(new_relaxed_problem_2)
            else:
                print("The problem is infeasible.")
        if self.x_best is None and self.z_best == -np.inf:
            print("The problem is infeasible.")
        print('---------------------------------------------------------')
        print('The best solution found is: {0} = {1} with objective {2}.'.format(
            self.x_best, self.best_solution, self.z_best
        ))


if __name__ == '__main__':
    # Exemplu din cursul 7
    # input = [[10, 7, '<=', 40], [1, 2, '<=', 5], ['max', 17, 12, 0]]

    # Exemplu din seminarul 7
    # input = [[10, 3, '<=', 52], [2, 3, '<=', 18], ['max', 5, 6, 0]]

    # Tema 3.1 exercitiul 3
    # subpunctul a)
    # input = [[1, -2, 2, '<=', 6], [1, 1, 2, '<=', 8], ['max', 1, -1, 2, 0]]

    # subpunctul b)
    # input = [[-1, 1, '<=', 0], [6, 2, '<=', 21], ['max', 2, 1, 0]]

    # subpunctul c)
    input = [[1, 2, 1, -1, '<=', 8], [1, 2, 0, 2, '<=', 7], [4, 1, 3, 0, '<=', 8], ['max', 1, 1, 1, 2, 0]]

    bb = BranchAndBound(input)
    bb.solve_problem()
