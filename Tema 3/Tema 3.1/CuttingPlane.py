import numpy as np
from TwoPhase import TwoPhase
from InputParser import InputParser


class CuttingPlane:
    def __init__(self, input):
        """
            Cutting Plane method for solving ILP problems
        """
        self.problem = input
        self.maximum_iterations = 5
        self.number_variables_original_problem = len(input[0][:-2])
        self.original_objective = input[-1]
        self.original_solution = np.zeros(shape=self.number_variables_original_problem)
        self.problem = input

    def solve_problem(self):
        for iteration in range(0, self.maximum_iterations):
            print("---------------------------------------------------")
            print("------------------ Iteration {0} ------------------".format(iteration))
            print("The new relaxed problem is {0}".format(self.problem))
            input_parser = InputParser(self.problem)
            original_tableau_problem = input_parser.tableau
            print("The new problem in standard form is: \n{0}".format(original_tableau_problem))
            two_phase = TwoPhase(self.problem)
            if two_phase.solve_phase_1_problem() == "The problem is unbounded":
                print("The problem is unbounded.")
            two_phase_result = two_phase.solve_phase_2_problem()
            # Check feasibility of the problem. The infeasibility would mean that
            # the objective of phase 1 solution is != 0.
            if two_phase_result != "The original problem is not feasible.":
                tableau, basic_variables, m, n = two_phase_result
                solution_zero = np.round(tableau[:m, n], 3)
                objective = np.round(tableau[m, n], 3)
                print("The tableau of the Two Phase solution problem is:\n{0}".format(np.round(tableau, 3)))
                print("The basic variables are {0} = {1} and the objective is {2}".format(basic_variables,
                                                                                          solution_zero, objective))
                if np.all(solution_zero == solution_zero.astype('int')):
                    # Process the solution of the original problem since it contains only
                    # the basic variables solution with the potentially added slacks.
                    for index in range(0, len(basic_variables)):
                        basic_variable = basic_variables[index]
                        index_variable = int(basic_variable[1:])
                        if index_variable <= self.number_variables_original_problem:
                            self.original_solution[index_variable - 1] = solution_zero[index]
                    print("The solution is {0} = {1} with objective {2}".format(
                        ['x' + str(index + 1) for index in range (0, self.number_variables_original_problem)],
                        self.original_solution, objective))
                    return self.original_solution, objective

                # We compute the fractional parts of the solution. We choose the variable
                # with the highest fractional part.
                fractional_part = solution_zero - solution_zero.astype('int')
                maximum_fractional_part = np.max(fractional_part)
                index_for_i_variable = np.where(fractional_part == maximum_fractional_part)
                # If there are more such variables sharing the maximum fractional part
                # then, we choose arbitrarily the first variable from this set.
                if index_for_i_variable[0].size > 1:
                    index_for_i_variable = np.array([index_for_i_variable[0][0]])
                equation = tableau[index_for_i_variable]
                gomory_fractional_cut = np.floor(equation)

                # Rewrite the new constraint in terms of original variables.
                slack_variables_rewrite = - original_tableau_problem[:m, :self.number_variables_original_problem]
                RHS_of_slacks_variables_rewritte = original_tableau_problem[:m, n]
                coefficients_of_slacks = np.array(gomory_fractional_cut[0, self.number_variables_original_problem: n])
                slack_rewritten_in_terms_of_original_variables = np.dot(coefficients_of_slacks, slack_variables_rewrite)
                constraint_in_original_variables = gomory_fractional_cut[0, :self.number_variables_original_problem] + slack_rewritten_in_terms_of_original_variables

                # Compute the RHS of the new constraint.
                new_RHS_value = np.dot(-coefficients_of_slacks, RHS_of_slacks_variables_rewritte) + gomory_fractional_cut[0, -1]

                # Add the new constraint to the problem.
                new_constraint = constraint_in_original_variables.tolist() + ['<=', new_RHS_value]
                problem = self.problem[:-1]
                problem.append(new_constraint)
                problem.append(self.original_objective)
                self.problem = problem


if __name__ == '__main__':
    # Exemplu din cursul 7
    # input = [[10, 7, '<=', 40], [1, 2, '<=', 5], ['max', 17, 12, 0]]

    # Exemplu din seminarul 7
    # input = [[-4, 6, "<=", 9], [1, 1, "<=", 4], ['min', 1, -2, 0]]

    # Tema 3.1 exercitiul 3
    # subpunctul a)
    # input = [[1, -2, 2, '<=', 6], [1, 1, 2, '<=', 8], ['max', 1, -1, 2, 0]]

    # subpunctul b)
    # input = [[-1, 1, '<=', 0], [6, 2, '<=', 21], ['max', 2, 1, 0]]

    # subpunctul c)
    input = [[1, 2, 1, -1, '<=', 8], [1, 2, 0, 2, '<=', 7], [4, 1, 3, 0, '<=', 8], ['max', 1, 1, 1, 2, 0]]

    cp = CuttingPlane(input)
    cp.solve_problem()
