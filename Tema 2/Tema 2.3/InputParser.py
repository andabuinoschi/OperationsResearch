import numpy as np


class InputParser:
    def __init__(self, input: list):
        """
        Constructor of the standard form input by adding slack or excess variables where needed
        :param - input: input of the problem
               - input can be defined in the following manner:
               [[a11, a12, '<=', b1], [a21, a22, '=', b2], ['min'/'max', a31, a32, '>=', b3]]

               Example: min z = -x1
                       s.t. x1 +  x2 >= 6
                           2x1 + 3x2 <= 4
                           x1, x2 >= 0

                    This problem will be passed in the constructor in the following manner:
                    [[1, 1, '>=', 6], [2, 3, '<=', 4], ['min', -1, 0, 0]]
                    Note that the last line represents the objective of the problem.

                    The standard problem would be:
                       min z = -x1
                       s.t. x1 +  x2 - x3      = 6 -> excess variable
                           2x1 + 3x2      + x4 = 4 -> slack variable
                           x1, x2, x3, x4 >= 0

                    In comparison to the standard form for Simplex or Two Phase, if we
                    add an excess variable, then we multiply the entire constraint equality by -1
                    and add the new resulted slack variable as basic in order to prepare the
                    problem for Dual Simplex method.
                        min z = -x1
                        s.t. -x1 -  x2 + x3      = -6
                             2x1 + 3x2      + x4 =  4

                    The parser should return the tableau of the following:
                    [[-1, -1, 1, 0, -6], [2, 3, 0, 1, 4], [-1, 0, 0, 0, 0]] which is
                    [[-1.0  -1.0  1.0  0.0  -6.0]
                     [ 2.0   3.0  0.0  1.0   4.0]
                     [-1.0   0.0  0.0  0.0   0.0]]
        """
        number_all_variables = len(input[-1][1:-1])
        all_variables = ['x' + str(index + 1) for index in range(0, number_all_variables)]
        basic_variables = np.full(shape=(len(input[:-1])), fill_value=-1.0).tolist()
        slack_variables = []
        slack_variable_label = number_all_variables + 1

        result = []
        RHS = []

        # Check if it is a maximization problem, then transform it into a minimization problem
        problem_type = input[-1][0]
        if problem_type == 'max':
            cost_function = np.array(input[-1][1:])
            cost_function = np.multiply(-1, cost_function).tolist()
            input[-1][0] = 'min'
            input[-1][1:] = cost_function

        # Create reduced cost row
        objective_function = np.array(input[-1][1:-1])

        # Check if we have sign '>=' in the constraints.
        # If we have it, then we multiply the constraint by -1
        # and transform the excess variable into a slack one. (Dual Simplex)
        for index in range(0, len(input[:-1])):
            equation = input[index]
            equation_sign = equation[-2]
            if equation_sign == '>=':
                coefficients_value = -1 * np.array(equation[:-2])
                RHS_value = -1 * equation[-1]
                new_equation_sign = '<='
                # Replace it in input
                input[index][:-2] = coefficients_value.tolist()
                input[index][-2] = new_equation_sign
                input[index][-1] = RHS_value

        # Create RHS column
        for equation in input:
            RHS.append(equation[-1])

        # Change sign for the objective function
        RHS[-1] = -RHS[-1]
        RHS = np.array(RHS)

        # Add equations coefficients (matrix A) in tableau
        for equation in input[:-1]:
            result += [equation[:number_all_variables]]
        result = np.array(result, dtype=np.float32)

        # Add slack and excess variables where needed
        number_constraints = len(input[:-1])
        for index_equation in range(0, number_constraints):
            equation_sign = input[index_equation][-2]
            slack_tableau_column = np.full(shape=(number_constraints, 1), fill_value=0)
            new_slack_variable_label = 'x' + str(slack_variable_label)
            slack_variable_label += 1
            if equation_sign == "<=":  # we need slack variable
                slack_variables.append(new_slack_variable_label)
                slack_tableau_column[index_equation, 0] = 1
                # We can add it as a basic variable since it's coefficient is 1 and b is negative
                # due to the preprocessing step above.
                basic_variables[index_equation] = new_slack_variable_label
                # Add value of the objective function
                objective_function = np.append(objective_function, 0)
                result = np.append(result, slack_tableau_column, axis=1)

        objective_function = np.expand_dims(objective_function, axis=0)
        RHS = np.expand_dims(RHS, axis=1)
        result = np.append(result, objective_function, axis=0)
        result = np.append(result, RHS, axis=1)

        slack_variables = np.array(slack_variables)
        all_variables = np.append(all_variables, slack_variables)

        self.tableau = result
        self.all_variables = all_variables
        self.basic_variables = basic_variables
        self.slack_variables = slack_variables

    def get_all_variables(self):
        return self.all_variables

    def get_basic_variables(self):
        return self.basic_variables

    def get_slack_variables(self):
        return self.slack_variables

    def get_tableau(self):
        return self.tableau


if __name__ == '__main__':
    input = [[1, 1, 1, '<=', 6], [-1, 1, 0, '<=', -1], [-1, 0, 1, '<=', -1], ['max', 2, -1, 2, 0]]
    parser = InputParser(input)
    print(parser.get_slack_variables())
    print(parser.get_basic_variables())
    print(parser.get_tableau())
