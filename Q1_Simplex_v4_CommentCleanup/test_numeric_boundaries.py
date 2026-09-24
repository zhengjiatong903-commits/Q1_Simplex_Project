"""Targeted float-boundary regressions. No third-party LP solver is used here."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
from pathlib import Path
import unittest
import numpy as np
from simplex0 import LPProblem, SimplexBase, ClassicalSimplex, DualSimplex, GeneralizedSimplex
from run_example import print_result

CALLS = 0


def solve(method, problem, **options):
    global CALLS
    CALLS += 1
    solver = method(problem, **options)
    result = solver.solve(check_alternatives=False)
    return solver, result


class NumericBoundaryTests(unittest.TestCase):
    def test_fractional_bounds_are_not_rounded(self):
        for method in (ClassicalSimplex, GeneralizedSimplex):
            for b in (2.375, 2.9999999997, 0.1 + 0.2):
                with self.subTest(method=method.__name__, bound=repr(b)):
                    _, r = solve(method, LPProblem('max', [[1]], [b], [1], ['<=']))
                    self.assertEqual(r['status'], 'OPTIMAL')
                    self.assertEqual(r['x'][0], b)

    def test_fractional_lower_bounds_all_algorithms(self):
        for method in (ClassicalSimplex, DualSimplex, GeneralizedSimplex):
            for b in (2.375, 0.125):
                with self.subTest(method=method.__name__, bound=b):
                    _, r = solve(method, LPProblem('min', [[1]], [b], [1], ['>=']))
                    self.assertEqual(r['status'], 'OPTIMAL')
                    self.assertEqual(r['x'][0], b)

    def test_one_third_is_not_replaced_by_two_decimals(self):
        for method in (ClassicalSimplex, GeneralizedSimplex):
            _, r = solve(method, LPProblem('max', [[3]], [1], [1], ['<=']))
            self.assertEqual(r['status'], 'OPTIMAL')
            self.assertLess(abs(r['x'][0] - float(Fraction(1, 3))), 1e-15)
            self.assertNotEqual(r['x'][0], 0.33)

    def test_close_ratios_are_not_feasibility_ties(self):
        for method in (ClassicalSimplex, GeneralizedSimplex):
            for rhs in ([1 + 5e-10, 1], [1, 1 + 5e-10]):
                _, r = solve(method, LPProblem('max', [[1], [1]], rhs, [1], ['<='] * 2))
                self.assertEqual(r['status'], 'OPTIMAL')
                self.assertEqual(r['x'][0], 1.0)

    def test_conflicting_close_bounds_are_not_called_optimal(self):
        # Fraction states the intended decimal conflict exactly for the test.
        self.assertGreater(Fraction('1.0000000005'), Fraction('1'))
        p = LPProblem('min', [[1], [1]], [1, 1 + 5e-10], [1], ['<=', '>='])
        for method in (ClassicalSimplex, DualSimplex, GeneralizedSimplex):
            _, default = solve(method, p)
            self.assertEqual(default['status'], 'NUMERICAL_ERROR')
            _, tight = solve(method, p, tolerance=1e-12, solution_tolerance=1e-12)
            self.assertEqual(tight['status'], 'INFEASIBLE')

    def test_negative_upper_bound_is_not_zeroed(self):
        p = LPProblem('min', [[1]], [-5e-10], [1], ['<='])
        for method in (ClassicalSimplex, DualSimplex, GeneralizedSimplex):
            _, r = solve(method, p)
            self.assertEqual(r['status'], 'NUMERICAL_ERROR')
            _, r = solve(method, p, tolerance=1e-12, solution_tolerance=1e-12)
            self.assertEqual(r['status'], 'INFEASIBLE')

    def test_small_valid_bound_with_suitable_tolerance(self):
        for method in (ClassicalSimplex, DualSimplex, GeneralizedSimplex):
            _, r = solve(method, LPProblem('min', [[1]], [1e-12], [1], ['>=']),
                         tolerance=1e-13, solution_tolerance=1e-13)
            self.assertEqual(r['status'], 'OPTIMAL')
            self.assertEqual(r['x'][0], 1e-12)

    def test_whole_row_scaling_does_not_change_bound(self):
        for method in (ClassicalSimplex, GeneralizedSimplex):
            for scale in (1e-200, 1e-10, 1.0, 1e10, 1e200):
                _, r = solve(method, LPProblem('max', [[scale]], [scale], [1], ['<=']))
                self.assertEqual(r['status'], 'OPTIMAL')
                self.assertEqual(r['x'][0], 1.0)

    def test_objective_constant_is_not_pivoted(self):
        for method in (ClassicalSimplex, GeneralizedSimplex):
            _, r = solve(method, LPProblem('max', [[1]], [1], [1e-300], ['<='],
                                         objective_constant=1e100))
            self.assertEqual(r['status'], 'OPTIMAL')
            self.assertEqual(r['x'][0], 1.0)
            self.assertEqual(r['objective_variable_part'], 1e-300)
            self.assertEqual(r['objective_constant'], 1e100)

    def test_small_negative_reduced_cost_is_not_silently_optimal(self):
        p = LPProblem('max', [[1, 0], [0, 1]], [1, 1], [1, 1e-10], ['<='] * 2)
        for method in (DualSimplex, GeneralizedSimplex):
            r = method(p).solve(initial_basis=[0, 3], check_alternatives=False)
            self.assertEqual(r['status'], 'NUMERICAL_ERROR')

    def test_display_keeps_returned_values_unchanged(self):
        p = LPProblem('min', [[1]], [1e-12], [1], ['>='])
        solver, r = solve(ClassicalSimplex, p)
        before = deepcopy(r)
        output = io.StringIO()
        with redirect_stdout(output):
            print_result(solver, r, show_steps=True)
        self.assertTrue(np.array_equal(before['x'], r['x']))
        self.assertEqual(before['objective'], r['objective'])
        self.assertIn('e-12', output.getvalue())

    def test_problem_data_remains_unchanged(self):
        p = LPProblem('min', [[.3]], [.1], [.7], ['>='])
        original = deepcopy(vars(p))
        solve(ClassicalSimplex, p)
        self.assertEqual(vars(p), original)

    def test_result_has_residual_report(self):
        for method in (ClassicalSimplex, DualSimplex, GeneralizedSimplex):
            _, r = solve(method, LPProblem('min', [[1]], [2.375], [1], ['>=']))
            report = r['numerical_report']
            self.assertEqual(report['arithmetic'], 'float64')
            self.assertEqual(report['max_constraint_violation'], 0.0)
            self.assertEqual(report['max_variable_sign_violation'], 0.0)

    def test_significant_coefficient_underflow_is_guarded(self):
        p = LPProblem('max', [[1e300, 1e-300]], [1], [1, 1], ['<='])
        for method in (ClassicalSimplex, GeneralizedSimplex):
            _, r = solve(method, p)
            self.assertEqual(r['status'], 'NUMERICAL_ERROR')
            self.assertIn('underflow', r['message'])

    def test_computation_does_not_call_decimal_rounding(self):
        tree = ast.parse(Path(__file__).with_name('simplex0.py').read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    names.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.append(node.func.attr)
        self.assertFalse({'round', 'around', 'rint', 'floor', 'ceil'} & set(names))

    def test_common_solve_is_inherited_not_copied(self):
        for method in (ClassicalSimplex, DualSimplex, GeneralizedSimplex):
            self.assertIs(method.solve, SimplexBase.solve)
            self.assertIn('run_algorithm', method.__dict__)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NumericBoundaryTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f'Counted helper solves: {CALLS}; additional direct API calls are covered by named tests.')
    raise SystemExit(not outcome.wasSuccessful())
