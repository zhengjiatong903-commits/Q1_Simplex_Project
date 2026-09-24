"""Deterministic tests. Run: python test_simplex.py

PSEUDOCODE: construct an LP -> invoke a named algorithm -> compare with an
independently known answer -> verify each recorded tableau and stage invariant.
These tests verify Dual and Generalized pivot paths, not empty placeholders.
"""
from collections import Counter
from pathlib import Path
import tempfile
import numpy as np
from simplex0 import LPProblem, ClassicalSimplex, DualSimplex, GeneralizedSimplex

CASES_RUN = 0
ASSERTIONS = 0
ALGORITHMS = Counter()


def expect(condition, message="Assertion failed"):
    global ASSERTIONS
    ASSERTIONS += 1
    assert condition, message


def check(name, method, problem, status="OPTIMAL", value=None, x=None,
          alternative=None, degenerate=None, solve_options=None, **options):
    global CASES_RUN
    solver = method(problem, **options)
    result = solver.solve(**(solve_options or {}))
    expect(result["status"] == status, f"{name}: {result}")
    if value is not None:
        # A fixed large atol would let zero pass as a tiny nonzero objective.
        absolute_error = 1e-12 if value == 0 else 1e-10 * abs(value)
        expect(np.isclose(result["objective"], value, atol=absolute_error, rtol=1e-9), result)
    if x is not None:
        expect(np.allclose(result["x"], x, atol=1e-12, rtol=1e-9), result)
    if alternative is not None:
        expect(result["alternative_info"]["status"] == alternative, result)
    if degenerate is not None:
        expect(result["final_degenerate"] is degenerate, result)
    for state in solver.history:
        expect(np.isfinite(state["T"]).all(), f"{name}: nonfinite tableau")
        if state["phase"] in ("DUAL", "GENERALIZED_DUAL"):
            expect(state["dual_feasible"], f"{name}: dual invariant violated")
        if state["phase"] in ("PHASE_I", "PHASE_I_CLEANUP", "PHASE_II",
                               "GENERALIZED_PRIMAL"):
            expect(state["primal_feasible"], f"{name}: primal invariant violated")
    CASES_RUN += 1
    ALGORITHMS[method.__name__] += 1
    print(f"PASS {CASES_RUN:02d} | {method.__name__:<20} | {name}: {status}")
    return solver, result


def run_tests():
    # The same known problems exercise independent Classical and Generalized paths.
    for method in (ClassicalSimplex, GeneralizedSimplex):
        check("bounded max", method,
              LPProblem("max", [[1, 2], [4, 0], [0, 4]], [8, 16, 12],
                        [2, 3], ["<="] * 3), value=14, x=[4, 2], alternative="NO")
        check("min with >=", method,
              LPProblem("min", [[1, 1], [1, 2]], [4, 6], [2, 3], [">=", ">="]),
              value=10, x=[2, 2], alternative="NO")
        check("negative RHS", method,
              LPProblem("max", [[1, 1], [-1, 1]], [5, -1], [2, 1], ["<="] * 2),
              value=10, x=[5, 0])
        check("mixed equality and inequalities", method,
              LPProblem("max", [[1, 1], [1, 0], [0, 1]], [4, 1, 2],
                        [3, 2], ["=", ">=", "<="]), value=12, x=[4, 0])
        check("free variable; original optimum unique", method,
              LPProblem("min", [[1]], [-2], [1], ["="], ["free"]),
              value=-2, x=[-2], alternative="NO")
        check("nonpositive variable and constant", method,
              LPProblem("min", [[1]], [-3], [1], [">="], ["nonpositive"], 5),
              value=2, x=[-3])
        check("infeasible", method,
              LPProblem("max", [[1], [1]], [1, 2], [1], ["<=", ">="]),
              status="INFEASIBLE")
        _, result = check("unbounded without rows", method,
              LPProblem("max", [], [], [1], []), status="UNBOUNDED")
        expect(result["certificate"]["direction"][0] > 0)
        check("unbounded after feasible-start repair", method,
              LPProblem("max", [[1]], [1], [1], [">="]), status="UNBOUNDED")
        check("duplicate equalities", method,
              LPProblem("max", [[1], [2]], [1, 2], [1], ["=", "="]),
              value=1, x=[1], alternative="NO")
        check("different optimal points", method,
              LPProblem("max", [[1, 1]], [1], [1, 1], ["<="]),
              value=1, alternative="YES")
        check("degenerate but unique original optimum", method,
              LPProblem("max", [[1, 1]], [0], [1, 0], ["<="]),
              value=0, x=[0, 0], alternative="NO", degenerate=True)
        check("constant objective, multiple points", method,
              LPProblem("max", [], [], [0], []), value=0, alternative="YES")
        check("NaN input", method,
              LPProblem("max", [[float("nan")]], [1], [1], ["<="]),
              status="INVALID_INPUT")
        check("strict inequality is not silently replaced", method,
              LPProblem("max", [[1]], [1], [1], ["<"]), status="INVALID_INPUT")
        check("shape mismatch", method,
              LPProblem("max", [[1, 2]], [1], [1], ["<="]), status="INVALID_INPUT")
        check("pivot limit", method,
              LPProblem("max", [[1]], [1], [1], ["<="]),
              status="ITERATION_LIMIT", max_iterations=0)
        check("scaled tiny row", method,
              LPProblem("max", [[1e-10]], [1e-10], [1], ["<="]), value=1, x=[1])
        check("scaled tiny objective", method,
              LPProblem("max", [[1]], [1], [1e-12], ["<="]), value=1e-12, x=[1])
        check("contradictory zero row", method,
              LPProblem("max", [[0]], [-1], [1], ["<="]), status="INFEASIBLE")
        check("two free variables retain correct indices", method,
              LPProblem("max", [[1, 0], [0, 1]], [-2, 3], [2, -1],
                        ["=", "="], ["free", "free"]),
              value=-7, x=[-2, 3], alternative="NO")
        check("Beale cycling example with Bland ordering", method,
              LPProblem("max", [[.5, -5.5, -2.5, 9], [.5, -1.5, -.5, 1],
                                [1, 0, 0, 0]], [0, 0, 1],
                        [10, -57, -9, -24], ["<="] * 3),
              value=1, solve_options={"check_alternatives": False})

    # Taha Section 4.4.1: the algorithm must use negative-RHS dual pivots.
    p = LPProblem("min", [[3, 1, 1], [-3, 3, 1], [1, 1, 1]],
                  [3, 6, 3], [3, 2, 1], [">=", ">=", "<="])
    for method in (DualSimplex, GeneralizedSimplex):
        s, r = check("Taha Example 4.4-1", method, p, value=4.5, x=[0, 1.5, 1.5])
        expect(r["initial_feasibility"] == {"primal": False, "dual": True})
        expect(any(h["event"] == "PIVOT" and "DUAL" in h["phase"] for h in s.history))
        expect(all(v["kind"] != "artificial" for v in s.variable_info))

    # Taha Section 4.4.2: both conditions fail initially; MUST repair then optimize.
    p = LPProblem("max", [[-1, 2, -2], [-1, 1, 1], [2, -1, 4]],
                  [8, 4, 10], [0, 0, 2], [">=", "<=", "<="])
    s, r = check("Taha Example 4.4-2", GeneralizedSimplex, p,
                 value=28 / 9, x=[56 / 9, 26 / 3, 14 / 9])
    expect(r["initial_feasibility"] == {"primal": False, "dual": False})
    expect(any(h["event"] == "PIVOT" and h["phase"] == "GENERALIZED_FEASIBILITY"
               for h in s.history))
    expect(any(h["event"] == "PIVOT" and h["phase"] == "GENERALIZED_PRIMAL"
               for h in s.history))
    expect(not any(h["phase"].startswith("PHASE_I") for h in s.history))
    expect(all(v["kind"] != "artificial" for v in s.variable_info))
    check("strict dual must reject non-dual-feasible basis", DualSimplex, p,
          status="INVALID_START")

    for method in (DualSimplex, GeneralizedSimplex):
        check("multiple dual pivots preserve only dual feasibility", method,
              LPProblem("min", [[1, 0], [0, 1]], [1, 2], [1, 1], [">=", ">="]),
              value=3, x=[1, 2])
        check("dual infeasibility certificate", method,
              LPProblem("min", [[1], [1]], [1, 2], [1], ["<=", ">="]),
              status="INFEASIBLE")
        check("dual equality expansion", method,
              LPProblem("min", [[1]], [2], [1], ["="]), value=2, x=[2])
        check("dual optimal alternatives", method,
              LPProblem("min", [[1, 1]], [1], [1, 1], [">="]),
              value=1, alternative="YES")
        check("dual initially optimal with zero pivots", method,
              LPProblem("min", [[1]], [1], [1], ["<="]),
              value=0, x=[0], max_iterations=0)
        check("dual iteration cap", method,
              LPProblem("min", [[1]], [1], [1], [">="]),
              status="ITERATION_LIMIT", max_iterations=0)
        check("duplicate initial basis is rejected", method,
              LPProblem("min", [[1], [2]], [1, 2], [1], ["<="] * 2),
              status="INVALID_START", solve_options={"initial_basis": [0, 0]})
        check("singular initial basis is rejected", method,
              LPProblem("min", [[1, 2], [2, 4]], [1, 2], [1, 1], ["<="] * 2),
              status="INVALID_START", solve_options={"initial_basis": [0, 1]})

        # Original system: x1+s1=2, x2+s2=3. Basis [x1,s2] is dual feasible
        # for max x1-x2, even though the default slack basis is not.
        check("user-supplied dual-feasible basis", method,
              LPProblem("max", [[1, 0], [0, 1]], [2, 3], [1, -1], ["<="] * 2),
              value=2, x=[2, 0], solve_options={"initial_basis": [0, 3]})

    # Rebuild the final basis with changed RHS rather than trusting a stale tableau.
    p = LPProblem("min", [[1], [1]], [1, 3], [1], [">=", "<="])
    first = DualSimplex(p)
    original = first.solve(check_alternatives=False)
    expect(original["status"] == "OPTIMAL")
    new = LPProblem("min", [[1], [1]], [4, 3], [1], [">=", "<="])
    check("RHS change makes old dual-feasible basis infeasible", DualSimplex,
          new, status="INFEASIBLE", solve_options={"initial_basis": first.basis})

    # No actual solver class may retain the previous NOT_IMPLEMENTED result.
    expect(issubclass(GeneralizedSimplex, DualSimplex))
    text = Path(__file__).with_name("simplex0.py").read_text()
    expect(not any("\u4e00" <= ch <= "\u9fff" for ch in text), "Code still contains Chinese text")
    expect("from scipy" not in text and "import scipy" not in text,
           "The core must not delegate the LP algorithm to SciPy")
    print(f"\nCompleted {CASES_RUN} deterministic solve cases; {ASSERTIONS} assertions passed.")
    print("Cases by solver:", dict(ALGORITHMS))


if __name__ == "__main__":
    run_tests()
