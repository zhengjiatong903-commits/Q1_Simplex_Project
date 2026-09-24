"""Optional differential tests; SciPy is an independent oracle, not a solver dependency.

Run: python test_reference.py
PSEUDOCODE: generate reproducible LPs -> solve original data with HiGHS ->
compare status and objective with our own tableau algorithms -> repeat after
positive row/objective scaling. Alternative-optimum flags are NOT compared here.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from simplex0 import LPProblem, ClassicalSimplex, DualSimplex, GeneralizedSimplex

SEED = 20260924
rng = np.random.default_rng(SEED)
counts = Counter()
reference_statuses = Counter()
guarded = []


def oracle(p):
    n = len(p.c)
    A = np.asarray(p.A, dtype=float).reshape(len(p.b), n)
    inequalities, inequality_rhs, equalities, equality_rhs = [], [], [], []
    for row, b, sign in zip(A, p.b, p.signs):
        if sign == "=":
            equalities.append(row)
            equality_rhs.append(b)
        else:
            multiplier = 1 if sign == "<=" else -1
            inequalities.append(multiplier * row)
            inequality_rhs.append(multiplier * b)
    bounds = [{"nonnegative": (0, None), "nonpositive": (None, 0),
               "free": (None, None)}[kind]
              for kind in (p.variable_types or ["nonnegative"] * n)]
    sign = 1 if p.sense == "min" else -1
    r = linprog(sign * np.asarray(p.c),
                A_ub=np.asarray(inequalities) if inequalities else None,
                b_ub=np.asarray(inequality_rhs) if inequalities else None,
                A_eq=np.asarray(equalities) if equalities else None,
                b_eq=np.asarray(equality_rhs) if equalities else None,
                bounds=bounds, method="highs", options={"presolve": False})
    statuses = {0: "OPTIMAL", 2: "INFEASIBLE", 3: "UNBOUNDED"}
    if r.status not in statuses:
        # A retry with presolve is independent from the implementation under test.
        r = linprog(sign * np.asarray(p.c),
                    A_ub=np.asarray(inequalities) if inequalities else None,
                    b_ub=np.asarray(inequality_rhs) if inequalities else None,
                    A_eq=np.asarray(equalities) if equalities else None,
                    b_eq=np.asarray(equality_rhs) if equalities else None,
                    bounds=bounds, method="highs")
    if r.status not in statuses:
        raise AssertionError(f"The reference failed, so no comparison can be claimed: {r.message}")
    return statuses[r.status], (sign * r.fun + p.objective_constant if r.status == 0 else None)


def compare(p, reference, methods, label, objective_factor=1.0):
    target_status, target_value = reference
    for method in methods:
        solver = method(p)
        r = solver.solve(check_alternatives=False)
        if r["status"] == "NUMERICAL_ERROR":
            guarded.append({"case": label, "algorithm": method.__name__,
                            "expected": target_status, "message": r["message"],
                            "problem": {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                        for key, value in vars(p).items()}})
            continue  # This is an INCONCLUSIVE exit, never counted as a match.
        assert r["status"] == target_status, (label, method.__name__, r, vars(p), reference)
        if target_status == "OPTIMAL":
            # Undo the applied test scale before comparing objective accuracy.
            actual = r["objective"] / objective_factor
            assert np.isclose(actual, target_value, atol=2e-6, rtol=2e-6), (
                label, method.__name__, actual, target_value, r)
        for h in solver.history:
            if h["phase"] in ("DUAL", "GENERALIZED_DUAL"):
                assert h["dual_feasible"], (label, h)
            if h["phase"] in ("PHASE_I", "PHASE_II", "PHASE_I_CLEANUP", "GENERALIZED_PRIMAL"):
                assert h["primal_feasible"], (label, h)
        counts[(method.__name__, label.split('/')[0])] += 1


def random_problem(dual_ready=False):
    n = int(rng.integers(1, 6))
    m = int(rng.integers(0, 9))
    A = rng.integers(-5, 6, size=(m, n)).astype(float)
    b = rng.integers(-8, 12, size=m).astype(float)
    signs = rng.choice(["<=", ">=", "="], size=m, p=[.5, .35, .15]).tolist()
    sense = "min" if dual_ready else str(rng.choice(["min", "max"]))
    c = rng.integers(0, 6, size=n) if dual_ready else rng.integers(-5, 6, size=n)
    types = (["nonnegative"] * n if dual_ready else
             rng.choice(["nonnegative", "nonpositive", "free"], size=n,
                        p=[.65, .15, .2]).tolist())
    return LPProblem(sense, A, b, c, signs, types, float(rng.integers(-3, 4)))


def main():
    for i in range(300):
        p = random_problem()
        reference = oracle(p)
        reference_statuses[("mixed", reference[0])] += 1
        compare(p, reference, (ClassicalSimplex, GeneralizedSimplex), f"mixed/{i}")
        row_factors = 10.0 ** rng.choice([-10, -5, 0, 5, 10], len(p.b))
        objective_factor = 10.0 ** int(rng.choice([-12, -6, 0, 6, 12]))
        scaled = LPProblem(
            p.sense, np.asarray(p.A) * row_factors[:, None], np.asarray(p.b) * row_factors,
            np.asarray(p.c) * objective_factor, p.signs, p.variable_types,
            p.objective_constant * objective_factor)
        compare(scaled, reference, (ClassicalSimplex, GeneralizedSimplex),
                f"scaled/{i}", objective_factor)
    for i in range(200):
        p = random_problem(dual_ready=True)
        reference = oracle(p)
        reference_statuses[("dual-ready", reference[0])] += 1
        compare(p, reference, (DualSimplex, GeneralizedSimplex), f"dual-ready/{i}")
    print("Seed:", SEED)
    print("Independent reference cases:", dict(reference_statuses))
    print("Solver comparisons:", dict(counts))
    print("Total successful status/objective comparisons:", sum(counts.values()))
    print("Numerical-guard exits (NOT counted as successful matches):", len(guarded))
    print("Numerical guards by algorithm:", dict(Counter(g["algorithm"] for g in guarded)))
    Path(__file__).with_name("REFERENCE_GUARDS.json").write_text(json.dumps(guarded, indent=2))
    print("Optional alternative-optimum checks were disabled in these comparisons.")


if __name__ == "__main__":
    main()
