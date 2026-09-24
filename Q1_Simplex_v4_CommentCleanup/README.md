# Q1 simplex code — reviewed v4

This version reviews the numerical boundaries and repetition in the previous
English v3. It retains Classical (Two-Phase), strict Dual, and Taha's
feasibility-first Generalized solver. It is a teaching implementation, not a
certified exact-arithmetic optimizer.

## Files and commands

- `simplex0.py`: the actual solvers and LP input/output API.
- `run_example.py`: change problem data here; do not edit the pivot engine to change a problem.
- `test_simplex.py`: the existing deterministic algorithm cases.
- `test_numeric_boundaries.py`: new rounding, boundary, tie, constant and refactoring checks.
- `test_excel.py`: existing input/export workflow checks.
- `test_reference.py`: optional comparisons with SciPy/HiGHS; this is not used by the solvers.
- `PSEUDOCODE.txt`: an outline of the actual call structure.
- `BEFORE_AFTER.txt`, `TEST_LOG.txt`: measured results, not guarantees for all inputs.
- `REFERENCE_GUARDS.json`: inputs where the optional reference check ended numerically inconclusively.

```bash
python3 -m pip install -r requirements.txt
python3 test_simplex.py
python3 test_numeric_boundaries.py
python3 test_excel.py
python3 run_example.py --method generalized --example generalized --show-steps
python3 run_example.py --method dual --example dual --show-steps
```

Optional independent checks:

```bash
python3 -m pip install -r requirements-test.txt
python3 test_reference.py
```

For an Excel input/output example:

```bash
python3 run_example.py --method classical --input input_example.xlsx --output result.xlsx
```

Existing output files are protected unless `--overwrite` is passed. The Excel
input schema remains Settings(key,value), Variables(variable_name,
objective_coefficient,variable_type), and Constraints(variable columns,sign,rhs).
The core API accepts the same LPProblem data as v3. All three classes inherit
one `solve(check_alternatives=True, initial_basis=None)` method; subclasses now
implement `run_algorithm`. Strict Dual needs a dual-feasible basis; INVALID_START
is not proof that the original LP is infeasible. Classical rejects a supplied
basis because its initialization is slacks/Two-Phase.

## Decimal bounds and rounding policy

1. Keep the original LP data unchanged. Convert real coefficients to float64,
   but never round a bound to an integer or to a chosen number of decimal places.
2. Do not round the tableau after each pivot. Hardware floating-point operations
   still have representation and rounding errors: “no explicit round()” does
   NOT mean exact arithmetic.
3. Format printed copies only. Display uses scientific notation where needed;
   returned x and objective values are not replaced by formatted strings.
4. Local Phase-I cleanup may still set an artificial basic RHS to zero, but only
   below a roundoff-scale guard, with a history entry and original-model checks.
5. Tolerances control decisions, not decimal places. Default pivot/sign tolerance
   is 1e-9; the residual tolerance is 1e-8 on scaled equivalent constraints. A
   separate `64 * machine_epsilon * scale` heuristic catches several ambiguous
   signs/residuals. It is NOT a rigorous error bound for all pivots or all inputs.
6. A gray-zone case can return NUMERICAL_ERROR. It is not silently called
   INFEASIBLE, UNBOUNDED, or OPTIMAL. Do not count it as a successful solve.

For the close-bound conflict x <= 1, x >= 1.0000000005, default v4 returns a
numerical uncertainty. For this specific well-scaled example, setting both
user tolerances to 1e-12 resolves the result as INFEASIBLE. Tightening tolerances
is not a universal cure; check units, conditioning and the mathematical model.

```python
solver = ClassicalSimplex(problem, tolerance=1e-12, solution_tolerance=1e-12)
result = solver.solve()
```

Equivalent command-line options are `--tolerance` and `--solution-tolerance`.

## What changed

- Classical and Dual use roundoff-scale ratio ties, not a broad feasibility
  tolerance as permission to choose a genuinely larger ratio.
- A small but material Phase-I artificial residual is not normalized away.
- Significant small coefficients cannot justify deleting a supposedly redundant
  row, or ignoring a more restrictive candidate pivot.
- Final constraint/sign residuals and small negative reduced costs receive
  additional gray-zone checks. Successful results include `numerical_report`.
- Objective constants stay out of tableau operations. The result returns both
  `objective_variable_part` and `objective_constant`; a float total can still
  lose the smaller term when the scales are vastly different.
- Scaling that erases a represented nonzero coefficient now triggers an error.
- Three repeated solve/error/result wrappers are replaced by one base method.
  Alternative-optimum postprocessing is also shared.
- Comments explain mathematical purpose rather than restate assignments.
  The test requiring at least twenty PSEUDOCODE labels was removed. Tests should
  check behavior, not the number of explanatory labels.
- Deterministic objective assertions now scale their tolerance to the expected
  value so that zero cannot pass as a nonzero tiny objective.

This is not a short beginner-only program. Input validation, Three algorithms,
Two-Phase cleanup, certificates, Excel I/O and distinct-original-solution checks
still take code. The aim is less duplication and clearer responsibilities, not
fewer lines at the cost of omitted checks. No claim about AI-authorship detection
is made.

## Limits to numerical rigor

OPTIMAL means a float64 result that passed this implementation's numerical
checks, not a mathematical proof over exact real or decimal input. The probe in
BEFORE_AFTER.txt deliberately shows that conflicts around 1e-15 may still be
accepted within roundoff-scale tolerances. Values already rounded/underflowed
before reaching the solver cannot be recovered. Near-singular bases, severe
cancellation, large coefficient ranges, overflow and exact multiplicity remain
limitations. Decimal/Fraction inputs are converted to float; this is not an
exact decimal or rational solver. A full rational implementation, not merely
wrapping float results in Fraction, would be needed for exact certificates.

Degeneracy and alternative-optimum labels are also tolerance-based. Coordinate
search for alternative optima solves additional LPs using Classical; it is
optional and can return UNCONFIRMED without erasing a verified primary result.
Tests of random LP statuses/objectives do not validate all alternative-optimum
flags or prove correctness on every problem.

## References for numerical policy

- Python floating-point tutorial: https://docs.python.org/3/tutorial/floatingpoint.html
- NumPy isclose warning: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
- Gurobi tolerances/scaling discussion: https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html
- Taha, Operations Research: An Introduction, 10th ed., Sections 4.4.1–4.4.2
  (the course source for the algorithms, not a claim that these guards are textbook requirements).
