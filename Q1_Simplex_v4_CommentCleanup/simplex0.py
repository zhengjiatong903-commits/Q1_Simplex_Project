"""Tableau LP solvers for Q1.

Internal form: maximize, nonnegative working variables, RHS in the last column.
The last row is z + q*y = v. Primal feasibility: RHS >= 0; dual: q_N >= 0.
Classical uses Two-Phase. Dual requires a dual-feasible basis. Generalized uses
Taha's feasibility-first combination of dual-style repair and primal simplex.

Inputs and iterations use float64 without decimal rounding. Tolerances and
roundoff guards are numerical policies, not exact-arithmetic certificates.
"""

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd

# A heuristic roundoff scale, not a rigorous forward-error bound.
ROUNDING_GUARD = 64 * np.finfo(float).eps


class LPError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@dataclass
class Action:
    status: str
    row: int = None
    column: int = None
    theta: float = None


class LPProblem:
    """Original LP; variables are nonnegative unless explicitly specified."""

    def __init__(self, sense, A, b, c, signs,
                 variable_types=None, objective_constant=0):
        self.sense = deepcopy(sense)
        self.A, self.b, self.c = deepcopy((A, b, c))
        self.signs = deepcopy(signs)
        self.variable_types = deepcopy(variable_types)
        self.objective_constant = deepcopy(objective_constant)

    def validate_and_parse(self):
        """Return a validated copy without changing the original input."""
        p = deepcopy(self)
        try:
            sense = str(p.sense).strip().lower()
            p.sense = {
                "maximize": "max", "minimize": "min"
            }.get(sense, sense)

            for name in ("A", "b", "c", "objective_constant"):
                raw = np.asarray(getattr(p, name))
                if np.iscomplexobj(raw) or raw.dtype.kind == "b":
                    raise ValueError(f"{name} must contain real numbers, not complex values or booleans")
            p.c = np.asarray(p.c, dtype=float)
            p.b = np.asarray(p.b, dtype=float)
            p.A = np.asarray(p.A, dtype=float)
            p.objective_constant = float(p.objective_constant)

            if p.c.ndim != 1 or len(p.c) == 0:
                raise ValueError("c must be a nonempty one-dimensional vector")
            if p.b.ndim != 1:
                raise ValueError("b must be a one-dimensional vector")

            m, n = len(p.b), len(p.c)
            if m == 0 and p.A.shape == (0,):
                p.A = np.empty((0, n))
            if p.A.shape != (m, n):
                raise ValueError(f"A must have shape {(m, n)}")

            p.signs = [
                str(s).strip().replace("≤", "<=").replace("≥", ">=")
                for s in p.signs
            ]
            if len(p.signs) != m:
                raise ValueError("The number of signs must equal the number of constraints")
            if any(s not in ("<=", ">=", "=") for s in p.signs):
                raise ValueError("Constraint signs must be <=, >=, or =")

            if p.variable_types is None:
                p.variable_types = ["nonnegative"] * n
            p.variable_types = [
                str(t).strip().lower() for t in p.variable_types
            ]
            allowed = {"nonnegative", "nonpositive", "free"}
            if len(p.variable_types) != n:
                raise ValueError("The number of variable types must equal the number of variables")
            if any(t not in allowed for t in p.variable_types):
                raise ValueError("Variable types must be nonnegative, nonpositive, or free")
            if p.sense not in ("max", "min"):
                raise ValueError("sense must be max or min")

            for name in ("A", "b", "c", "objective_constant"):
                if not np.isfinite(getattr(p, name)).all():
                    raise ValueError(f"{name} contains a missing value or NaN/Infinity")
        except (ValueError, TypeError, OverflowError) as exc:
            raise LPError("INVALID_INPUT", str(exc)) from exc
        return p

    @classmethod
    def from_excel(cls, path):
        """
        Settings: key, value
        Variables: variable_name, objective_coefficient, variable_type (optional)
        Constraints: one coefficient column per variable, sign, rhs
        """
        try:
            sheets = pd.read_excel(path, sheet_name=None)
            settings = sheets["Settings"]
            variables = sheets["Variables"]
            constraints = sheets["Constraints"].copy()

            keys = settings["key"]
            if keys.isna().any():
                raise ValueError("Settings keys cannot be empty")
            keys = keys.astype(str).str.strip()
            if keys.duplicated().any():
                raise ValueError("Settings keys must be unique")
            config = dict(zip(keys, settings["value"]))

            raw_names = variables["variable_name"]
            if raw_names.isna().any():
                raise ValueError("Variable names cannot be empty")
            names = raw_names.astype(str).str.strip().tolist()
            if (any(not name for name in names)
                    or len(set(names)) != len(names)
                    or any(name in ("sign", "rhs") for name in names)):
                raise ValueError("Variable names are empty, repeated, or conflict with sign/rhs")

            constraints.columns = [
                str(name).strip() for name in constraints.columns
            ]
            if constraints.columns.duplicated().any():
                raise ValueError("Constraints contains duplicate column names")
            if set(constraints.columns) != set(names + ["sign", "rhs"]):
                raise ValueError("Constraints columns must be the variable names, sign, and rhs")

            types = (
                variables["variable_type"].tolist()
                if "variable_type" in variables else None
            )
            problem = cls(
                config["sense"],
                constraints[names].to_numpy(),
                constraints["rhs"].to_numpy(),
                variables["objective_coefficient"].to_numpy(),
                constraints["sign"].tolist(),
                types,
                config.get("objective_constant", 0),
            )
            return problem.validate_and_parse()
        except LPError:
            raise
        except (KeyError, ValueError, TypeError) as exc:
            raise LPError("INVALID_INPUT", f"Invalid Excel format: {exc}") from exc


class SimplexBase:
    def __init__(self, problem, tolerance=1e-9,
                 solution_tolerance=1e-8, max_iterations=10000):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(solution_tolerance) or solution_tolerance <= 0:
            raise ValueError("solution_tolerance must be finite and positive")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")

        self.original_problem = deepcopy(problem)
        self.eps = tolerance
        self.sol_eps = solution_tolerance
        self.limit = max_iterations
        self.reset()

    def reset(self):
        self.T = None
        self.p = None
        self.basis = []
        self.variable_info = []
        self.history = []
        self.transformations = []
        self.iteration = 0
        self.phase = "NOT_STARTED"
        self.last_action = None
        self.result = None
        self.objective_scale = 1.0
        self.constraint_scale = None
        self.A_verify = None
        self.b_verify = None
        self.c_verify = None
        self.row_transform = None
        self.slack_matrix = None
        self.slack_rhs = None
        self.initial_feasibility = None
        self.initialization = None

    def prepare_working_problem(self):
        self.p = self.original_problem.validate_and_parse()
        p = self.p
        columns = []

        # M maps nonnegative working variables to original variables: x = M*y.
        for j, kind in enumerate(p.variable_types):
            weights = {
                "nonnegative": [1],
                "nonpositive": [-1],
                "free": [1, -1],
            }[kind]
            terms = []
            for weight in weights:
                vector = np.zeros(len(p.c))
                vector[j] = weight
                k = len(columns)
                columns.append(vector)
                self.variable_info.append({
                    "variable_id": k,
                    "name": f"y{k + 1}",
                    "kind": "decision",
                    "original": j,
                    "weight": weight,
                })
                terms.append(f"{weight:+d}*y{k + 1}")
            self.transformations.append({
                "event": "VARIABLE",
                "details": f"x{j + 1} = {' '.join(terms)}",
            })

        self.M = np.column_stack(columns)
        self.n_work = self.M.shape[1]
        # Do not floor the row divisor at 1: tiny valid rows must be scaled up.
        self.constraint_scale = np.max(np.abs(p.A), axis=1)
        zero_rows = self.constraint_scale == 0
        self.constraint_scale[zero_rows] = np.where(
            np.abs(p.b[zero_rows]) > 0, np.abs(p.b[zero_rows]), 1.0
        )
        self.A_verify = p.A / self.constraint_scale[:, None]
        self.b_verify = p.b / self.constraint_scale
        for i, divisor in enumerate(self.constraint_scale):
            if divisor != 1.0:
                self.transformations.append({
                    "event": "SCALE_ROW",
                    "details": f"Divide original constraint {i + 1} by {divisor:.17g}",
                })

        self.A = self.A_verify @ self.M
        self.b = self.b_verify.copy()
        self.signs = p.signs.copy()
        self.objective_sign = 1 if p.sense == "max" else -1

        # Keep the objective constant out of pivot decisions; add it back in the result.
        self.objective_scale = float(np.max(np.abs(p.c))) or 1.0
        self.c_verify = p.c / self.objective_scale
        self.c_work = self.objective_sign * (self.c_verify @ self.M)
        self.constant_work = 0.0
        if (np.any((p.A != 0) & (self.A_verify == 0))
                or np.any((p.b != 0) & (self.b_verify == 0))
                or np.any((p.c != 0) & (self.c_verify == 0))):
            raise LPError("NUMERICAL_ERROR", "Scaling underflow erased a nonzero input; change units")
        if self.objective_scale != 1.0:
            self.transformations.append({
                "event": "SCALE_OBJECTIVE",
                "details": f"Divide working objective by {self.objective_scale:.17g}",
            })

    def variable_id(self, column):
        return self.variable_info[column]["variable_id"]

    def nonbasic_columns(self):
        return sorted(set(range(self.T.shape[1] - 1)) - set(self.basis))

    def feasibility_flags(self):
        primal = bool(np.all(self.T[:-1, -1] >= -self.eps))
        nonbasic = self.nonbasic_columns()
        dual = bool(np.all(self.T[-1, nonbasic] >= -self.eps))
        return primal, dual

    def validate_tableau_structure(self):
        m, n = self.T.shape[0] - 1, self.T.shape[1] - 1
        bad = (
            not np.isfinite(self.T).all()
            or len(self.basis) != m
            or len(set(self.basis)) != m
            or len(self.variable_info) != n
            or any(j < 0 or j >= n for j in self.basis)
        )
        if bad:
            raise LPError("NUMERICAL_ERROR", "Tableau structure or finite-number check failed")

        if m and (
            not np.allclose(self.T[:-1, self.basis], np.eye(m),
                            atol=10 * self.eps, rtol=0)
            or np.any(np.abs(self.T[-1, self.basis]) > 10 * self.eps)
        ):
            raise LPError("NUMERICAL_ERROR", "Basis columns or objective row are not canonical")

    def record_state(self, event, **details):
        primal, dual = self.feasibility_flags()
        self.history.append({
            "event": event,
            "phase": self.phase,
            "iteration": self.iteration,
            "basis": [self.variable_id(j) for j in self.basis],
            "columns": [v["variable_id"] for v in self.variable_info],
            "primal_feasible": primal,
            "dual_feasible": dual,
            "degenerate": (
                bool(np.any(np.abs(self.T[:-1, -1]) <= self.eps))
                if primal else None
            ),
            "objective_value": float(self.T[-1, -1]),
            # History records the internal objective, not necessarily the original value.
            "objective_scale": (
                1.0 if self.phase.startswith("PHASE_I") else self.objective_scale
            ),
            **details,
            "T": self.T.copy(),
        })

    def set_objective(self, cost, constant=0):
        # Eliminate basic variables to put the new objective row in canonical form.
        self.T[-1] = np.r_[-np.asarray(cost), constant]
        for i, column in enumerate(self.basis):
            factor = self.T[-1, column]
            self.T[-1] -= factor * self.T[i]
        self.validate_tableau_structure()
        self.record_state("SET_OBJECTIVE")

    def pivot(self, row, column, theta=None):
        # Pseudocode: normalize pivot row -> eliminate column -> update basis.
        if self.iteration >= self.limit:
            raise LPError("ITERATION_LIMIT", "The total pivot limit was reached")
        pivot_value = float(self.T[row, column])
        if abs(pivot_value) <= self.eps:
            raise LPError("NUMERICAL_ERROR", "The pivot is too small for reliable elimination")

        entering = self.variable_id(column)
        leaving = self.variable_id(self.basis[row])
        self.T[row] /= pivot_value
        if self.row_transform is not None:
            self.row_transform[row] /= pivot_value

        for i in range(len(self.T)):
            if i != row:
                factor = self.T[i, column]
                self.T[i] -= factor * self.T[row]
                if self.row_transform is not None and i < len(self.basis):
                    self.row_transform[i] -= factor * self.row_transform[row]

        self.basis[row] = column
        self.iteration += 1
        self.validate_tableau_structure()
        self.record_state(
            "PIVOT", entering=entering, leaving=leaving,
            pivot=pivot_value, theta=theta,
        )

    def check_iteration_invariant(self):
        """Subclasses specify the invariant; the base validates shared structure.

        Classical checks primal feasibility.
        Dual checks dual feasibility, not nonnegative RHS values.
        """
        return None

    def iterate(self, select_pivot, phase):
        self.phase = phase
        self.record_state("INITIAL")
        while True:
            self.validate_tableau_structure()
            self.check_iteration_invariant()
            action = select_pivot()
            self.last_action = action
            if action.status != "PIVOT":
                self.record_state("TERMINATE", status=action.status)
                return action.status
            self.pivot(action.row, action.column, action.theta)
            self.check_iteration_invariant()

    def internal_values(self):
        y = np.zeros(self.T.shape[1] - 1)
        y[self.basis] = self.T[:-1, -1]
        return y

    def extract_current_original_values(self):
        x = self.M @ self.internal_values()[:self.n_work]
        z = float(self.p.c @ x + self.p.objective_constant)
        return x, z

    def verify_vector(self, vector, ray=False):
        """Check the unchanged original constraints through scaled equivalents."""
        if not np.isfinite(vector).all():
            raise LPError("NUMERICAL_ERROR", "The solution or direction contains nonfinite numbers")
        rhs = np.zeros_like(self.b_verify) if ray else self.b_verify
        residual = self.A_verify @ vector - rhs
        magnitude = np.abs(self.A_verify) @ np.abs(vector)
        scale = magnitude if ray else 1 + np.abs(rhs) + magnitude
        limits = self.sol_eps * scale
        # Residuals above roundoff but below user tolerance remain uncertain.
        guards = np.minimum(limits, ROUNDING_GUARD * scale)
        violations = np.array([
            max(0.0, value) if sign == "<=" else
            max(0.0, -value) if sign == ">=" else abs(value)
            for value, sign in zip(residual, self.p.signs)
        ])
        if np.any(violations > limits):
            raise LPError("NUMERICAL_ERROR", "A constraint residual exceeds the solution tolerance")
        if np.any(violations > guards):
            raise LPError("NUMERICAL_ERROR", "A constraint residual is in the tolerance gray zone; rescale or tighten tolerances")
        sign_violations = []
        for value, kind in zip(vector, self.p.variable_types):
            violation = (max(0.0, -value) if kind == "nonnegative" else
                         max(0.0, value) if kind == "nonpositive" else 0.0)
            scale_v = abs(value) if ray else 1 + abs(value)
            if violation > min(self.sol_eps, ROUNDING_GUARD) * scale_v:
                raise LPError("NUMERICAL_ERROR", "A variable sign is uncertain or violated; do not round it to zero")
            sign_violations.append(violation)
        raw_rhs = np.zeros_like(self.p.b) if ray else self.p.b
        raw_residual = self.p.A @ vector - raw_rhs
        raw_violations = np.array([
            max(0.0, v) if sign == "<=" else max(0.0, -v) if sign == ">=" else abs(v)
            for v, sign in zip(raw_residual, self.p.signs)
        ])
        return {
            "max_constraint_violation": float(np.max(raw_violations, initial=0.0)),
            "max_scaled_constraint_violation": float(np.max(violations, initial=0.0)),
            "max_variable_sign_violation": max(sign_violations, default=0.0),
        }

    def build_unbounded_ray(self, column):
        d = np.zeros(self.T.shape[1] - 1)
        d[column] = 1
        d[self.basis] = -self.T[:-1, column]

        x, _ = self.extract_current_original_values()
        direction = self.M @ d[:self.n_work]
        self.verify_vector(x)
        self.verify_vector(direction, ray=True)

        gain = self.objective_sign * float(self.c_verify @ direction)
        if not np.isfinite(gain) or gain <= self.eps:
            raise LPError("NUMERICAL_ERROR", "The proposed ray does not strictly improve the original objective")
        return {"point": x, "direction": direction}

    def make_result(self, status, message="", certificate=None):
        x = value = variable_value = None
        numerical_report = {
            "arithmetic": "float64",
            "pivot_tolerance": self.eps,
            "solution_tolerance": self.sol_eps,
            "interpretation": "Tolerance-based result, not an exact-arithmetic certificate",
        }
        if status == "OPTIMAL":
            if not all(self.feasibility_flags()):
                raise LPError("NUMERICAL_ERROR", "Final feasibility or optimality verification failed")
            q = self.T[-1, self.nonbasic_columns()]
            q_guard = np.minimum(self.eps, ROUNDING_GUARD * np.maximum(1.0, np.abs(q)))
            if np.any(q < -q_guard):
                raise LPError("NUMERICAL_ERROR", "A reduced cost is in the tolerance gray zone; optimality is unconfirmed")
            x, value = self.extract_current_original_values()
            numerical_report.update(self.verify_vector(x))
            variable_value = float(self.p.c @ x)
            scaled_value = self.objective_sign * float(self.c_verify @ x)
            table_value = float(self.T[-1, -1])
            tol = self.sol_eps * (1 + abs(scaled_value) + abs(table_value))
            if not np.isfinite(value) or abs(scaled_value - table_value) > tol:
                raise LPError("NUMERICAL_ERROR", "Variable objective disagrees with the tableau objective")
            numerical_report["scaled_objective_residual"] = abs(scaled_value - table_value)

        phases = list(dict.fromkeys(
            ["PHASE_I", "PHASE_I_CLEANUP", "PHASE_II"]
            + [state["phase"] for state in self.history]
        ))
        degeneracy_by_phase = {
            phase: any(
                s["phase"] == phase and s.get("degenerate") is True
                for s in self.history
            )
            for phase in phases
        }
        final_degenerate = (
            bool(np.any(np.abs(self.T[:-1, -1]) <= self.eps))
            if status == "OPTIMAL" else None
        )
        result = {
            "status": status,
            "algorithm": type(self).__name__,
            "initialization": self.initialization,
            "initial_feasibility": self.initial_feasibility,
            "message": message,
            "x": x,
            "objective": value,
            "objective_variable_part": variable_value,
            "objective_constant": self.p.objective_constant if self.p else None,
            "numerical_report": numerical_report,
            "iterations": self.iteration,
            "phase": self.phase,
            "basis": [self.variable_id(j) for j in self.basis],
            "certificate": certificate,
            "alternative_info": {"status": "NOT_CHECKED"},
            "final_degenerate": final_degenerate,
            "encountered_degeneracy": any(degeneracy_by_phase.values()),
            "degeneracy_by_phase": degeneracy_by_phase,
            "objective_scale": self.objective_scale,
            "basis_names": [self.variable_info[j]["name"] for j in self.basis],
        }
        self.result = result
        return result

    def write_result_to_excel(self, path, result=None):
        r = self.result if result is None else result
        if r is None:
            raise ValueError("Call solve() before exporting a result")

        summary = [
            (key, str(value)) for key, value in r.items()
            if key not in ("x", "certificate")
        ]
        if r["x"] is not None:
            summary.extend(
                (f"x{i + 1}", float(value))
                for i, value in enumerate(r["x"])
            )
        if r["certificate"] is not None:
            summary.append(("certificate", str(r["certificate"])))

        iterations = []
        for state in self.history:
            metadata = {
                key: str(value) for key, value in state.items() if key != "T"
            }
            for row, values in enumerate(state["T"]):
                iterations.append({
                    **metadata,
                    "tableau_row": row,
                    **{f"col_{j}": v for j, v in enumerate(values[:-1])},
                    "RHS": values[-1],
                })

        original = (
            vars(self.p) if self.p is not None
            else vars(self.original_problem)
        )
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(summary, columns=["key", "value"]).to_excel(
                writer, sheet_name="Summary", index=False
            )
            pd.DataFrame(
                [(k, str(v)) for k, v in original.items()],
                columns=["field", "value"],
            ).to_excel(writer, sheet_name="OriginalProblem", index=False)
            pd.DataFrame(self.transformations).to_excel(
                writer, sheet_name="Transformations", index=False
            )
            pd.DataFrame(iterations).to_excel(
                writer, sheet_name="Iterations", index=False
            )

    def choose_primal_pivot(self):
        # Pseudocode: choose an improving column -> minimum ratio -> Bland tie-break.
        if not self.feasibility_flags()[0]:
            return Action("INVALID_START")

        candidates = [
            j for j in self.nonbasic_columns()
            if self.T[-1, j] < -self.eps
        ]
        if not candidates:
            nonbasic = self.nonbasic_columns()
            roundoff = 64 * np.finfo(float).eps * max(
                1.0, float(np.max(np.abs(self.T[-1, :-1]), initial=0.0))
            )
            if any(self.T[-1, j] < -roundoff for j in nonbasic):
                raise LPError("NUMERICAL_ERROR", "An improving coefficient is too small; rescale the model")
            return Action("OPTIMAL")

        column = min(candidates, key=self.variable_id)
        rows = np.flatnonzero(self.T[:-1, column] > self.eps)
        if len(rows) == 0:
            # A positive coefficient below eps can still block an unbounded ray.
            if np.any(self.T[:-1, column] > 0):
                raise LPError(
                    "NUMERICAL_ERROR", "The entering column has tiny positive entries; unboundedness is uncertain"
                )
            return Action("UNBOUNDED", column=column)

        rhs = self.T[rows, -1]
        tiny_negative = rows[rhs < 0]
        if len(tiny_negative):
            self.record_state(
                "RATIO_RHS_AS_ZERO", rows=tiny_negative.tolist()
            )

        ratios = np.maximum(rhs, 0) / self.T[rows, column]
        theta = float(ratios.min())
        tie_tol = min(self.eps, ROUNDING_GUARD * max(1.0, abs(theta)))
        small_rows = np.flatnonzero((self.T[:-1, column] > 0)
                                    & (self.T[:-1, column] <= self.eps))
        if len(small_rows):
            small_ratios = np.maximum(self.T[small_rows, -1], 0) / self.T[small_rows, column]
            if np.any(small_ratios < theta - tie_tol):
                raise LPError("NUMERICAL_ERROR", "A small pivot could limit this step; change units")
        tied = rows[np.abs(ratios - theta) <= tie_tol]
        row = min(tied, key=lambda i: self.variable_id(self.basis[i]))
        return Action("PIVOT", int(row), column, theta)

    def confirm_alternative_optima(self, result):
        # Pseudocode: fix the optimal value -> vary each original coordinate.
        # Disable the alternative check in auxiliary solves to avoid recursion.
        """Check changes in ORIGINAL variables on the optimal face; avoid false multiplicity."""
        if all(
            self.T[-1, j] > self.eps for j in self.nonbasic_columns()
        ):
            return {"status": "NO"}

        p = self.p
        x_star, z_star = result["x"], result["objective"]
        variable_target = float(self.c_verify @ x_star)
        A = np.vstack([p.A, self.c_verify])
        b = np.r_[p.b, variable_target]
        signs = p.signs + ["="]
        failures = []
        auxiliary_iterations = 0

        def different_optimum(x):
            self.verify_vector(x)
            z = float(self.c_verify @ x)
            if abs(z - variable_target) > self.sol_eps * (
                1 + abs(z) + abs(variable_target)
            ):
                return False
            tol = self.sol_eps * (1 + np.maximum(np.abs(x), np.abs(x_star)))
            return bool(np.any(np.abs(x - x_star) > tol))

        for j in range(len(p.c)):
            objective = np.zeros(len(p.c))
            objective[j] = 1
            for sense in ("min", "max"):
                auxiliary = LPProblem(
                    sense, A, b, objective, signs, p.variable_types, 0
                )
                solver = ClassicalSimplex(
                    auxiliary, self.eps, self.sol_eps, self.limit
                )
                r = solver.solve(check_alternatives=False)
                auxiliary_iterations += r["iterations"]

                try:
                    if r["status"] == "OPTIMAL":
                        candidate = r["x"]
                    elif r["status"] == "UNBOUNDED":
                        point = r["certificate"]["point"]
                        direction = r["certificate"]["direction"]
                        size = np.max(np.abs(direction))
                        if size <= self.eps:
                            raise LPError("NUMERICAL_ERROR", "The auxiliary ray is too small")
                        step = max(1.0, 0.1 * np.max(np.abs(x_star))) / size
                        candidate = point + step * direction
                        if not different_optimum(candidate):
                            candidate = point + 2 * step * direction
                        if not different_optimum(candidate):
                            failures.append(f"x{j + 1}/{sense}: Auxiliary-ray verification failed")
                            continue
                    else:
                        failures.append(f"x{j + 1}/{sense}: {r['status']}")
                        continue

                    if different_optimum(candidate):
                        return {
                            "status": "YES",
                            "x": candidate,
                            "auxiliary_iterations": auxiliary_iterations,
                        }
                except LPError as exc:
                    failures.append(str(exc))

        return {
            "status": "UNCONFIRMED" if failures else "NO",
            "details": failures,
            "auxiliary_iterations": auxiliary_iterations,
        }

    def add_alternative_check(self, result, check_alternatives):
        if result["status"] == "OPTIMAL" and check_alternatives:
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise"):
                    result["alternative_info"] = self.confirm_alternative_optima(result)
            except (LPError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                result["alternative_info"] = {"status": "UNCONFIRMED", "details": str(exc)}
        return result

    def solve(self, check_alternatives=True, initial_basis=None):
        """Shared entry point; subclasses supply run_algorithm(), not another wrapper."""
        self.reset()
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                self.prepare_working_problem()
                status, certificate = self.run_algorithm(initial_basis)
                result = self.make_result(status, certificate=certificate)
        except LPError as exc:
            return self.make_result(exc.status, str(exc))
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            return self.make_result("NUMERICAL_ERROR", str(exc))
        return self.add_alternative_check(result, check_alternatives)

    def run_algorithm(self, initial_basis=None):
        raise NotImplementedError("Choose ClassicalSimplex, DualSimplex, or GeneralizedSimplex")


class ClassicalSimplex(SimplexBase):
    def check_iteration_invariant(self):
        """Every classical step must preserve primal feasibility of its current LP."""
        if not self.feasibility_flags()[0]:
            raise LPError("NUMERICAL_ERROR", "Primal feasibility was lost during a classical iteration")

    def build_initial_tableau(self):
        # Initial basis: <= adds slack; >= adds surplus and artificial; = adds artificial.
        A, b, signs = self.A.copy(), self.b.copy(), self.signs.copy()
        m = len(b)

        for i in range(m):
            if b[i] < 0:
                A[i] *= -1
                b[i] *= -1
                signs[i] = {"<=": ">=", ">=": "<=", "=": "="}[signs[i]]
                self.transformations.append({
                    "event": "FLIP_ROW", "details": f"Reverse original constraint {i + 1}"
                })

        columns = [A[:, j] for j in range(self.n_work)]
        self.artificial = set()

        def add_column(row, coefficient, kind):
            j = len(columns)
            column = np.zeros(m)
            column[row] = coefficient
            columns.append(column)
            self.variable_info.append({
                "variable_id": j,
                "name": f"{kind}_{row + 1}",
                "kind": kind,
                "original": None,
                "weight": 0,
            })
            return j

        for i, sign in enumerate(signs):
            if sign == "<=":
                self.basis.append(add_column(i, 1, "slack"))
            else:
                if sign == ">=":
                    add_column(i, -1, "surplus")
                j = add_column(i, 1, "artificial")
                self.artificial.add(j)
                self.basis.append(j)

        matrix = np.column_stack(columns)
        n = matrix.shape[1]
        self.T = np.zeros((m + 1, n + 1))
        self.T[:-1, :-1] = matrix
        self.T[:-1, -1] = b
        self.phase_two_cost = np.r_[
            self.c_work, np.zeros(n - self.n_work)
        ]
        self.validate_tableau_structure()

    def choose_pivot(self):
        return self.choose_primal_pivot()

    def phase_one(self):
        # Phase I maximizes minus the artificial sum; a positive minimum means infeasible.
        self.phase = "PHASE_I"
        cost = np.zeros(self.T.shape[1] - 1)
        cost[list(self.artificial)] = -1
        self.set_objective(cost)

        status = self.iterate(self.choose_pivot, "PHASE_I")
        if status == "UNBOUNDED":
            raise LPError("NUMERICAL_ERROR", "The Phase I auxiliary objective cannot be unbounded above")
        if status != "OPTIMAL":
            raise LPError(status, "Phase I did not finish successfully")

        W = float(self.internal_values()[list(self.artificial)].sum())
        if abs(W + self.T[-1, -1]) > self.sol_eps * (1 + abs(W)):
            raise LPError("NUMERICAL_ERROR", "Artificial-variable sum disagrees with the auxiliary objective")
        if W > self.eps:
            raise LPError("INFEASIBLE", "The optimal Phase I artificial-variable sum is positive")
        if W < -self.eps:
            raise LPError("NUMERICAL_ERROR", "The artificial-variable sum is unexpectedly negative")
        artificial_values = self.internal_values()[list(self.artificial)]
        guards = np.minimum(self.eps, ROUNDING_GUARD * np.maximum(1.0, np.abs(artificial_values)))
        if np.any(np.abs(artificial_values) > guards):
            raise LPError("NUMERICAL_ERROR", "Phase I is inside the tolerance gray zone; feasibility is unconfirmed")
        self.remove_artificial_variables()

    def remove_artificial_variables(self):
        # Remove zero artificial basic variables (or redundant rows) before deleting columns.
        self.phase = "PHASE_I_CLEANUP"

        while any(j in self.artificial for j in self.basis):
            row = next(
                i for i, j in enumerate(self.basis) if j in self.artificial
            )
            rhs = float(self.T[row, -1])
            guard = min(self.eps, ROUNDING_GUARD * max(1.0, abs(rhs)))
            if abs(rhs) > guard:
                raise LPError("NUMERICAL_ERROR", "Artificial cleanup cannot discard this nonzero RHS")
            # Only roundoff-level artificial RHS cleanup sets a value to zero; log the change.
            self.T[row, -1] = 0
            self.record_state("NORMALIZE_RHS", row=row, old_rhs=rhs)

            candidates = [
                j for j in self.nonbasic_columns()
                if j not in self.artificial
                and abs(self.T[row, j]) > self.eps
            ]
            if candidates:
                column = min(candidates, key=self.variable_id)
                self.pivot(row, column, theta=0)
                if not self.feasibility_flags()[0]:
                    raise LPError("NUMERICAL_ERROR", "Artificial cleanup lost primal feasibility")
            else:
                keep = [
                    j for j in range(self.T.shape[1] - 1)
                    if j not in self.artificial
                ]
                if np.any(np.abs(self.T[row, keep]) > min(self.eps, ROUNDING_GUARD)):
                    raise LPError("NUMERICAL_ERROR", "A small nonzero coefficient prevents deleting this row")
                self.T = np.delete(self.T, row, axis=0)
                del self.basis[row]
                self.transformations.append({
                    "event": "DELETE_ROW",
                    "details": f"Delete redundant row {row} of the current transformed tableau",
                })
                self.record_state("DELETE_ROW", row=row)

        keep = [
            j for j in range(self.T.shape[1] - 1)
            if j not in self.artificial
        ]
        remap = {old: new for new, old in enumerate(keep)}
        removed_ids = [self.variable_id(j) for j in sorted(self.artificial)]
        self.T = self.T[:, keep + [self.T.shape[1] - 1]]
        self.basis = [remap[j] for j in self.basis]
        self.variable_info = [self.variable_info[j] for j in keep]
        self.phase_two_cost = self.phase_two_cost[keep]

        # Decision columns stay first, so removing artificial columns preserves x = M*y.
        self.artificial.clear()
        self.transformations.append({
            "event": "DELETE_ARTIFICIAL",
            "details": str(removed_ids),
        })
        self.validate_tableau_structure()
        self.record_state("DELETE_ARTIFICIAL", removed=removed_ids)

    def run_algorithm(self, initial_basis=None):
        # Pseudocode: initial basis -> Phase I if needed -> original objective -> primal.
        if initial_basis is not None:
            raise LPError("INVALID_START", "Classical initializes with slacks/Two-Phase; a supplied basis is unsupported")
        self.initialization = "TWO_PHASE_OR_SLACK"
        self.build_initial_tableau()
        if self.artificial:
            self.phase_one()
        self.phase = "PHASE_II"
        self.set_objective(self.phase_two_cost, self.constant_work)
        status = self.iterate(self.choose_pivot, "PHASE_II")
        certificate = (self.build_unbounded_ray(self.last_action.column)
                       if status == "UNBOUNDED" else None)
        return status, certificate


class DualSimplex(SimplexBase):
    """Strict dual simplex on a dual-feasible basis.

    Raw constraints are converted to <= form with nonnegative slack variables.
    Negative RHS values are RETAINED. An equality becomes two equivalent
    inequalities. If the real objective is not dual feasible for the initial
    basis, return INVALID_START; do not secretly run primal or generalized steps.
    An optional initial_basis selects columns of this expanded slack system.
    """

    def build_slack_tableau(self, initial_basis=None):
        # Convert to <= and add slacks; split equalities and retain negative RHS values.
        rows, rhs, row_sources = [], [], []
        for i, sign in enumerate(self.signs):
            multipliers = [1] if sign == "<=" else [-1] if sign == ">=" else [1, -1]
            for multiplier in multipliers:
                rows.append(multiplier * self.A[i])
                rhs.append(multiplier * self.b[i])
                row_sources.append((i, multiplier))
                self.transformations.append({
                    "event": "SLACK_FORM_ROW",
                    "details": (
                        f"Expanded row {len(rows)} = {multiplier:+d} times "
                        f"scaled original row {i + 1}; append a nonnegative slack"
                    ),
                })
        m = len(rows)
        coefficients = np.asarray(rows, dtype=float).reshape(m, self.n_work)
        self.slack_matrix = np.c_[coefficients, np.eye(m)]
        self.slack_rhs = np.asarray(rhs, dtype=float)
        self.slack_row_sources = row_sources
        self.artificial = set()
        for i in range(m):
            j = self.n_work + i
            self.variable_info.append({
                "variable_id": j,
                "name": f"slack_{i + 1}",
                "kind": "slack",
                "original": None,
                "weight": 0,
            })
        n = self.slack_matrix.shape[1]
        self.T = np.zeros((m + 1, n + 1))
        self.phase_two_cost = np.r_[self.c_work, np.zeros(m)]
        self.row_transform = np.eye(m)
        self.basis = list(range(self.n_work, n))
        self.initialization = "SLACK_BASIS"

        # Rebuild B^{-1}[A|b] from the current problem, not a stale tableau.
        if initial_basis is not None:
            try:
                proposed = list(initial_basis)
                if (len(proposed) != m or len(set(proposed)) != m
                        or any(isinstance(j, (bool, np.bool_))
                               or not isinstance(j, (int, np.integer))
                               or j < 0 or j >= n for j in proposed)):
                    raise ValueError("initial_basis must contain m distinct valid column indices")
                self.basis = [int(j) for j in proposed]
                B = self.slack_matrix[:, self.basis]
                if m:
                    self.row_transform = np.linalg.solve(B, np.eye(m))
                self.initialization = "USER_BASIS_REBUILT"
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                raise LPError("INVALID_START", f"Invalid or singular initial basis: {exc}") from exc
        self.T[:-1, :-1] = self.row_transform @ self.slack_matrix
        self.T[:-1, -1] = self.row_transform @ self.slack_rhs
        self.validate_tableau_structure()

    def check_iteration_invariant(self):
        # Dual steps preserve q_N >= 0; negative RHS values may remain.
        if not self.feasibility_flags()[1]:
            raise LPError("NUMERICAL_ERROR", "Dual feasibility was lost during dual simplex")

    def choose_dual_pivot(self, feasibility_only=False):
        # Strict dual: negative RHS row -> minimize q_j/(-a_rj) over negative entries.
        # Feasibility repair uses a zero auxiliary objective, retaining the real objective row.
        # In repair mode, select the smallest-ID eligible negative entry.
        if not feasibility_only and not self.feasibility_flags()[1]:
            return Action("INVALID_START")
        negative_rows = np.flatnonzero(self.T[:-1, -1] < -self.eps)
        if len(negative_rows) == 0:
            roundoff = np.minimum(self.eps, ROUNDING_GUARD * np.maximum(
                1.0, np.abs(self.T[:-1, -1])
            ))
            if np.any(self.T[:-1, -1] < -roundoff):
                raise LPError("NUMERICAL_ERROR", "A negative basic value is too small to classify safely")
            return Action("FEASIBLE" if feasibility_only else "OPTIMAL")

        # Use Bland's leaving-variable order instead of Taha's most-negative RHS rule.
        row = min(negative_rows, key=lambda i: self.variable_id(self.basis[i]))
        nonbasic = self.nonbasic_columns()
        candidates = [j for j in nonbasic if self.T[row, j] < -self.eps]
        if not candidates:
            if any(self.T[row, j] < 0 for j in nonbasic):
                raise LPError("NUMERICAL_ERROR", "Tiny negative row entries prevent a reliable infeasibility claim")
            return Action("INFEASIBLE", row=int(row))

        if feasibility_only:
            column = min(candidates, key=self.variable_id)
            theta = float(self.T[row, -1] / self.T[row, column])
        else:
            ratios = np.array([
                max(0.0, self.T[-1, j]) / (-self.T[row, j])
                for j in candidates
            ])
            theta = float(ratios.min())
            tie_tol = min(self.eps, ROUNDING_GUARD * max(1.0, abs(theta)))
            tied = [j for j, ratio in zip(candidates, ratios)
                    if abs(ratio - theta) <= tie_tol]
            column = min(tied, key=self.variable_id)
        return Action("PIVOT", int(row), int(column), theta)

    def choose_pivot(self):
        return self.choose_dual_pivot(feasibility_only=False)

    def build_infeasibility_certificate(self, row):
        # Verify the contradictory row as a combination of the original expanded equalities.
        # Nonnegative variables cannot satisfy nonnegative coefficients with a negative RHS.
        weights = self.row_transform[row].copy()
        coefficients = weights @ self.slack_matrix
        rhs = float(weights @ self.slack_rhs)
        errors = self.sol_eps * (1 + np.abs(weights) @ np.abs(self.slack_matrix))
        if (not np.isfinite(coefficients).all() or not np.isfinite(rhs)
                or rhs >= -self.eps or np.any(coefficients < -errors)
                or not np.allclose(coefficients, self.T[row, :-1],
                                   atol=10 * self.sol_eps, rtol=10 * self.sol_eps)
                or not np.isclose(rhs, self.T[row, -1],
                                  atol=10 * self.sol_eps, rtol=10 * self.sol_eps)):
            raise LPError("NUMERICAL_ERROR", "The infeasibility row failed independent reconstruction")
        return {
            "type": "CONTRADICTORY_NONNEGATIVE_ROW",
            "row": int(row),
            "weights": weights,
            "combined_coefficients": coefficients,
            "combined_rhs": rhs,
            "variable_names": [v["name"] for v in self.variable_info],
            "expanded_rows": self.slack_row_sources,
            "explanation": "Nonnegative variables cannot sum to the negative RHS of this row.",
        }


    def initialize_dual_basis(self, initial_basis, phase):
        self.build_slack_tableau(initial_basis)
        self.phase = phase
        self.set_objective(self.phase_two_cost, self.constant_work)
        primal, dual = self.feasibility_flags()
        self.initial_feasibility = {"primal": primal, "dual": dual}
        return primal, dual

    def run_algorithm(self, initial_basis=None):
        # Pseudocode: build basis -> require dual feasibility -> dual ratio pivots.
        _, dual = self.initialize_dual_basis(initial_basis, "DUAL_START")
        if not dual:
            raise LPError("INVALID_START", "Strict dual simplex requires a dual-feasible basis; use a valid basis or GeneralizedSimplex")
        status = self.iterate(self.choose_pivot, "DUAL")
        certificate = (self.build_infeasibility_certificate(self.last_action.row)
                       if status == "INFEASIBLE" else None)
        return status, certificate


class GeneralizedSimplex(DualSimplex):
    """Taha's feasibility-first method; reuse dual repair and primal pivots."""

    def check_iteration_invariant(self):
        # Feasibility-only repair requires neither primal nor real-objective dual feasibility.
        if self.phase == "GENERALIZED_DUAL":
            DualSimplex.check_iteration_invariant(self)
        elif self.phase == "GENERALIZED_PRIMAL":
            if not self.feasibility_flags()[0]:
                raise LPError("NUMERICAL_ERROR", "Generalized primal stage lost primal feasibility")
        elif self.phase == "GENERALIZED_FEASIBILITY":
            return None

    def choose_pivot(self):
        if self.phase == "GENERALIZED_DUAL":
            return self.choose_dual_pivot(feasibility_only=False)
        if self.phase == "GENERALIZED_FEASIBILITY":
            return self.choose_dual_pivot(feasibility_only=True)
        if self.phase == "GENERALIZED_PRIMAL":
            return self.choose_primal_pivot()
        raise LPError("INVALID_START", "Generalized simplex has no active iteration stage")

    def run_algorithm(self, initial_basis=None):
        # Pseudocode: dual if feasible; otherwise repair RHS, then primal on the same basis.
        # This generalized path adds no artificial variables.
        primal, dual = self.initialize_dual_basis(initial_basis, "GENERALIZED_START")
        if dual:
            status = self.iterate(self.choose_pivot, "GENERALIZED_DUAL")
        else:
            status = ("FEASIBLE" if primal else
                      self.iterate(self.choose_pivot, "GENERALIZED_FEASIBILITY"))
            if status == "FEASIBLE":
                self.phase = "GENERALIZED_PRIMAL"
                self.set_objective(self.phase_two_cost, self.constant_work)
                status = self.iterate(self.choose_pivot, "GENERALIZED_PRIMAL")
        certificate = None
        if status == "INFEASIBLE":
            certificate = self.build_infeasibility_certificate(self.last_action.row)
        elif status == "UNBOUNDED":
            certificate = self.build_unbounded_ray(self.last_action.column)
        return status, certificate


if __name__ == "__main__":
    # Taha, Section 4.4.2.
    problem = LPProblem(
        sense="max",
        A=[[-1, 2, -2], [-1, 1, 1], [2, -1, 4]],
        b=[8, 4, 10],
        c=[0, 0, 2],
        signs=[">=", "<=", "<="],
    )
    solver = GeneralizedSimplex(problem)
    result = solver.solve()
    print("Algorithm:", result["algorithm"])
    print("Initial feasibility:", result["initial_feasibility"])
    print("Status:", result["status"])
    if result["status"] == "OPTIMAL":
        print("Solution:", result["x"])
        print("Objective:", result["objective"])
        print("Final degenerate basis:", result["final_degenerate"])
        print("Alternative optima:", result["alternative_info"])
    else:
        print("Message:", result["message"])
