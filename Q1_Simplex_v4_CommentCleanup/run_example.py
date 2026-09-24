"""Examples and a command-line entry point.

PSEUDOCODE: select input -> select algorithm -> solve -> show status/history
-> optionally export to Excel. Change problem data here, not inside the engine.
"""
import argparse
from pathlib import Path
import numpy as np
from simplex0 import LPProblem, ClassicalSimplex, DualSimplex, GeneralizedSimplex


def examples():
    return {
        "simple": LPProblem("min", [[1, 1], [1, 2]], [4, 6], [2, 3], [">=", ">="]),
        "dual": LPProblem("min", [[3, 1, 1], [-3, 3, 1], [1, 1, 1]],
                          [3, 6, 3], [3, 2, 1], [">=", ">=", "<="]),
        "generalized": LPProblem("max", [[-1, 2, -2], [-1, 1, 1], [2, -1, 4]],
                                 [8, 4, 10], [0, 0, 2], [">=", "<=", "<="]),
    }


def print_result(solver, result, show_steps=False):
    print("\nAlgorithm:", result["algorithm"])
    print("Status:", result["status"])
    print("Initial primal/dual feasibility:", result["initial_feasibility"])
    print("Main pivot count:", result["iterations"])
    if result["status"] == "OPTIMAL":
        print("Original solution:", np.array2string(result["x"], precision=12, suppress_small=False))
        print("Original objective:", format(result["objective"], ".12g"))
        print("Numerical checks:", result["numerical_report"])
        print("Displayed digits are formatted; returned values are unchanged.")
        print("Final degenerate basis:", result["final_degenerate"])
        print("Degeneracy by phase:", result["degeneracy_by_phase"])
        print("Alternative optimal solutions:", result["alternative_info"])
    else:
        print("Message:", result["message"])
        if result["certificate"] is not None:
            print("Certificate:", result["certificate"])
    if show_steps:
        print("\nRecorded tableau steps (RHS last; objective row last):")
        for h in solver.history:
            print(f"\n{h['phase']} / {h['event']} / iteration {h['iteration']}")
            print("Basis IDs:", h["basis"], "primal:", h["primal_feasible"],
                  "dual:", h["dual_feasible"])
            if h["event"] == "PIVOT":
                print("Entering ID:", h["entering"], "Leaving ID:", h["leaving"],
                      "Pivot:", h["pivot"], "Ratio/repair step:", h["theta"])
            print(np.array2string(h["T"], precision=12, suppress_small=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["classical", "dual", "generalized", "all"],
                        default="generalized")
    parser.add_argument("--example", choices=list(examples()), default="generalized")
    parser.add_argument("--input", type=Path, help="Read a structured Excel input file")
    parser.add_argument("--output", type=Path, help="Write result and iteration history to Excel")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing an existing output")
    parser.add_argument("--show-steps", action="store_true")
    parser.add_argument("--skip-alternatives", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="Pivot/sign decision tolerance; not decimal places")
    parser.add_argument("--solution-tolerance", type=float, default=1e-8,
                        help="Residual tolerance; not decimal places")
    args = parser.parse_args()
    if args.output and args.method == "all":
        parser.error("Choose one method when exporting a single output workbook")
    if args.output and args.output.exists() and not args.overwrite:
        parser.error("Output already exists; use another path or explicitly pass --overwrite")
    problem = LPProblem.from_excel(args.input) if args.input else examples()[args.example]
    algorithms = {"classical": ClassicalSimplex, "dual": DualSimplex, "generalized": GeneralizedSimplex}
    selected = algorithms.values() if args.method == "all" else [algorithms[args.method]]
    for method in selected:
        solver = method(problem, tolerance=args.tolerance,
                        solution_tolerance=args.solution_tolerance)
        result = solver.solve(check_alternatives=not args.skip_alternatives)
        print_result(solver, result, args.show_steps)
        if args.output:
            solver.write_result_to_excel(args.output)
            print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
