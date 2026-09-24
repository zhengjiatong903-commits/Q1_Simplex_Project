"""Read the bundled template and test export for all three algorithms.

PSEUDOCODE: load structured original LP -> solve -> write to a temporary file
-> check required sheets and key results. Temporary output files are removed.
"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from simplex0 import LPProblem, ClassicalSimplex, DualSimplex, GeneralizedSimplex


def main():
    root = Path(__file__).resolve().parent
    p = LPProblem.from_excel(root / "input_example.xlsx")
    for method in (ClassicalSimplex, GeneralizedSimplex):
        s = method(p)
        r = s.solve(check_alternatives=False)
        assert r["status"] == "OPTIMAL", r
        assert np.isclose(r["objective"], 28 / 9), r
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xlsx"
            s.write_result_to_excel(path)
            with pd.ExcelFile(path) as book:
                assert set(book.sheet_names) == {
                    "Summary", "OriginalProblem", "Transformations", "Iterations"}
            summary = pd.read_excel(path, sheet_name="Summary")
            data = dict(zip(summary["key"], summary["value"]))
            assert data["status"] == "OPTIMAL"
            assert data["algorithm"] == method.__name__
        print("PASS Excel input/export:", method.__name__)
    # Exercise Dual export using a compatible problem instead of silently
    # applying another algorithm to an incompatible starting basis.
    p = LPProblem("min", [[1]], [2], [1], [">="])
    s = DualSimplex(p)
    r = s.solve(check_alternatives=False)
    assert r["status"] == "OPTIMAL" and np.isclose(r["objective"], 2)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dual.xlsx"
        s.write_result_to_excel(path)
        assert path.exists()
        history = pd.read_excel(path, sheet_name="Iterations")
        assert "DUAL" in set(history["phase"])
    print("PASS Excel export: DualSimplex")
    print("Completed 3 Excel workflow checks.")


if __name__ == "__main__":
    main()
