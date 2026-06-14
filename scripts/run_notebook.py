"""Execute energy_load_forecasting.ipynb end-to-end."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    notebook_path = project_root / "notebooks" / "energy_load_forecasting.ipynb"
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}", file=sys.stderr)
        return 1

    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(
        timeout=7200,
        kernel_name="python3",
        allow_errors=False,
    )
    ep.preprocess(nb, {"metadata": {"path": str(project_root / "notebooks")}})

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook executed successfully: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
