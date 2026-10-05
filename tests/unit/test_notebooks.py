"""The guide notebooks: their code must still run, and print what their stored outputs show."""

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

GUIDE = Path(__file__).parents[2] / "docs" / "guide"
NOTEBOOKS = sorted(GUIDE.glob("*.ipynb"))


def printed(
    cell: dict[str, Any],  # A code cell of a notebook
) -> str:  # What the cell printed when the notebook was last run and saved
    """Read the text a cell's stored outputs show on standard output."""
    streams = [o for o in cell["outputs"] if o["output_type"] == "stream" and o["name"] == "stdout"]
    return "".join("".join(o["text"]) for o in streams)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_prints_what_its_stored_outputs_show(
    notebook: Path,  # A notebook of the guide
) -> None:
    """Run the code cells in order, as plain Python, and compare what each prints with its output.

    The docs site shows the stored outputs without running anything. This test keeps them true.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    namespace: dict[str, Any] = {}
    for cell in (c for c in cells if c["cell_type"] == "code"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec("".join(cell["source"]), namespace)  # noqa: S102 - running the notebook is the test
        assert out.getvalue() == printed(cell)


def test_there_is_a_notebook_to_check() -> None:
    """The guide has at least one notebook, so the test above is not vacuous."""
    assert NOTEBOOKS
