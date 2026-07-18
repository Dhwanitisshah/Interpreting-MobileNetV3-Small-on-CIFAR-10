"""Shared plain-text table formatting for the scripts/report_*.py reporting layer."""

from typing import IO, List, Sequence, Tuple


def write_table(out: IO[str], cols: Sequence[Tuple[str, int, str]], rows: Sequence[Sequence[str]]) -> None:
    """Write a fixed-width plain-text table.

    `cols` is a sequence of (header, width, align) where align is "<" or ">".
    `rows` is a sequence of already-formatted string cells, one tuple per row.
    """
    header = "  ".join(f"{name:{align}{width}}" for name, width, align in cols)
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    for cells in rows:
        out.write("  ".join(f"{cell:{align}{width}}" for cell, (_, width, align) in zip(cells, cols)) + "\n")
