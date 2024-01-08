__all__ = ["CsvWriter"]

import csv
from typing import Any, Sequence, Mapping


class CsvWriter(csv.DictWriter):
    """Create a CSV writer that maps dictionaries onto output rows and columns."""

    def writecols(self, cols: Mapping[str, Sequence[Any]]):
        """Write all elements in columns to the writer's file object."""
        colnames = set(self.fieldnames)
        if not colnames.issubset(cols.keys()):
            raise ValueError("Missing a column")
        if self.extrasaction == "raise" and not colnames.issuperset(cols.keys()):
            raise ValueError("Extra column found")
        key1, *keys = self.fieldnames
        if not all(len(cols[key]) == len(cols[key1]) for key in keys):
            raise ValueError("Columns of varying length")
        rows = (
            tuple(cols[key][i] for key in self.fieldnames)
            for i in range(len(cols[key1]))
        )
        self.writer.writerows(rows)
