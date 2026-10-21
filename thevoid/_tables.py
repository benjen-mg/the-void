"""Optional pandas."""

from __future__ import annotations


def dataframe(rows: list, columns=None):
    """A pandas DataFrame of `rows` (dicts), or a clear ImportError."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "to_dataframe() needs pandas, which the Void does not require: "
            "pip install pandas (or use .rows / .to_csv())") from None
    return pd.DataFrame(rows, columns=columns)
