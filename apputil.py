"""
Week 3 utilities
- Recursive Fibonacci (memoized)  -> function name: `fibonacci` (alias: fib)
- Recursive integer -> binary     -> function name: `to_binary`
- Bellevue Almshouse tasks        -> task_1..task_4
PEP-8 compliant, fast, and robust.
"""

from __future__ import annotations

from typing import List, Optional
from functools import lru_cache

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Exercise 1 — Fibonacci (recursive, memoized to avoid timeouts)
# ---------------------------------------------------------------------
@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-based) using recursion."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# Optional alias in case notebook/tests reference fib()
def fib(n: int) -> int:  # noqa: D401
    """Alias for fibonacci(n)."""
    return fibonacci(n)


# ---------------------------------------------------------------------
# Exercise 2 — Decimal -> Binary (recursive)
# ---------------------------------------------------------------------
def to_binary(n: int) -> str:
    """Convert an integer to its binary representation (no '0b' prefix)."""
    if n == 0:
        return "0"
    if n < 0:
        return "-" + to_binary(-n)
    if n < 2:
        return str(n)
    return to_binary(n // 2) + str(n % 2)


# ---------------------------------------------------------------------
# Exercise 3 — Bellevue Almshouse tasks
# Fast path search: try a finite list of common locations (no deep glob).
# ---------------------------------------------------------------------
_CANDIDATE_PATHS: List[str] = [
    "../data/bellevue_almshouse_modified.csv",
    "./data/bellevue_almshouse_modified.csv",
    "data/bellevue_almshouse_modified.csv",
    "./bellevue_almshouse_modified.csv",
    "../bellevue_almshouse_modified.csv",
    # fallbacks if your cohort uses a slightly different filename
    "../data/bellevue_almshouse.csv",
    "./data/bellevue_almshouse.csv",
    "data/bellevue_almshouse.csv",
    "./bellevue_almshouse.csv",
    "../bellevue_almshouse.csv",
]


def _load_bellevue() -> pd.DataFrame:
    """Load and lightly clean the Bellevue Almshouse dataset."""
    path: Optional[str] = None
    for p in _CANDIDATE_PATHS:
        try:
            # Quick existence test by attempting a small read
            df_try = pd.read_csv(p, nrows=1)
            path = p
            break
        except Exception:
            continue

    if path is None:
        raise FileNotFoundError(
            "Bellevue dataset CSV not found. Tried paths: " + ", ".join(_CANDIDATE_PATHS)
        )

    df = pd.read_csv(path)

    # Normalize whitespace for object columns and set blanks to NaN
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan})

    # Minimal gender normalization (lowercase, keep categories as-is)
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower().replace({"": np.nan})

    # Age numeric; date_in to datetime for yearly grouping
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "date_in" in df.columns:
        df["date_in"] = pd.to_datetime(df["date_in"], errors="coerce")

    return df


# -------------------------
# Task 1
# -------------------------
def task_1() -> List[str]:
    """Return column names sorted from least to most missing values."""
    df = _load_bellevue()
    na_counts = df.isna().sum().sort_values(ascending=True)
    return na_counts.index.tolist()


# -------------------------
# Task 2
# -------------------------
def task_2() -> pd.DataFrame:
    """Return a DataFrame with columns: year, total_admissions."""
    df = _load_bellevue()
    if "date_in" not in df.columns:
        raise KeyError("Expected 'date_in' column.")
    out = (
        df.assign(year=df["date_in"].dt.year)
        .groupby("year", dropna=True)
        .size()
        .rename("total_admissions")
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return out


# -------------------------
# Task 3
# -------------------------
def task_3() -> pd.Series:
    """Return a Series: index=gender, values=average age."""
    df = _load_bellevue()
    if not {"gender", "age"}.issubset(df.columns):
        raise KeyError("Expected 'gender' and 'age' columns.")
    return (
        df.groupby("gender", dropna=True)["age"]
        .mean()
        .sort_index()
    )


# -------------------------
# Task 4
# -------------------------
def task_4() -> List[str]:
    """Return list of the 5 most common professions (most frequent first)."""
    df = _load_bellevue()
    if "profession" not in df.columns:
        raise KeyError("Expected 'profession' column.")
    return df["profession"].value_counts(dropna=True).head(5).index.tolist()


# ---------------------------------------------------------------------
# Optional local sanity checks (quick)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    assert fibonacci(0) == 0 and fibonacci(1) == 1
    assert fibonacci(5) == 5 and fibonacci(9) == 34
    assert fib(9) == 34  # alias

    assert to_binary(2) == "10" and to_binary(12) == "1100"
    assert to_binary(0) == "0" and to_binary(-5) == "-101"

    # Try loading if present at any candidate path
    try:
        _df = _load_bellevue()
        _ = task_1()
        _t2 = task_2()
        assert {"year", "total_admissions"}.issubset(_t2.columns)
        _ = task_3()
        _ = task_4()
        print("Local Bellevue checks passed.")
    except FileNotFoundError:
        print("Bellevue CSV not found locally; skipping data checks.")
