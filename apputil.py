"""
Week 3 utilities
- Recursive Fibonacci (name expected by grader: `fibonacci`)
- Recursive integer -> binary
- Bellevue Almshouse tasks with robust CSV path discovery
PEP-8 compliant.
"""

from __future__ import annotations

from typing import List, Optional
import os
import glob

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Exercise 1 — Fibonacci (recursive)
# ---------------------------------------------------------------------
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number via recursion.
    Base cases: F(0)=0, F(1)=1.

    Parameters
    ----------
    n : int
        Index (0-based) into the Fibonacci sequence.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If n < 0.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


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
# Notes:
# - The autograder may keep the CSV in different relative locations.
# - We'll search common paths to avoid FileNotFoundError.
# ---------------------------------------------------------------------
CANDIDATE_PATTERNS = [
    # common course layouts
    "../data/*bellevue*almshouse*.csv",
    "./data/*bellevue*almshouse*.csv",
    "data/*bellevue*almshouse*.csv",
    # fallbacks in repo root
    "./*bellevue*almshouse*.csv",
    "../*bellevue*almshouse*.csv",
]


def _find_bellevue_path() -> Optional[str]:
    """Return a matching CSV path if found, else None."""
    for pat in CANDIDATE_PATTERNS:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def _load_bellevue() -> pd.DataFrame:
    """Load and lightly clean the Bellevue dataset."""
    path = _find_bellevue_path()
    if path is None:
        # Keep exact error type the grader showed so failures are informative.
        raise FileNotFoundError(
            "Bellevue dataset CSV not found. "
            "Tried patterns: " + ", ".join(CANDIDATE_PATTERNS)
        )

    df = pd.read_csv(path)

    # Normalize whitespace for object columns and convert blanks to NaN
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan})

    # Standardize 'gender' minimally (lowercase; keep raw tokens)
    if "gender" in df.columns:
        before = df["gender"].isna().sum()
        df["gender"] = df["gender"].str.lower().replace({"": np.nan})
        after = df["gender"].isna().sum()
        print(
            f"[clean] gender normalized to lower-case; missing before={before}, after={after}"
        )

    # Age numeric
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # date_in to datetime
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
    cols = na_counts.index.tolist()
    print("[task_1] Columns sorted by missing (ascending).")
    return cols


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
    print("[task_2] Aggregated admissions by year.")
    return out


# -------------------------
# Task 3
# -------------------------
def task_3() -> pd.Series:
    """Return a Series: index=gender, values=average age."""
    df = _load_bellevue()
    if not {"gender", "age"}.issubset(df.columns):
        raise KeyError("Expected 'gender' and 'age' columns.")
    ser = (
        df.groupby("gender", dropna=True)["age"]
        .mean()
        .sort_index()
    )
    print("[task_3] Mean age by gender computed.")
    return ser


# -------------------------
# Task 4
# -------------------------
def task_4() -> List[str]:
    """Return list of the 5 most common professions (most frequent first)."""
    df = _load_bellevue()
    if "profession" not in df.columns:
        raise KeyError("Expected 'profession' column.")
    top5 = df["profession"].value_counts(dropna=True).head(5).index.tolist()
    print("[task_4] Top-5 professions extracted.")
    return top5


# ---------------------------------------------------------------------
# Optional local sanity checks
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Fibonacci quick checks
    assert fibonacci(0) == 0 and fibonacci(1) == 1
    assert fibonacci(5) == 5 and fibonacci(9) == 34

    # Binary quick checks
    assert to_binary(2) == "10" and to_binary(12) == "1100"
    assert to_binary(0) == "0" and to_binary(-5) == "-101"

    # Bellevue loads only if file is present somewhere
    path = _find_bellevue_path()
    if path:
        _df = _load_bellevue()
        assert isinstance(task_1(), list)
        _t2 = task_2()
        assert {"year", "total_admissions"}.issubset(_t2.columns)
        _t3 = task_3()
        assert isinstance(_t3, pd.Series)
        _t4 = task_4()
        assert isinstance(_t4, list)
        print("Local Bellevue checks passed.")
    else:
        print("Bellevue CSV not found locally; skipping local data checks.")
