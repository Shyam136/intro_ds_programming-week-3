"""
Week 3 utilities
- Recursive Fibonacci (memoized)  -> function name: `fibonacci` (alias: fib)
- Recursive integer -> binary     -> function name: `to_binary`
- Bellevue Almshouse tasks        -> task_1..task_4
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Exercise 1 — Fibonacci (recursive, memoized)
# ---------------------------------------------------------------------
@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-based) using recursion."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fib(n: int) -> int:
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
# ---------------------------------------------------------------------
_URL = (
    "https://github.com/melaniewalsh/Intro-Cultural-Analytics/raw/master/"
    "book/data/bellevue_almshouse_modified.csv"
)
_FALLBACK_PATHS: List[str] = [
    "../data/bellevue_almshouse_modified.csv",
    "./data/bellevue_almshouse_modified.csv",
    "data/bellevue_almshouse_modified.csv",
    "./bellevue_almshouse_modified.csv",
    "../bellevue_almshouse_modified.csv",
    "../data/bellevue_almshouse.csv",
    "./data/bellevue_almshouse.csv",
    "data/bellevue_almshouse.csv",
    "./bellevue_almshouse.csv",
    "../bellevue_almshouse.csv",
]


def _load_bellevue() -> pd.DataFrame:
    """Load and lightly clean the Bellevue Almshouse dataset.

    Primary: load from the course GitHub URL (as in the lab).
    Fallback: try a short list of common local paths.
    """
    df: Optional[pd.DataFrame] = None

    # Try URL first
    try:
        df = pd.read_csv(_URL)
    except Exception:
        df = None

    # Try local fallbacks
    if df is None:
        for p in _FALLBACK_PATHS:
            try:
                df = pd.read_csv(p)
                break
            except Exception:
                continue

    if df is None:
        raise FileNotFoundError(
            "Bellevue dataset CSV not found or reachable. "
            "Tried the URL then paths: " + ", ".join(_FALLBACK_PATHS)
        )

    # Light cleaning for consistent behavior
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan})

    if "gender" in df.columns:
        df["gender"] = df["gender"].str.lower().replace({"": np.nan})

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if "date_in" in df.columns:
        df["date_in"] = pd.to_datetime(df["date_in"], errors="coerce")

    return df


# -------------------------
# Task 1
# -------------------------
def task_1() -> list[str]:
    """Return column names sorted from least to most missing values.
    Ties are broken alphabetically by column name.
    """
    df = _load_bellevue()
    na_counts = df.isna().sum()

    # Robust: make a frame with explicit column names, then sort
    order = (
        na_counts
        .to_frame(name="na")          # explicit name for counts
        .reset_index()                # 'index' holds the column names
        .rename(columns={"index": "column"})
        .sort_values(by=["na", "column"], ascending=[True, True])
        ["column"]
        .tolist()
    )
    return order


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
    """Return a Series with index=gender and values=average age."""
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
    """Return a list of the 5 most common professions (most frequent first)."""
    df = _load_bellevue()
    if "profession" not in df.columns:
        raise KeyError("Expected 'profession' column.")
    return df["profession"].value_counts(dropna=True).head(5).index.tolist()