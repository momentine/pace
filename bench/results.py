from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_run_tables(run_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    res_file = run_dir / "results.csv"
    per_file = run_dir / "per_check.csv"

    if res_file.exists():
        try:
            res_df = pd.read_csv(res_file)
        except pd.errors.EmptyDataError:
            res_df = pd.DataFrame()
    else:
        res_df = pd.DataFrame()

    if per_file.exists():
        try:
            per_df = pd.read_csv(per_file)
        except pd.errors.EmptyDataError:
            per_df = pd.DataFrame()
    else:
        per_df = None

    return res_df, per_df


def run_kpis(res_df: pd.DataFrame) -> dict:
    if res_df is None or res_df.empty:
        return {"rows": 0, "pct_ok": "0%", "mean_norm_ok": "-", "total_cost": 0.0}

    rows = int(len(res_df))
    ok = res_df[res_df["ok"] == True]
    pct_ok = (len(ok) / rows) if rows else 0.0

    mean_norm_ok = "-"
    if not ok.empty and "norm_score" in ok.columns:
        mean_norm_ok = float(np.nanmean(ok["norm_score"].astype(float)))

    total_cost = float(np.nansum(res_df["cost"].astype(float))) if "cost" in res_df.columns else 0.0

    return {
        "rows": rows,
        "pct_ok": f"{pct_ok*100:.1f}%",
        "mean_norm_ok": f"{mean_norm_ok:.3f}" if isinstance(mean_norm_ok, float) else mean_norm_ok,
        "total_cost": f"{total_cost:.4f}",
    }
