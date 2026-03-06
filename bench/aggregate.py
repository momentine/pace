# bench/aggregate.py
from __future__ import annotations
import pandas as pd
import numpy as np

def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    results_df expected columns:
      model_id, condition_id, variant_id, component_id, rep_idx,
      raw_score, max_score, norm_score, cost
    """
    gcols = ["model_id", "condition_id", "variant_id", "component_id"]
    agg = results_df.groupby(gcols).agg(
        mean_norm=("norm_score", "mean"),
        sd_norm=("norm_score", "std"),
        mean_raw=("raw_score", "mean"),
        sd_raw=("raw_score", "std"),
        mean_cost=("cost", "mean"),
        total_cost=("cost", "sum"),
        n=("norm_score", "count"),
    ).reset_index()
    agg["sd_norm"] = agg["sd_norm"].fillna(0.0)
    agg["sd_raw"] = agg["sd_raw"].fillna(0.0)
    return agg

def per_check_summary(per_check_df: pd.DataFrame) -> pd.DataFrame:
    """
    per_check_df columns:
      model_id, condition_id, variant_id, component_id, rep_idx, check_id, score
    """
    gcols = ["model_id","condition_id","variant_id","component_id","check_id"]
    agg = per_check_df.groupby(gcols).agg(
        mean=("score","mean"),
        sd=("score","std"),
        n=("score","count"),
    ).reset_index()
    agg["sd"] = agg["sd"].fillna(0.0)
    return agg
