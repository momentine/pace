from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bench.openrouter import OpenRouterClient, calculate_cost
from bench.registry import load_all
from bench.scoring import score_checks
from bench.axe_runner import run_axe_on_fragment


RESULT_KEY_COLS = ["model_id", "condition_id", "variant_id", "component_id", "rep_idx"]
PER_CHECK_BASE_COLS = [
    "model_id",
    "model_name",
    "condition_id",
    "condition_name",
    "variant_id",
    "variant_label",
    "component_id",
    "component_title",
    "rep_idx",
    "check_id",
    "score",
    "rationale",
    "output_file",
]


def safe_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def is_nan_like(v: Any) -> bool:
    try:
        return pd.isna(v)
    except Exception:
        return False


def parse_component_meta(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, float) and pd.isna(value):
        return {}
    if not isinstance(value, str):
        return {}

    s = value.strip()
    if not s:
        return {}

    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        pass

    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def load_html(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def relative_to_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return path.name


def _save_html_exact(
    outputs_root: Path,
    model_id: str,
    condition_id: str,
    variant_id: str,
    component_id: str,
    rep_idx: int,
    html: str,
) -> Path:
    safe_model = model_id.replace("/", "_").replace(":", "_")
    d = outputs_root / safe_model / condition_id / variant_id / component_id / f"rep_{int(rep_idx):02d}"
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"{uuid.uuid4().hex[:10]}.html"
    fn.write_text(html, encoding="utf-8")
    return fn


def find_existing_output(row: pd.Series, outputs_root: Path) -> Optional[Path]:
    for col in ["output_file", "output_path", "html_path", "response_path", "file_path"]:
        if col not in row.index:
            continue
        raw = safe_str(row[col])
        if not raw:
            continue
        p = Path(raw)
        cands = []
        if p.is_absolute():
            cands.append(p)
        else:
            cands.append(outputs_root / p)
            cands.append(Path.cwd() / p)
            cands.append(outputs_root / p.name)
        for cand in cands:
            if cand.exists() and cand.is_file():
                return cand.resolve()

    model_id = safe_str(row.get("model_id"))
    condition_id = safe_str(row.get("condition_id"))
    variant_id = safe_str(row.get("variant_id"))
    component_id = safe_str(row.get("component_id"))
    rep_idx = safe_str(row.get("rep_idx"))

    if model_id and condition_id and variant_id and component_id and rep_idx:
        safe_model = model_id.replace("/", "_").replace(":", "_")
        rep_dir = outputs_root / safe_model / condition_id / variant_id / component_id / f"rep_{int(float(rep_idx)):02d}"
        if rep_dir.exists():
            htmls = sorted(rep_dir.glob("*.html"))
            if htmls:
                return htmls[-1].resolve()

    return None


def find_existing_axe_json(output_path: Optional[Path]) -> Optional[Path]:
    if output_path is None:
        return None

    same_stem = output_path.with_suffix(".axe.json")
    if same_stem.exists():
        return same_stem

    matches = sorted(output_path.parent.glob("*.axe.json"))
    if matches:
        return matches[-1]

    return None


def read_existing_axe_json(axe_json_path: Optional[Path], outputs_root: Path) -> Dict[str, Any]:
    blank = {
        "axe_ok": False,
        "axe_error": "",
        "axe_profile": "form_fragment_rules",
        "axe_violations": math.nan,
        "axe_incomplete": math.nan,
        "axe_passes": math.nan,
        "axe_inapplicable": math.nan,
        "axe_score": math.nan,
        "axe_json_file": "",
    }

    if axe_json_path is None or not axe_json_path.exists():
        blank["axe_error"] = "No existing .axe.json found."
        return blank

    try:
        with axe_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        summary = payload.get("summary", {}) or {}
        counts = summary.get("counts", {}) or {}
        meta = payload.get("meta", {}) or {}

        return {
            "axe_ok": True,
            "axe_error": "",
            "axe_profile": safe_str(meta.get("axe_profile")) or "form_fragment_rules",
            "axe_violations": int(counts.get("violations", 0) or 0),
            "axe_incomplete": int(counts.get("incomplete", 0) or 0),
            "axe_passes": int(counts.get("passes", 0) or 0),
            "axe_inapplicable": int(counts.get("inapplicable", 0) or 0),
            "axe_score": float(summary.get("pass_rate_strict", 0.0) or 0.0),
            "axe_json_file": relative_to_or_name(axe_json_path, outputs_root),
        }
    except Exception as e:
        blank["axe_error"] = f"Failed reading .axe.json: {e}"
        return blank


def needs_backfill(row: pd.Series) -> bool:
    if safe_str(row.get("ok")).lower() in {"false", "0", ""}:
        return True
    if not safe_str(row.get("output_file")):
        return True
    for col in ["raw_score", "max_score", "norm_score", "axe_score"]:
        if col in row.index and is_nan_like(row[col]):
            return True
    return False


def build_component_check_map(score_spec: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        str(comp["id"]).strip(): list(comp["checks"])
        for comp in score_spec.get("components", [])
    }


def load_score_spec(score_json_path: Path) -> Dict[str, Any]:
    with score_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_key_from_series(row: pd.Series) -> Tuple[str, str, str, str, int]:
    return (
        safe_str(row.get("model_id")),
        safe_str(row.get("condition_id")),
        safe_str(row.get("variant_id")),
        safe_str(row.get("component_id")),
        int(float(row.get("rep_idx"))),
    )


def row_key_mask(df: pd.DataFrame, key: Tuple[str, str, str, str, int]) -> pd.Series:
    model_id, condition_id, variant_id, component_id, rep_idx = key
    return (
        df["model_id"].astype(str).str.strip().eq(model_id)
        & df["condition_id"].astype(str).str.strip().eq(condition_id)
        & df["variant_id"].astype(str).str.strip().eq(variant_id)
        & df["component_id"].astype(str).str.strip().eq(component_id)
        & pd.to_numeric(df["rep_idx"], errors="coerce").fillna(-1).astype(int).eq(int(rep_idx))
    )


def ensure_backup(path: Path) -> Path:
    backup = path.with_name(path.stem + ".pre_backfill.bak" + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def save_dataframes(results_df: pd.DataFrame, per_check_df: pd.DataFrame, results_csv: Path, per_check_csv: Path) -> None:
    results_df.to_csv(results_csv, index=False)
    per_check_df.to_csv(per_check_csv, index=False)


def run_backfill(
    *,
    results_csv: Path,
    per_check_csv: Path,
    outputs_root: Path,
    score_json: Path,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rate_limit: float = 0.0,
    axe_timeout_ms: int = 10000,
) -> None:
    score_spec = load_score_spec(score_json)
    component_check_map = build_component_check_map(score_spec)

    reg = load_all()
    client = OpenRouterClient()

    results_df = pd.read_csv(results_csv)
    per_check_df = pd.read_csv(per_check_csv) if per_check_csv.exists() else pd.DataFrame(columns=PER_CHECK_BASE_COLS)

    ensure_backup(results_csv)
    ensure_backup(per_check_csv) if per_check_csv.exists() else None

    total_rows = len(results_df)

    for idx in range(total_rows):
        row = results_df.iloc[idx].copy()

        if not needs_backfill(row):
            continue

        key = row_key_from_series(row)

        model_id = safe_str(row.get("model_id"))
        model_name = safe_str(row.get("model_name"))
        condition_id = safe_str(row.get("condition_id"))
        condition_name = safe_str(row.get("condition_name"))
        variant_id = safe_str(row.get("variant_id"))
        variant_label = safe_str(row.get("variant_label"))
        component_id = safe_str(row.get("component_id"))
        component_title = safe_str(row.get("component_title"))
        rep_idx = int(float(row.get("rep_idx")))

        prompt = safe_str(row.get("prompt"))
        component_meta = parse_component_meta(row.get("component_meta"))
        checks = component_check_map.get(component_id, [])

        output_path = find_existing_output(row, outputs_root)

        gen_ok = True
        gen_error = ""
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        html = ""

        if output_path is not None:
            html = load_html(output_path)
        else:
            system_prompt = ""
            for c in reg.prompt_conditions.get("prompt_conditions", []):
                if safe_str(c.get("condition_id")) == condition_id:
                    system_prompt = safe_str(c.get("system_prompt"))
                    break

            if not system_prompt:
                gen_ok = False
                gen_error = "Missing system prompt during backfill."
            elif not prompt:
                gen_ok = False
                gen_error = "Missing saved user prompt during backfill."
            else:
                try:
                    gen = client.generate(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                    )
                    gen_ok = bool(getattr(gen, "ok", False))
                    gen_error = safe_str(getattr(gen, "error", ""))
                    html = safe_str(getattr(gen, "content", ""))
                    prompt_tokens = int(getattr(gen, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(gen, "completion_tokens", 0) or 0)

                    if gen_ok:
                        for m in reg.models.get("data", []):
                            if safe_str(m.get("id")) == model_id:
                                cost = float(calculate_cost(m, prompt_tokens, completion_tokens))
                                break

                        output_path = _save_html_exact(
                            outputs_root=outputs_root,
                            model_id=model_id,
                            condition_id=condition_id,
                            variant_id=variant_id,
                            component_id=component_id,
                            rep_idx=rep_idx,
                            html=html,
                        )
                except Exception as e:
                    gen_ok = False
                    gen_error = f"Generation failed: {e}"

        raw_score = math.nan
        max_score = math.nan
        norm_score = math.nan

        axe_info = {
            "axe_ok": math.nan,
            "axe_error": "",
            "axe_profile": "form_fragment_rules",
            "axe_violations": math.nan,
            "axe_incomplete": math.nan,
            "axe_passes": math.nan,
            "axe_inapplicable": math.nan,
            "axe_score": math.nan,
            "axe_json_file": "",
        }

        new_per_rows: List[Dict[str, Any]] = []

        if gen_ok and output_path is not None:
            raw, mx, per_scores = score_checks(
                html=html,
                component_checks=checks,
                score_spec=score_spec,
                component_meta=component_meta,
            )
            raw_score = float(raw)
            max_score = float(mx)
            norm_score = float((raw / mx) if mx > 0 else 0.0)

            try:
                axe_res = run_axe_on_fragment(
                    html=html,
                    output_file=str(output_path),
                    meta={
                        "model_id": model_id,
                        "condition_id": condition_id,
                        "variant_id": variant_id,
                        "component_id": component_id,
                        "rep_idx": rep_idx,
                        "axe_profile": "form_fragment_rules",
                    },
                    timeout_ms=int(axe_timeout_ms),
                )

                axe_info = {
                    "axe_ok": bool(axe_res.get("ok", False)),
                    "axe_error": safe_str(axe_res.get("error", "")),
                    "axe_profile": "form_fragment_rules",
                    "axe_violations": int((axe_res.get("counts") or {}).get("violations", 0) or 0),
                    "axe_incomplete": int((axe_res.get("counts") or {}).get("incomplete", 0) or 0),
                    "axe_passes": int((axe_res.get("counts") or {}).get("passes", 0) or 0),
                    "axe_inapplicable": int((axe_res.get("counts") or {}).get("inapplicable", 0) or 0),
                    "axe_score": float(axe_res.get("axe_score", 0.0) or 0.0),
                    "axe_json_file": relative_to_or_name(Path(axe_res.get("json_path", "")), outputs_root)
                    if safe_str(axe_res.get("json_path", "")) else "",
                }
            except Exception as e:
                axe_json_path = find_existing_axe_json(output_path)
                axe_info = read_existing_axe_json(axe_json_path, outputs_root)
                if not axe_info["axe_ok"]:
                    axe_info["axe_error"] = f"axe rerun failed: {e}"

            for pcs in per_scores:
                new_per_rows.append({
                    "model_id": model_id,
                    "model_name": model_name,
                    "condition_id": condition_id,
                    "condition_name": condition_name,
                    "variant_id": variant_id,
                    "variant_label": variant_label,
                    "component_id": component_id,
                    "component_title": component_title,
                    "rep_idx": rep_idx,
                    "check_id": pcs.check_id,
                    "score": pcs.score,
                    "rationale": pcs.rationale,
                    "output_file": relative_to_or_name(output_path, outputs_root),
                })

        results_df.loc[idx, "model_id"] = model_id
        results_df.loc[idx, "model_name"] = model_name
        results_df.loc[idx, "condition_id"] = condition_id
        results_df.loc[idx, "condition_name"] = condition_name
        results_df.loc[idx, "variant_id"] = variant_id
        results_df.loc[idx, "variant_label"] = variant_label
        results_df.loc[idx, "component_id"] = component_id
        results_df.loc[idx, "component_title"] = component_title
        results_df.loc[idx, "rep_idx"] = rep_idx
        results_df.loc[idx, "prompt"] = prompt
        results_df.loc[idx, "ok"] = bool(gen_ok)
        results_df.loc[idx, "error"] = gen_error
        results_df.loc[idx, "prompt_tokens"] = prompt_tokens
        results_df.loc[idx, "completion_tokens"] = completion_tokens
        results_df.loc[idx, "cost"] = cost
        results_df.loc[idx, "output_file"] = relative_to_or_name(output_path, outputs_root) if output_path is not None else ""
        results_df.loc[idx, "raw_score"] = raw_score
        results_df.loc[idx, "max_score"] = max_score
        results_df.loc[idx, "norm_score"] = norm_score

        for k, v in axe_info.items():
            if k not in results_df.columns:
                results_df[k] = pd.NA
            results_df.loc[idx, k] = v

        if not per_check_df.empty:
            mask = row_key_mask(per_check_df, key)
            per_check_df = per_check_df.loc[~mask].copy()

        if new_per_rows:
            insert_df = pd.DataFrame(new_per_rows)
            per_check_df = pd.concat([per_check_df, insert_df], ignore_index=True)

        if not per_check_df.empty:
            for col in RESULT_KEY_COLS + ["check_id"]:
                if col not in per_check_df.columns:
                    per_check_df[col] = pd.NA
            per_check_df["_rep_sort"] = pd.to_numeric(per_check_df["rep_idx"], errors="coerce").fillna(-1).astype(int)
            per_check_df = per_check_df.sort_values(
                by=["model_id", "condition_id", "variant_id", "component_id", "_rep_sort", "check_id"],
                kind="stable",
            ).drop(columns=["_rep_sort"])

        save_dataframes(results_df, per_check_df, results_csv, per_check_csv)

        if rate_limit and rate_limit > 0:
            time.sleep(float(rate_limit))

        print(f"[DONE] row {idx+1}/{total_rows} | {model_id} {condition_id} {variant_id} {component_id} rep_{rep_idx:02d}")

    print(f"[DONE] Updated {results_csv}")
    print(f"[DONE] Updated {per_check_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch results.csv and per_check.csv in place for failed/NaN rows.")
    parser.add_argument("--results_csv", type=Path, required=True)
    parser.add_argument("--per_check_csv", type=Path, required=True)
    parser.add_argument("--outputs_root", type=Path, required=True)
    parser.add_argument("--score_json", type=Path, required=True)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--rate_limit", type=float, default=0.0)
    parser.add_argument("--axe_timeout_ms", type=int, default=10000)
    args = parser.parse_args()

    run_backfill(
        results_csv=args.results_csv,
        per_check_csv=args.per_check_csv,
        outputs_root=args.outputs_root,
        score_json=args.score_json,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rate_limit=args.rate_limit,
        axe_timeout_ms=args.axe_timeout_ms,
    )


if __name__ == "__main__":
    main()