from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from bench.scoring import score_checks


TEXT_EXTENSIONS = {".html", ".htm", ".txt", ".md", ".json", ".xml"}


def load_score_spec(score_json_path: Path) -> Dict[str, Any]:
    with score_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_component_check_map(score_spec: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        str(comp["id"]).strip(): list(comp["checks"])
        for comp in score_spec.get("components", [])
    }


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


def read_output_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def is_probable_output_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS and not path.name.endswith(".axe.json")


def index_output_files(outputs_root: Path) -> List[Path]:
    return [p for p in outputs_root.rglob("*") if is_probable_output_file(p)]


def safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def relative_display_path(path: Path, anchor: Path) -> str:
    try:
        return str(path.resolve().relative_to(anchor.resolve()))
    except Exception:
        return path.name


def candidate_names_from_row(row: pd.Series) -> List[str]:
    vals = []
    for col in ["output_path", "output_file", "filename", "file_name", "html_path", "response_path", "file_path"]:
        if col in row.index:
            raw = safe_str(row[col])
            if raw:
                vals.append(Path(raw).name)
                vals.append(Path(raw).stem)
    return [v for v in dict.fromkeys(vals) if v]


def row_tokens(row: pd.Series) -> List[str]:
    toks = []
    for col in [
        "model_name",
        "model_id",
        "condition_id",
        "condition",
        "system_instruction",
        "instruction",
        "variant_id",
        "variant",
        "component_id",
        "component_title",
        "component_name",
        "rep",
        "repetition",
        "trial",
    ]:
        if col in row.index:
            s = safe_str(row[col])
            if s:
                toks.append(s.lower())
    return toks


def try_stored_path(row: pd.Series, outputs_root: Path) -> Optional[Path]:
    possible_cols = ["output_path", "output_file", "html_path", "response_path", "file_path"]
    for col in possible_cols:
        if col not in row.index:
            continue
        raw = safe_str(row[col])
        if not raw:
            continue

        p = Path(raw)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(outputs_root / p)
            candidates.append(Path.cwd() / p)
        candidates.append(outputs_root / p.name)

        for cand in candidates:
            if cand.exists() and cand.is_file():
                return cand.resolve()
    return None


def recover_output_path(row: pd.Series, indexed_files: List[Path], outputs_root: Path) -> Optional[Path]:
    stored = try_stored_path(row, outputs_root)
    if stored is not None:
        return stored

    names = candidate_names_from_row(row)
    if names:
        exact_name_matches = [p for p in indexed_files if p.name in names]
        if len(exact_name_matches) == 1:
            return exact_name_matches[0]

        exact_stem_matches = [p for p in indexed_files if p.stem in names]
        if len(exact_stem_matches) == 1:
            return exact_stem_matches[0]

        candidate_pool = exact_name_matches or exact_stem_matches
        if candidate_pool:
            toks = row_tokens(row)
            scored = []
            for p in candidate_pool:
                full = str(p).lower()
                score = sum(tok in full for tok in toks)
                scored.append((score, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1]

    toks = row_tokens(row)
    if toks:
        scored = []
        for p in indexed_files:
            full = str(p).lower()
            score = sum(tok in full for tok in toks)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1]

    return None


def extract_meta_from_row(row: pd.Series) -> Dict[str, Any]:
    meta = {}

    if "component_meta" in row.index:
        meta.update(parse_component_meta(row["component_meta"]))

    for key in ["test_id", "title", "label", "component", "attribute"]:
        if key in row.index:
            val = row[key]
            if not (isinstance(val, float) and pd.isna(val)) and safe_str(val):
                meta.setdefault(key, val)

    if "component_id" in row.index and safe_str(row["component_id"]):
        meta.setdefault("test_id", safe_str(row["component_id"]))

    return meta


def component_display_name(score_spec: Dict[str, Any], component_id: str) -> str:
    for comp in score_spec.get("components", []):
        if safe_str(comp.get("id")) == component_id:
            return safe_str(comp.get("name"))
    return ""


def find_existing_axe_json(output_path: Path) -> Optional[Path]:
    # same basename first
    same_stem = output_path.with_suffix(".axe.json")
    if same_stem.exists():
        return same_stem

    # any .axe.json in same folder
    matches = list(output_path.parent.glob("*.axe.json"))
    if len(matches) == 1:
        return matches[0]

    return None


def read_existing_axe_json(axe_json_path: Optional[Path]) -> Dict[str, Any]:
    blank = {
        "axe_score": 0.0,
        "axe_coverage": 0.0,
        "axe_violations": 0,
        "axe_incomplete": 0,
        "axe_passes": 0,
        "axe_inapplicable": 0,
        "axe_json_path": "",
        "axe_error": "",
        "axe_ok": False,
    }

    if axe_json_path is None or not axe_json_path.exists():
        blank["axe_error"] = "No existing .axe.json found."
        return blank

    try:
        with axe_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        summary = payload.get("summary", {})
        counts = summary.get("counts", {})

        blank.update({
            "axe_score": float(summary.get("pass_rate_strict", 0.0) or 0.0),
            "axe_coverage": float(summary.get("coverage_rules", 0.0) or 0.0),
            "axe_violations": int(counts.get("violations", 0) or 0),
            "axe_incomplete": int(counts.get("incomplete", 0) or 0),
            "axe_passes": int(counts.get("passes", 0) or 0),
            "axe_inapplicable": int(counts.get("inapplicable", 0) or 0),
            "axe_ok": True,
        })
        return blank
    except Exception as e:
        blank["axe_error"] = f"Failed reading .axe.json: {e}"
        return blank


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore saved outputs using updated schema and existing .axe.json files.")
    parser.add_argument("--old_csv", type=Path, required=True, help="Old CSV with one row per saved output.")
    parser.add_argument("--outputs_root", type=Path, required=True, help="Root folder containing saved outputs.")
    parser.add_argument("--score_json", type=Path, required=True, help="Updated score spec JSON.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for refreshed CSVs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    score_spec = load_score_spec(args.score_json)
    component_check_map = build_component_check_map(score_spec)

    df = pd.read_csv(args.old_csv)
    indexed_files = index_output_files(args.outputs_root)

    per_check_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        component_id = safe_str(row.get("component_id"))
        if not component_id:
            print(f"[WARN] Row {idx}: missing component_id, skipped.")
            continue
        if component_id not in component_check_map:
            print(f"[WARN] Row {idx}: unknown component_id={component_id}, skipped.")
            continue

        resolved_path = recover_output_path(row, indexed_files, args.outputs_root)
        if resolved_path is None:
            print(f"[WARN] Row {idx}: could not resolve output file, skipped.")
            continue

        try:
            html = read_output_text(resolved_path)
        except Exception as e:
            print(f"[WARN] Row {idx}: failed reading {resolved_path}: {e}")
            continue

        component_meta = extract_meta_from_row(row)
        checks = component_check_map[component_id]

        raw_score, max_score, per_scores = score_checks(
            html=html,
            component_checks=checks,
            score_spec=score_spec,
            component_meta=component_meta,
        )
        norm_score = raw_score / max_score if max_score > 0 else 0.0

        axe_json_path = find_existing_axe_json(resolved_path)
        axe_info = read_existing_axe_json(axe_json_path)
        if axe_json_path is not None:
            axe_info["axe_json_path"] = relative_display_path(axe_json_path, args.outputs_root)

        base = row.to_dict()
        base["resolved_output_path"] = relative_display_path(resolved_path, args.outputs_root)
        base["component_name_from_score_json"] = component_display_name(score_spec, component_id)

        summary_rows.append({
            **base,
            "raw_score": raw_score,
            "max_score": max_score,
            "norm_score": norm_score,
            **axe_info,
        })

        for pcs in per_scores:
            per_check_rows.append({
                **base,
                "resolved_output_path": relative_display_path(resolved_path, args.outputs_root),
                "check_id": pcs.check_id,
                "check_score": pcs.score,
                "rationale": pcs.rationale,
                "raw_score_total": raw_score,
                "max_score_total": max_score,
                "norm_score_total": norm_score,
                "axe_score": axe_info["axe_score"],
                "axe_coverage": axe_info["axe_coverage"],
                "axe_violations": axe_info["axe_violations"],
                "axe_incomplete": axe_info["axe_incomplete"],
                "axe_passes": axe_info["axe_passes"],
                "axe_inapplicable": axe_info["axe_inapplicable"],
                "axe_json_path": axe_info["axe_json_path"],
                "axe_ok": axe_info["axe_ok"],
                "axe_error": axe_info["axe_error"],
            })

    summary_df = pd.DataFrame(summary_rows)
    per_check_df = pd.DataFrame(per_check_rows)

    summary_path = args.out_dir / "results_rescored.csv"
    per_check_path = args.out_dir / "per_check_rescored.csv"

    summary_df.to_csv(summary_path, index=False)
    per_check_df.to_csv(per_check_path, index=False)

    print(f"[DONE] Wrote {summary_path}")
    print(f"[DONE] Wrote {per_check_path}")


if __name__ == "__main__":
    main()