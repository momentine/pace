from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from bench.io import (
    load_models,
    load_components,
    load_score,
    load_prompt_conditions,
    load_variants,
    save_models,
    save_components,
    save_score,
    save_prompt_conditions,
    save_variants,
)

# ---------- defaults ----------
DEFAULT_PROMPT_CONDITIONS = {
    "prompt_conditions": [
        {
            "condition_id": "S1",
            "name": "Baseline",
            "system_prompt": "Respond with HTML only.\nDo not include CSS, JavaScript, comments, or explanations."
        },
        {
            "condition_id": "S2",
            "name": "Strict",
            "system_prompt": "Generate HTML that matches the request exactly as written. Do not add inferred features or assumptions beyond what is described.\nRespond with HTML only.\nDo not include CSS, JavaScript, comments, or explanations."
        },
        {
            "condition_id": "S3",
            "name": "Front-end Persona",
            "system_prompt": "You are a senior front-end engineer contributing to a mature codebase. Write HTML that reflects real-world engineering standards and conventional structure that other developers would expect.\nRespond with HTML only.\nDo not include CSS, JavaScript, comments, or explanations."
        },
        {
            "condition_id": "S4",
            "name": "Accessibility Perspective",
            "system_prompt": "Generate the HTML from the perspective of someone with a disability navigating primarily with a keyboard, screen reader, or assistive technology. Structure the markup so navigation and meaning are clear through the HTML itself.\nRespond with HTML only.\nDo not include CSS, JavaScript, comments, or explanations."
        }
    ]
}

DEFAULT_VARIANTS = {
    "variants": [
        {
            "variant_id": "G1",
            "label": "Build request",
            "template": "Build a {component} for “{label}”{suffix}."
        },
        {
            "variant_id": "G2",
            "label": "Shorthand",
            "template": "{component} — “{label}”{suffix}"
        },
        {
            "variant_id": "G3",
            "label": "Intent-first",
            "template": "Users need to provide “{label}”{suffix}. Add a {component}."
        },
        {
            "variant_id": "G4",
            "label": "Implementation",
            "template": "Implement {component} for “{label}”{suffix}. Treat this like it has to pass review."
        },
        {
            "variant_id": "G5",
            "label": "Design-system",
            "template": "Add a reusable {component} for “{label}”{suffix}, consistent with a shared component library."
        },
        {
            "variant_id": "G6",
            "label": "Legacy integration",
            "template": "We’re adding {component} to an existing page. Implement it for “{label}”{suffix} with minimal assumptions."
        },
        {
            "variant_id": "G7",
            "label": "Robustness",
            "template": "Implement a {component} for “{label}”{suffix} that behaves sensibly across common edge cases."
        },
        {
            "variant_id": "G8",
            "label": "Debug / repair",
            "template": "Create a {component} for “{label}”{suffix}. Assume previous attempts often get the basics wrong."
        },
        {
            "variant_id": "G9",
            "label": "Documentation example",
            "template": "Provide an example {component} for “{label}”{suffix} as if it’s going into docs."
        },
        {
            "variant_id": "G10",
            "label": "Product copy request",
            "template": "Add a {component} so someone can enter “{label}”{suffix}."
        },
        {
            "variant_id": "G11",
            "label": "Accessibility keyword",
            "template": "Create an accessible {component} for “{label}”{suffix}."
        }
    ]
}

# minimal shells
DEFAULT_MODELS = {"data": []}
DEFAULT_COMPONENTS = {"tests": []}
DEFAULT_SCORE = {
    "scoring_scale": {"min": 0, "max": 2},
    "check_definitions": {},
    "components": [],
}


@dataclass
class Registries:
    models: Dict[str, Any]
    components: Dict[str, Any]
    score: Dict[str, Any]
    prompt_conditions: Dict[str, Any]
    variants: Dict[str, Any]


def _safe_load_or_default(loader_fn, default_obj: dict, saver_fn):
    try:
        obj = loader_fn()
        if not isinstance(obj, dict) or (not obj):
            raise ValueError("Empty/invalid JSON object.")
        return obj
    except Exception:
        saver_fn(default_obj)
        return default_obj


def ensure_defaults() -> None:
    _safe_load_or_default(load_models, DEFAULT_MODELS, save_models)
    _safe_load_or_default(load_components, DEFAULT_COMPONENTS, save_components)
    _safe_load_or_default(load_score, DEFAULT_SCORE, save_score)
    _safe_load_or_default(load_prompt_conditions, DEFAULT_PROMPT_CONDITIONS, save_prompt_conditions)
    _safe_load_or_default(load_variants, DEFAULT_VARIANTS, save_variants)


def load_all() -> Registries:
    return Registries(
        models=load_models(),
        components=load_components(),
        score=load_score(),
        prompt_conditions=_safe_load_or_default(load_prompt_conditions, DEFAULT_PROMPT_CONDITIONS, save_prompt_conditions),
        variants=_safe_load_or_default(load_variants, DEFAULT_VARIANTS, save_variants),
    )


# ---------- validation ----------
def validate_all(reg: Registries) -> List[str]:
    problems: List[str] = []

    # variants
    vids = [v.get("variant_id") for v in (reg.variants.get("variants", []) or []) if v.get("variant_id")]
    if len(set(vids)) != len(vids):
        problems.append("variants.json: duplicate variant_id values")

    # prompt conditions
    cids = [c.get("condition_id") for c in (reg.prompt_conditions.get("prompt_conditions", []) or []) if c.get("condition_id")]
    if len(set(cids)) != len(cids):
        problems.append("prompt_conditions.json: duplicate condition_id values")

    # components prompts keyed by variant_id
    tests = reg.components.get("tests", []) or []
    enabled_vids = [v.get("variant_id") for v in (reg.variants.get("variants", []) or []) if v.get("enabled", True) and v.get("variant_id")]
    for i, t in enumerate(tests):
        prompts = t.get("prompts")
        if not isinstance(prompts, dict):
            problems.append(f"components.json: tests[{i}] prompts should be a dict keyed by variant_id")
            continue
        missing = [vid for vid in enabled_vids if not (prompts.get(vid) or "").strip()]
        if missing:
            problems.append(f"components.json: '{t.get('title','(untitled)')}' missing prompt text for {missing}")

    # score mapping refers to defined checks
    defs = (reg.score.get("check_definitions", {}) or {})
    for c in (reg.score.get("components", []) or []):
        for chk in (c.get("checks", []) or []):
            if chk not in defs:
                problems.append(f"score.json: component {c.get('id','?')} references undefined check '{chk}'")

    return problems


# ---------- models editor ----------
def df_models(models_json: dict) -> pd.DataFrame:
    rows = models_json.get("data", []) or []
    out = []
    for r in rows:
        out.append(
            {
                "id": r.get("id", ""),
                "name": r.get("name", ""),
                "prompt_price": float((r.get("pricing", {}) or {}).get("prompt", 0) or 0),
                "completion_price": float((r.get("pricing", {}) or {}).get("completion", 0) or 0),
            }
        )
    return pd.DataFrame(out)


def models_json_from_df(df: pd.DataFrame, original: dict) -> dict:
    orig_by_id = {r.get("id"): r for r in (original.get("data", []) or []) if r.get("id")}
    new_rows = []
    for _, row in df.iterrows():
        mid = str(row.get("id", "")).strip()
        if not mid:
            continue
        base = dict(orig_by_id.get(mid, {}) or {})
        pricing = dict(base.get("pricing", {}) or {})
        pricing["prompt"] = str(row.get("prompt_price", 0))
        pricing["completion"] = str(row.get("completion_price", 0))
        base.update({"id": mid, "name": str(row.get("name", "")).strip() or mid, "pricing": pricing})
        new_rows.append(base)
    return {"data": new_rows}


# ---------- instruction editor ----------
def df_prompt_conditions(pc_json: dict) -> pd.DataFrame:
    rows = pc_json.get("prompt_conditions", []) or []
    return pd.DataFrame(
        [{"condition_id": r.get("condition_id", ""), "name": r.get("name", ""), "system_prompt": r.get("system_prompt", "")} for r in rows]
    )


def prompt_conditions_json_from_df(df: pd.DataFrame) -> dict:
    rows = []
    for _, r in df.iterrows():
        cid = str(r.get("condition_id", "")).strip()
        if not cid:
            continue
        rows.append(
            {"condition_id": cid, "name": str(r.get("name", "")).strip(), "system_prompt": str(r.get("system_prompt", "")).strip()}
        )
    return {"prompt_conditions": rows}


# ---------- variants editor ----------
def df_variants(v_json: dict) -> pd.DataFrame:
    rows = v_json.get("variants", []) or []
    return pd.DataFrame(
        [{"variant_id": r.get("variant_id", ""), "label": r.get("label", ""), "enabled": bool(r.get("enabled", True))} for r in rows]
    )


def variants_json_from_df(df: pd.DataFrame) -> dict:
    rows = []
    for _, r in df.iterrows():
        vid = str(r.get("variant_id", "")).strip()
        if not vid:
            continue
        rows.append({"variant_id": vid, "label": str(r.get("label", "")).strip(), "enabled": bool(r.get("enabled", True))})
    return {"variants": rows}


# ---------- components editor (prompts keyed by variant_id) ----------
def df_components(components_json: dict, variant_ids: List[str]) -> pd.DataFrame:
    tests = components_json.get("tests", []) or []
    out = []
    for t in tests:
        row = {"title": t.get("title", "")}
        prompts = t.get("prompts", {}) or {}
        for vid in variant_ids:
            row[f"prompt_{vid}"] = (prompts.get(vid, "") or "")
        out.append(row)
    return pd.DataFrame(out)


def components_json_from_df(df: pd.DataFrame, variant_ids: List[str]) -> dict:
    tests = []
    for _, r in df.iterrows():
        title = str(r.get("title", "")).strip()
        if not title:
            continue
        prompts = {vid: str(r.get(f"prompt_{vid}", "")).strip() for vid in variant_ids}
        tests.append({"title": title, "prompts": prompts})
    return {"tests": tests}


# ---------- rubric: definitions ----------
def all_check_ids(score_json: dict) -> List[str]:
    defs = (score_json.get("check_definitions", {}) or {})
    return sorted([k for k in defs.keys() if isinstance(k, str) and k.strip()])


def df_check_definitions(score_json: dict) -> pd.DataFrame:
    defs = (score_json.get("check_definitions", {}) or {})
    rows = []
    for check_id, d in defs.items():
        d = d or {}
        dr = (d.get("decision_rules", {}) or {})
        wcag = d.get("wcag", []) or []
        if isinstance(wcag, str):
            wcag = [wcag]
        rows.append(
            {
                "check_id": check_id,
                "title": d.get("title", ""),
                "wcag": ", ".join([str(x).strip() for x in wcag if str(x).strip()]),
                "notes": d.get("notes", ""),
                "rule_0": (dr.get("0") or dr.get(0) or ""),
                "rule_1": (dr.get("1") or dr.get(1) or ""),
                "rule_2": (dr.get("2") or dr.get(2) or ""),
            }
        )
    return pd.DataFrame(rows)


def score_json_from_check_definitions_df(df: pd.DataFrame, score_json: dict) -> dict:
    new = dict(score_json)
    defs_out: Dict[str, Dict[str, Any]] = {}

    for _, r in df.iterrows():
        cid = str(r.get("check_id", "")).strip()
        if not cid:
            continue
        wcag_list = [x.strip() for x in str(r.get("wcag", "")).split(",") if x.strip()]

        defs_out[cid] = {
            "title": str(r.get("title", "")).strip(),
            "wcag": wcag_list,
            "decision_rules": {
                "0": str(r.get("rule_0", "")).strip(),
                "1": str(r.get("rule_1", "")).strip(),
                "2": str(r.get("rule_2", "")).strip(),
            },
            "notes": str(r.get("notes", "")).strip(),
        }

    new["check_definitions"] = defs_out
    return new


# ---------- rubric: mapping ----------
def df_component_check_map(score_json: dict) -> pd.DataFrame:
    comps = score_json.get("components", []) or []
    rows = []
    for c in comps:
        rows.append({"id": c.get("id", ""), "name": c.get("name", ""), "checks": list(c.get("checks", []) or [])})
    return pd.DataFrame(rows)


def score_json_from_component_check_map_df(df: pd.DataFrame, score_json: dict) -> dict:
    new = dict(score_json)
    comps = []
    for _, r in df.iterrows():
        cid = str(r.get("id", "")).strip()
        if not cid:
            continue
        checks_val = r.get("checks", [])
        if isinstance(checks_val, list):
            checks = [str(x).strip() for x in checks_val if str(x).strip()]
        elif isinstance(checks_val, str):
            checks = [x.strip() for x in checks_val.split(",") if x.strip()]
        else:
            checks = []
        comps.append({"id": cid, "name": str(r.get("name", "")).strip(), "checks": checks})
    new["components"] = comps
    return new


# ---------- passthrough saves ----------
def save_models(obj: Dict[str, Any]) -> None:
    from bench.io import save_models as _s
    _s(obj)

def save_prompt_conditions(obj: Dict[str, Any]) -> None:
    from bench.io import save_prompt_conditions as _s
    _s(obj)

def save_variants(obj: Dict[str, Any]) -> None:
    from bench.io import save_variants as _s
    _s(obj)

def save_components(obj: Dict[str, Any]) -> None:
    from bench.io import save_components as _s
    _s(obj)

def save_score(obj: Dict[str, Any]) -> None:
    from bench.io import save_score as _s
    _s(obj)
