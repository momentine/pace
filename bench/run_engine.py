# bench/run_engine.py
from __future__ import annotations

import csv
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from bench.openrouter import OpenRouterClient, calculate_cost
from bench.registry import Registries
from bench.scoring import score_checks

# xe-core wrapper (you implement this in bench/axe_runner.py)
# Expected to return a dict-like or object with counts + optional raw JSON.
from bench.axe_runner import run_axe_on_fragment  # type: ignore


@dataclass
class RunConfig:
    run_name: str
    out_root: Path
    model_ids: list[str]
    condition_ids: list[str]
    variant_ids: list[str]
    component_indices: list[int]
    repetitions: int
    max_tokens: int
    temperature: float
    top_p: float
    rate_limit: float
    do_score: bool
    do_axe: bool = True  # NEW: keep default True so “everything works” without changing your UI yet

    # Optional: experiment with axe configurations
    axe_profile: str = "form_fragment_rules"
    axe_config: dict[str, Any] | None = None
    axe_ruleset: list[str] | None = None
    axe_timeout_ms: int = 10_000


def _save_html(
    run_dir: Path,
    model_id: str,
    condition_id: str,
    variant_id: str,
    component_id: str,
    rep_idx: int,
    html: str,
) -> Path:
    safe_model = model_id.replace("/", "_").replace(":", "_")
    d = run_dir / safe_model / condition_id / variant_id / component_id / f"rep_{rep_idx:02d}"
    d.mkdir(parents=True, exist_ok=True)
    fn = d / f"{uuid.uuid4().hex[:10]}.html"
    fn.write_text(html, encoding="utf-8")
    return fn


def _variant_index_from_id(variant_id: str) -> int | None:
    """
    G1 -> 0, G2 -> 1, ...
    """
    m = re.match(r"^G(\d+)$", (variant_id or "").strip())
    if not m:
        return None
    n = int(m.group(1))
    if n < 1:
        return None
    return n - 1


def _prompt_from_test_variant(test: dict[str, Any], variant_id: str, variant_row: dict[str, Any]) -> str:
    """
    Supports multiple component schemas:

    A) prompts as dict keyed by variant_id:
       test["prompts"] = {"G1": "...", "G2": "...", ...}

    B) prompts as list in order (G1..):
       test["prompts"] = ["...", "...", ...]

    C) schema using variant template:
       variant: template: "Insert a {component} for “{label}”{suffix}"
       test: {"component":"text field","label":"...","suffix":" with hint ..."}
    """
    prompts = test.get("prompts")

    # ---- A) prompts dict keyed by variant_id
    if isinstance(prompts, dict):
        p = str(prompts.get(variant_id, "") or "").strip()
        if p:
            return p

    # ---- B) prompts list in G1 order
    if isinstance(prompts, list) and prompts:
        idx = _variant_index_from_id(variant_id)
        if idx is not None and 0 <= idx < len(prompts):
            p = str(prompts[idx] or "").strip()
            if p:
                return p

    # ---- C) template fill
    tpl = str(variant_row.get("template") or "").strip()
    if not tpl:
        return ""

    component = str(
        test.get("component")
        or test.get("component_phrase")
        or test.get("component_text")
        or ""
    ).strip()
    label = str(test.get("label") or "").strip()
    suffix = str(test.get("suffix") or "").strip()

    if suffix and not suffix.startswith(" "):
        suffix = " " + suffix

    if not component or not label:
        return ""

    fields = {"component": component, "label": label, "suffix": suffix}

    def repl(m: re.Match) -> str:
        k = m.group(1)
        return fields.get(k, "")

    out = re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, tpl).strip()
    return out


class RunCancelled(Exception):
    """Raised to stop nested loops cleanly while keeping partial outputs."""
    pass


def run_benchmark(
    reg: Registries,
    cfg: RunConfig,
    client: Optional[OpenRouterClient] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,  # ✅ supports Stop
) -> Path:
    client = client or OpenRouterClient()
    should_cancel = should_cancel or (lambda: False)

    out_root = cfg.out_root
    runs_root = out_root / "_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- registries ----
    models = reg.models.get("data", []) or []
    model_by_id: dict[str, dict[str, Any]] = {
        str(m.get("id")).strip(): m
        for m in models
        if isinstance(m, dict) and str(m.get("id") or "").strip()
    }

    conds = reg.prompt_conditions.get("prompt_conditions", []) or []
    cond_by_id: dict[str, dict[str, Any]] = {
        str(c.get("condition_id")).strip(): c
        for c in conds
        if isinstance(c, dict) and str(c.get("condition_id") or "").strip()
    }

    variants = reg.variants.get("variants", []) or []
    variant_by_id: dict[str, dict[str, Any]] = {
        str(v.get("variant_id")).strip(): v
        for v in variants
        if isinstance(v, dict) and str(v.get("variant_id") or "").strip()
    }

    tests = reg.components.get("tests", []) or []
    score_components: dict[str, dict[str, Any]] = {
        str(c.get("id")).strip(): c
        for c in (reg.score.get("components", []) or [])
        if isinstance(c, dict) and str(c.get("id") or "").strip()
    }

    def comp_id_from_index(i: int) -> str:
        return f"C{i+1:02d}"

    def model_name(mid: str) -> str:
        m = model_by_id.get(mid, {}) or {}
        return str(m.get("name") or mid).strip() or mid

    def condition_name(cid: str) -> str:
        c = cond_by_id.get(cid, {}) or {}
        return str(c.get("name") or cid).strip() or cid

    def variant_label(vid: str) -> str:
        v = variant_by_id.get(vid, {}) or {}
        return str(v.get("label") or vid).strip() or vid

    total = (
        len(cfg.model_ids)
        * len(cfg.condition_ids)
        * len(cfg.variant_ids)
        * len(cfg.component_indices)
        * int(cfg.repetitions)
    )
    done = 0

    results_path = run_dir / "results.csv"
    per_path = run_dir / "per_check.csv"

    # ✅ Dynamic columns:
    # - score columns only if cfg.do_score
    # - axe columns only if cfg.do_axe
    results_cols = [
        "model_id",
        "model_name",
        "condition_id",
        "condition_name",
        "variant_id",
        "variant_label",
        "component_id",
        "component_title",
        "rep_idx",
        "prompt",
        "ok",
        "error",
        "prompt_tokens",
        "completion_tokens",
        "cost",
        "output_file",
    ]

    if cfg.do_score:
        results_cols += [
            "raw_score",
            "max_score",
            "norm_score",
        ]

    if getattr(cfg, "do_axe", False):
        results_cols += [
            "axe_ok",
            "axe_error",
            "axe_profile",
            "axe_violations",
            "axe_incomplete",
            "axe_passes",
            "axe_inapplicable",
            "axe_json_file",
        ]

    per_cols = [
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

    write_results_header = (not results_path.exists()) or results_path.stat().st_size == 0

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(done, total, msg)

    # Open results.csv always; open per_check.csv only if scoring is enabled
    with results_path.open("a", newline="", encoding="utf-8") as rf:
        rwriter = csv.DictWriter(rf, fieldnames=results_cols, extrasaction="ignore")

        if write_results_header:
            rwriter.writeheader()
            rf.flush()

        pf = None
        pwriter = None
        if cfg.do_score:
            write_per_header = (not per_path.exists()) or per_path.stat().st_size == 0
            pf = per_path.open("a", newline="", encoding="utf-8")
            pwriter = csv.DictWriter(pf, fieldnames=per_cols, extrasaction="ignore")
            if write_per_header:
                pwriter.writeheader()
                pf.flush()

        try:
            for mid in cfg.model_ids:
                m_name = model_name(mid)

                for cid in cfg.condition_ids:
                    c_row = cond_by_id.get(cid, {}) or {}
                    sys_prompt = str(c_row.get("system_prompt") or "")
                    c_name = condition_name(cid)

                    for vid in cfg.variant_ids:
                        v_row = variant_by_id.get(vid, {}) or {}
                        v_lab = variant_label(vid)

                        for i in cfg.component_indices:
                            if i < 0 or i >= len(tests):
                                continue

                            component_id = comp_id_from_index(i)
                            comp_title = str(tests[i].get("title", f"Component {i+1}") or f"Component {i+1}")

                            user_prompt = _prompt_from_test_variant(tests[i], vid, v_row).strip()
                            checks = (score_components.get(component_id, {}) or {}).get("checks", [])
                            if not isinstance(checks, list):
                                checks = []

                            for rep in range(int(cfg.repetitions)):
                                if should_cancel():
                                    raise RunCancelled()

                                done += 1
                                _progress(
                                    f"{done:,}/{total:,} | model={mid} | condition={cid} | "
                                    f"variant={vid} | component={component_id} | rep={rep+1}"
                                )

                                gen_ok = False
                                gen_error = ""
                                gen_content = ""
                                prompt_tokens = 0
                                completion_tokens = 0

                                if not sys_prompt.strip():
                                    gen_ok = False
                                    gen_error = "Empty system prompt for this prompt condition."
                                elif not user_prompt:
                                    gen_ok = False
                                    gen_error = "Empty user prompt for this component×variant (check components.json)."
                                else:
                                    gen = client.generate(
                                        model_id=mid,
                                        system_prompt=sys_prompt,
                                        user_prompt=user_prompt,
                                        max_tokens=int(cfg.max_tokens),
                                        temperature=float(cfg.temperature),
                                        top_p=float(cfg.top_p),
                                    )
                                    gen_ok = bool(gen.ok)
                                    gen_error = gen.error or ""
                                    gen_content = gen.content
                                    prompt_tokens = int(gen.prompt_tokens)
                                    completion_tokens = int(gen.completion_tokens)

                                cost = 0.0
                                out_file = ""
                                raw_score = np.nan
                                max_score = np.nan
                                norm_score = np.nan

                                # Default axe fields (written only if axe cols exist)
                                axe_ok: Any = np.nan
                                axe_error = ""
                                axe_violations: Any = np.nan
                                axe_incomplete: Any = np.nan
                                axe_passes: Any = np.nan
                                axe_inapplicable: Any = np.nan
                                axe_json_file = ""

                                if gen_ok:
                                    cost = float(calculate_cost(model_by_id.get(mid, {}), prompt_tokens, completion_tokens))
                                    out_file = str(_save_html(run_dir, mid, cid, vid, component_id, rep + 1, gen_content))

                                    # ✅ run axe-core only if selected
                                    if getattr(cfg, "do_axe", False):
                                        if should_cancel():
                                            raise RunCancelled()
                                        try:
                                            axe_res = run_axe_on_fragment(
                                                html=gen_content,
                                                run_dir=run_dir,
                                                output_file=out_file,
                                                meta={
                                                    "model_id": mid,
                                                    "condition_id": cid,
                                                    "variant_id": vid,
                                                    "component_id": component_id,
                                                    "rep_idx": rep + 1,
                                                    "axe_profile": str(getattr(cfg, "axe_profile", "unknown")),
                                                },
                                                axe_config=getattr(cfg, "axe_config", None),
                                                ruleset=getattr(cfg, "axe_ruleset", None),
                                                timeout_ms=int(getattr(cfg, "axe_timeout_ms", 10_000)),
                                            )

                                            if isinstance(axe_res, dict):
                                                axe_ok = bool(axe_res.get("ok", True))
                                                axe_error = str(axe_res.get("error") or "")
                                                counts = axe_res.get("counts") or {}
                                                if isinstance(counts, dict):
                                                    axe_violations = int(counts.get("violations", counts.get("violation", 0)) or 0)
                                                    axe_incomplete = int(counts.get("incomplete", 0) or 0)
                                                    axe_passes = int(counts.get("passes", 0) or 0)
                                                    axe_inapplicable = int(counts.get("inapplicable", 0) or 0)
                                                axe_json_file = str(axe_res.get("json_path") or axe_res.get("axe_json_file") or "")
                                            else:
                                                axe_ok = True
                                                axe_violations = int(getattr(axe_res, "violations", getattr(axe_res, "n_violations", 0)) or 0)
                                                axe_incomplete = int(getattr(axe_res, "incomplete", getattr(axe_res, "n_incomplete", 0)) or 0)
                                                axe_passes = int(getattr(axe_res, "passes", getattr(axe_res, "n_passes", 0)) or 0)
                                                axe_inapplicable = int(getattr(axe_res, "inapplicable", getattr(axe_res, "n_inapplicable", 0)) or 0)
                                                axe_json_file = str(getattr(axe_res, "json_path", "") or "")
                                        except Exception as e:
                                            axe_ok = False
                                            axe_error = str(e)

                                    # ✅ scoring only if selected
                                    if cfg.do_score:
                                        if should_cancel():
                                            raise RunCancelled()
                                        raw, mx, per = score_checks(gen_content, checks, reg.score)
                                        raw_score = float(raw)
                                        max_score = float(mx)
                                        norm_score = float((raw / mx) if mx > 0 else 0.0)

                                        if pwriter is not None:
                                            for pcs in per:
                                                pwriter.writerow(
                                                    {
                                                        "model_id": mid,
                                                        "model_name": m_name,
                                                        "condition_id": cid,
                                                        "condition_name": c_name,
                                                        "variant_id": vid,
                                                        "variant_label": v_lab,
                                                        "component_id": component_id,
                                                        "component_title": comp_title,
                                                        "rep_idx": rep + 1,
                                                        "check_id": pcs.check_id,
                                                        "score": pcs.score,
                                                        "rationale": pcs.rationale,
                                                        "output_file": out_file,
                                                    }
                                                )
                                            if pf is not None:
                                                pf.flush()

                                rwriter.writerow(
                                    {
                                        "model_id": mid,
                                        "model_name": m_name,
                                        "condition_id": cid,
                                        "condition_name": c_name,
                                        "variant_id": vid,
                                        "variant_label": v_lab,
                                        "component_id": component_id,
                                        "component_title": comp_title,
                                        "rep_idx": rep + 1,
                                        "prompt": user_prompt,
                                        "ok": bool(gen_ok),
                                        "error": gen_error,
                                        "prompt_tokens": int(prompt_tokens),
                                        "completion_tokens": int(completion_tokens),
                                        "cost": float(cost),
                                        "output_file": out_file,
                                        # score fields (only in CSV if selected)
                                        "raw_score": raw_score,
                                        "max_score": max_score,
                                        "norm_score": norm_score,
                                        # axe fields (only in CSV if selected)
                                        "axe_ok": axe_ok,
                                        "axe_error": axe_error,
                                        "axe_profile": str(getattr(cfg, "axe_profile", "")),
                                        "axe_violations": axe_violations,
                                        "axe_incomplete": axe_incomplete,
                                        "axe_passes": axe_passes,
                                        "axe_inapplicable": axe_inapplicable,
                                        "axe_json_file": axe_json_file,
                                    }
                                )
                                rf.flush()

                                if cfg.rate_limit and cfg.rate_limit > 0:
                                    if should_cancel():
                                        raise RunCancelled()
                                    time.sleep(float(cfg.rate_limit))

        except RunCancelled:
            _progress("Cancelled by user. Partial CSV saved.")
            return run_dir
        finally:
            if pf is not None:
                pf.close()

    return run_dir