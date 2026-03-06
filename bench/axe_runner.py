# bench/axe_runner.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.sync_api import Browser, Page, Playwright, sync_playwright

# ----------------------------
# Default ruleset for form fragments
# NOTE: not all axe versions have all of these; we sanitize at runtime.
# ----------------------------

DEFAULT_FORM_FRAGMENT_RULES = [

    # ---- Accessible name / labeling ----
    "label",
    "label-title-only",
    "form-field-multiple-labels",
    "aria-input-field-name",
    "button-name",
    "input-button-name",
    "select-name",
    "textarea-name",
    "input-image-alt",

    # ---- ARIA correctness ----
    "aria-valid-attr",
    "aria-allowed-attr",
    "aria-required-attr",
    "aria-valid-attr-value",
    "aria-hidden-focus",
    "aria-required-children",
    "aria-required-parent",
    "aria-roles",
    "aria-roles-required-attr",

    # ---- Error state semantics ----
    "aria-errormessage",
    "aria-invalid",

    # ---- ID / reference integrity ----
    "duplicate-id",
    "duplicate-id-aria",
    "duplicate-id-active",

    # ---- Form attribute validity ----
    "autocomplete-valid",

    # ---- Group semantics ----
    "fieldset",

    # ---- Toggle / switch ----
    "aria-toggle-field-name",
]



def build_axe_config_for_rules(rules: List[str]) -> Dict[str, Any]:
    return {
        "runOnly": {"type": "rule", "values": list(rules)},
        "resultTypes": ["violations", "incomplete", "passes", "inapplicable"],
    }


# ----------------------------
# Data model
# ----------------------------

@dataclass
class AxeCounts:
    passes: int
    violations: int
    incomplete: int
    inapplicable: int


@dataclass
class AxeImpactCounts:
    minor: int
    moderate: int
    serious: int
    critical: int
    unknown: int = 0


@dataclass
class AxeViolationSummary:
    rule_id: str
    impact: str
    help: str
    help_url: str
    nodes: int


@dataclass
class AxeSummary:
    axe_version: str
    browser_name: str
    browser_version: str
    counts: AxeCounts
    impact_counts: AxeImpactCounts
    coverage_rules: float
    pass_rate_strict: float
    pass_rate_lenient: float
    status: str  # PASS | WARN | FAIL
    violations: List[AxeViolationSummary]
    raw: Dict[str, Any]


# ----------------------------
# Core axe runner (returns AxeSummary)
# ----------------------------

def run_axe_summary(
    html_fragment: str,
    *,
    axe_js_path: str | Path,
    axe_config: Optional[Dict[str, Any]] = None,
    ruleset: Optional[List[str]] = None,
    viewport: Tuple[int, int] = (900, 700),
    timeout_ms: int = 10_000,
    include_raw: bool = True,
) -> AxeSummary:
    axe_js_path = Path(axe_js_path)
    if not axe_js_path.exists():
        raise FileNotFoundError(f"axe_js_path not found: {axe_js_path}")

    wrapper_html = _wrap_fragment(html_fragment)

    import sys, asyncio
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            asyncio.set_event_loop(asyncio.ProactorEventLoop())
        except Exception:
            pass

    with sync_playwright() as p:
        browser = _launch_browser(p)
        try:
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)

            page.set_content(wrapper_html, wait_until="domcontentloaded", timeout=timeout_ms)

            # Inject axe
            page.add_script_tag(path=str(axe_js_path))

            # Sanitize config/ruleset so older/newer axe versions don’t crash on unknown rule IDs
            if axe_config is None:
                desired = ruleset or DEFAULT_FORM_FRAGMENT_RULES
                axe_config, _missing = _build_sanitized_runonly_config(page, desired)
            else:
                axe_config, _missing = _sanitize_existing_config(page, axe_config)

            # Run axe in page context (no timeout kwarg; Playwright versions differ)
            result = _axe_run(page, axe_config=axe_config)

            # Attach missing rules info into raw for debug
            if isinstance(result, dict) and _missing:
                result["_pace_missing_rules"] = list(_missing)

            return _summarize_axe_result(
                result,
                browser_name="chromium",
                browser_version=browser.version,
                include_raw=include_raw,
            )
        finally:
            browser.close()


# ----------------------------
# Internals
# ----------------------------

def _launch_browser(p: Playwright) -> Browser:
    return p.chromium.launch(headless=True)


def _wrap_fragment(fragment: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>axe-runner</title>
</head>
<body>
  <main id="root">
    {fragment}
  </main>
</body>
</html>
"""


def _axe_run(page: Page, *, axe_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config_json = json.dumps(axe_config or {})
    js = f"""
(() => {{
  if (!window.axe) {{
    throw new Error("axe not found on window. Did axe.min.js load?");
  }}
  const root = document.querySelector('#root') || document.body;
  const cfg = {config_json};
  return window.axe.run(root, cfg);
}})()
"""
    return page.evaluate(js)


def _get_available_rule_ids(page: Page) -> List[str]:
    js = """
(() => {
  if (!window.axe || !window.axe.getRules) return [];
  const rules = window.axe.getRules() || [];
  return rules.map(r => String(r.ruleId || r.id || "")).filter(Boolean);
})()
"""
    out = page.evaluate(js)
    if isinstance(out, list):
        return [str(x) for x in out if str(x)]
    return []


def _build_sanitized_runonly_config(page: Page, desired_rules: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    available = set(_get_available_rule_ids(page))
    if not available:
        # If we can't introspect rules, safest is: run all rules (no runOnly),
        # instead of crashing.
        return {"resultTypes": ["violations", "incomplete", "passes", "inapplicable"]}, []

    keep = [r for r in desired_rules if r in available]
    missing = [r for r in desired_rules if r not in available]

    # If everything got filtered out, run all rules rather than returning empty runOnly.
    if not keep:
        return {"resultTypes": ["violations", "incomplete", "passes", "inapplicable"]}, missing

    return build_axe_config_for_rules(keep), missing


def _sanitize_existing_config(page: Page, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    # Only sanitize runOnly.type == "rule"
    run_only = cfg.get("runOnly")
    if not isinstance(run_only, dict):
        return cfg, []

    if str(run_only.get("type") or "").strip().lower() != "rule":
        return cfg, []

    values = run_only.get("values")
    if not isinstance(values, list):
        return cfg, []

    desired = [str(v) for v in values if str(v)]
    sanitized, missing = _build_sanitized_runonly_config(page, desired)

    # Preserve other top-level config fields, but overwrite runOnly + resultTypes if present
    out = dict(cfg)
    out["resultTypes"] = sanitized.get("resultTypes", out.get("resultTypes"))
    if "runOnly" in sanitized:
        out["runOnly"] = sanitized["runOnly"]
    else:
        # drop runOnly if we decided to run all rules
        out.pop("runOnly", None)

    return out, missing


def _summarize_axe_result(
    axe_result: Dict[str, Any],
    *,
    browser_name: str,
    browser_version: str,
    include_raw: bool,
) -> AxeSummary:
    passes = axe_result.get("passes") or []
    violations = axe_result.get("violations") or []
    incomplete = axe_result.get("incomplete") or []
    inapplicable = axe_result.get("inapplicable") or []

    counts = AxeCounts(
        passes=len(passes),
        violations=len(violations),
        incomplete=len(incomplete),
        inapplicable=len(inapplicable),
    )

    impact_counts = _count_impacts(violations)

    coverage_rules = _safe_div(
        counts.passes + counts.violations + counts.incomplete,
        counts.passes + counts.violations + counts.incomplete + counts.inapplicable,
        default=0.0,
    )

    pass_rate_strict = _safe_div(
        counts.passes,
        counts.passes + counts.violations + counts.incomplete,
        default=1.0,
    )

    pass_rate_lenient = _safe_div(
        counts.passes,
        counts.passes + counts.violations,
        default=1.0,
    )

    status = _status_from_impacts(impact_counts, counts)

    v_summaries: List[AxeViolationSummary] = []
    for v in violations:
        v_summaries.append(
            AxeViolationSummary(
                rule_id=str(v.get("id") or ""),
                impact=str(v.get("impact") or "unknown"),
                help=str(v.get("help") or ""),
                help_url=str(v.get("helpUrl") or ""),
                nodes=len(v.get("nodes") or []),
            )
        )

    axe_version = str((axe_result.get("testEngine") or {}).get("version") or "")

    return AxeSummary(
        axe_version=axe_version,
        browser_name=browser_name,
        browser_version=browser_version,
        counts=counts,
        impact_counts=impact_counts,
        coverage_rules=coverage_rules,
        pass_rate_strict=pass_rate_strict,
        pass_rate_lenient=pass_rate_lenient,
        status=status,
        violations=v_summaries,
        raw=axe_result if include_raw else {},
    )


def _count_impacts(violations: List[Dict[str, Any]]) -> AxeImpactCounts:
    minor = moderate = serious = critical = unknown = 0
    for v in violations:
        impact = str(v.get("impact") or "unknown").strip().lower()
        if impact == "minor":
            minor += 1
        elif impact == "moderate":
            moderate += 1
        elif impact == "serious":
            serious += 1
        elif impact == "critical":
            critical += 1
        else:
            unknown += 1
    return AxeImpactCounts(minor=minor, moderate=moderate, serious=serious, critical=critical, unknown=unknown)


def _status_from_impacts(imp: AxeImpactCounts, counts: AxeCounts) -> str:
    if imp.serious > 0 or imp.critical > 0:
        return "FAIL"
    if counts.violations > 0 or counts.incomplete > 0:
        return "WARN"
    return "PASS"


def _safe_div(num: int, den: int, *, default: float) -> float:
    if den <= 0:
        return default
    return float(num) / float(den)


def summary_to_dict(s: AxeSummary) -> Dict[str, Any]:
    d = asdict(s)
    d["counts"] = asdict(s.counts)
    d["impact_counts"] = asdict(s.impact_counts)
    return d


# ----------------------------
# Compatibility wrapper for bench/run_engine.py
# ----------------------------

def _default_axe_js_path() -> Path:
    env = (os.getenv("AXE_JS_PATH") or "").strip()
    if env:
        return Path(env).expanduser()

    candidates = [
        Path("node_modules/axe-core/axe.min.js"),
        Path("bench/vendor/axe.min.js"),
        Path("vendor/axe.min.js"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

def run_axe_on_fragment(
    *,
    html: str,
    run_dir: Optional[Path] = None,
    output_file: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    axe_js_path: Optional[str | Path] = None,
    axe_config: Optional[Dict[str, Any]] = None,
    ruleset: Optional[List[str]] = None,
    timeout_ms: int = 10_000,
) -> Dict[str, Any]:
    try:
        js_path = Path(axe_js_path) if axe_js_path is not None else _default_axe_js_path()

        summary = run_axe_summary(
            html_fragment=html,
            axe_js_path=js_path,
            axe_config=axe_config,
            ruleset=ruleset,
            timeout_ms=timeout_ms,
            include_raw=True,
        )

        json_path = ""
        try:
            json_p: Optional[Path]
            if output_file:
                json_p = Path(output_file).with_suffix(".axe.json")
            elif run_dir:
                json_p = Path(run_dir) / "axe_last.json"
            else:
                json_p = None

            if json_p is not None:
                payload = {
                    "meta": meta or {},
                    "summary": summary_to_dict(summary),
                    "raw": summary.raw,
                }
                json_p.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                json_path = str(json_p)
        except Exception:
            json_path = ""

        # ---------------------------------------------------------
        # quantitative scoring columns
        # ---------------------------------------------------------

        axe_score = float(summary.pass_rate_strict)
        axe_coverage = float(summary.coverage_rules)

        return {
            "ok": True,
            "error": "",
            "counts": {
                "violations": int(summary.counts.violations),
                "incomplete": int(summary.counts.incomplete),
                "passes": int(summary.counts.passes),
                "inapplicable": int(summary.counts.inapplicable),
            },
            "axe_score": axe_score,
            "axe_coverage": axe_coverage,
            "json_path": json_path,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "counts": {
                "violations": 0,
                "incomplete": 0,
                "passes": 0,
                "inapplicable": 0,
            },
            "axe_score": 0.0,
            "axe_coverage": 0.0,
            "json_path": "",
        }
