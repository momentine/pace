
# app.py
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from bench.openrouter import OpenRouterClient
from bench.registry import ensure_defaults, load_all, save_prompt_conditions
from bench.run_engine import RunConfig, run_benchmark
from bench.axe_runner import DEFAULT_FORM_FRAGMENT_RULES

try:
    import numpy as np
except Exception:
    np = None

try:
    import altair as alt
except Exception:
    alt = None


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="PACE: Prompt Accessibility Controlled Evaluation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_ROOT = Path("output")
RUNS_ROOT = OUT_ROOT / "_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

SELECTED_MODELS_PATH = DATA_DIR / "selected_models.json"
VARIANTS_PATH = DATA_DIR / "variants.json"
COMPONENTS_PATH = DATA_DIR / "components.json"
SCORE_PATH = DATA_DIR / "score.json"

# ----------------------------
# Global CSS
# ----------------------------

st.markdown(
    """
<style>
/* =========================
   Global layout polish (no theme color changes)
   ========================= */

/* Center the app content with a readable max-width */
.block-container{
  max-width: 1120px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: 1.75rem !important;
  padding-right: 1.75rem !important;
  padding-top: 1.25rem !important;
  padding-bottom: 2.25rem !important;
}

/* Hide sidebar + the collapse button */
section[data-testid="stSidebar"]{ display:none !important; }
button[kind="headerNoPadding"]{ display:none !important; }

/* Smoother typography + spacing */
html, body, [data-testid="stAppViewContainer"]{
  -webkit-font-smoothing: antialiased;
  text-rendering: geometricPrecision;
}
h1, h2, h3 { letter-spacing: -0.01em; }
p, li { line-height: 1.45; }

/* Make columns and horizontal blocks breathe a bit */
div[data-testid="stHorizontalBlock"]{ gap: 1.0rem !important; }

/* Inputs: consistent height and corner radius (keeps theme colors) */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] div[role="combobox"],
div[data-testid="stMultiSelect"] div[role="combobox"]{
  border-radius: 12px !important;
}

/* Buttons: consistent shape/weight (keeps theme colors) */
div.stButton > button{
  border-radius: 12px !important;
  font-weight: 600 !important;
  padding: 0.7rem 0.95rem !important;
}

/* Tabs: a bit tighter + clearer separation (keeps active color) */
div[data-baseweb="tab-list"]{
  gap: 0.25rem !important;
}
div[data-baseweb="tab"]{
  border-radius: 12px !important;
  padding: 0.55rem 0.8rem !important;
}

/* Expanders: softer edges */
details[data-testid="stExpander"]{
  border-radius: 14px !important;
  overflow: hidden;
}

/* =========================
   Settings dialog (centered, scrollable, full-screen backdrop)
   ========================= */

/* Backdrop (covers header too) */
div[data-baseweb="backdrop"]{
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 200000 !important;
}

/* Modal root */
div[data-baseweb="modal"]{
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 200001 !important;
}

/* Dialog box */
div[data-baseweb="modal"] [role="dialog"],
div[data-testid="stDialog"] [role="dialog"]{
  position: fixed !important;
  top: 50% !important;
  left: 50% !important;
  transform: translate(-50%, -50%) !important;

  width: min(920px, 92vw) !important;
  max-height: 86vh !important;
  overflow: auto !important;

  border-radius: 18px !important;
}

/* If Streamlit wraps dialog content, keep it scrollable */
div[data-baseweb="modal"] [role="dialog"] > div{
  max-height: 86vh !important;
  overflow: auto !important;
}

/* Make sure nothing sits above the backdrop */
.app-header{ z-index: 2 !important; }
header[data-testid="stHeader"]{ z-index: 1 !important; }

/* =========================
   Small niceties
   ========================= */

/* Reduce overly-wide code blocks */
pre, code { border-radius: 12px !important; }

/* Cleaner separators */
hr{ opacity: 0.35; }


/* =========================
   Extra layout polish (keeps existing colors)
   ========================= */

/* Slightly tighter rhythm + better readability */
.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 2.5rem !important;
}

/* Cards: soften corners + subtle elevation without changing theme colors */
div[data-testid="stContainer"],
div[data-testid="stExpander"],
div[data-testid="stForm"],
div[data-testid="stPopoverBody"],
div[data-testid="stDialog"] > div {
  border-radius: 0px !important;
}

/* Inputs: consistent radius + comfy hit targets */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="tag"] {
  border-radius: 14px !important;
}

/* Buttons: same colors, nicer shape + spacing */
div.stButton > button,
button[kind="secondary"],
button[kind="primary"]{
  border-radius: 14px !important;
  padding: 0.7rem 1rem !important;
  font-weight: 600 !important;
}

/* Tabs: pill-like without recoloring */
button[data-baseweb="tab"]{
  border-radius: 999px !important;
  padding: 0.55rem 0.9rem !important;
}

/* Expanders: cleaner header spacing */
div[data-testid="stExpander"] summary{
  padding-top: 0.35rem !important;
  padding-bottom: 0.35rem !important;
}

/* Reduce visual jitter on reruns */
* {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Session flags
# ----------------------------
if "_settings_open" not in st.session_state:
    st.session_state["_settings_open"] = False
if "_reg_reload" not in st.session_state:
    st.session_state["_reg_reload"] = False

def _render_axe_settings() -> None:
    st.caption("Configure axe-core scoring for runs. These settings act as experimental parameters.")

    st.session_state.setdefault("axe_profile", "form_fragment_rules")

    profile = st.selectbox(
        "Axe profile",
        options=[
            "form_fragment_rules",
            "all_rules",
            "wcag2a_tags",
            "wcag2aa_tags",
            "wcag21aa_tags",
        ],
        format_func=lambda x: {
            "form_fragment_rules": "Form fragment rules (PACE default)",
            "all_rules": "All axe rules",
            "wcag2a_tags": "WCAG 2 A tags",
            "wcag2aa_tags": "WCAG 2 AA tags",
            "wcag21aa_tags": "WCAG 2.1 AA tags",
        }.get(x, x),
        key="axe_profile",
    )

    # Keep timeout out of UI (not an experimental parameter)
    timeout_ms = 10_000
    result_types = ["violations", "incomplete", "passes", "inapplicable"]

    axe_config: dict[str, Any] | None = None
    axe_ruleset: list[str] | None = None

    if profile == "all_rules":
        axe_config = {"resultTypes": result_types}
        axe_ruleset = None

    elif profile in ("wcag2a_tags", "wcag2aa_tags", "wcag21aa_tags"):
        tag = {
            "wcag2a_tags": "wcag2a",
            "wcag2aa_tags": "wcag2aa",
            "wcag21aa_tags": "wcag21aa",
        }[profile]
        axe_config = {"runOnly": {"type": "tag", "values": [tag]}, "resultTypes": result_types}
        axe_ruleset = None

    else:  # form_fragment_rules
        axe_ruleset = list(DEFAULT_FORM_FRAGMENT_RULES)
        axe_config = None  # triggers sanitized runOnly(rule) path in axe_runner

    # Persist for RunConfig construction
    st.session_state["axe_timeout_ms_obj"] = int(timeout_ms)
    st.session_state["axe_config_obj"] = axe_config
    st.session_state["axe_ruleset_obj"] = axe_ruleset

    st.divider()
    st.markdown("**What will run:**")

    if profile == "form_fragment_rules":
        st.write("**Mode:** Rule list (PACE default)")
        st.write(f"**Rules:** {len(DEFAULT_FORM_FRAGMENT_RULES)} rule IDs")
        with st.expander("Show rule IDs", expanded=False):
            st.code("\n".join(DEFAULT_FORM_FRAGMENT_RULES))

    elif profile == "all_rules":
        st.write("**Mode:** All rules in axe-core")
        st.write("**Filter:** None")
        st.caption("This runs every rule loaded in axe.min.js.")

    else:
        tag = {
            "wcag2a_tags": "wcag2a",
            "wcag2aa_tags": "wcag2aa",
            "wcag21aa_tags": "wcag21aa",
        }[profile]
        st.write("**Mode:** Tag filter")
        st.write(f"**Tag:** `{tag}`")
        st.caption("This runs all axe-core rules tagged with the selected WCAG level.")

    with st.expander("Show raw config", expanded=False):
        st.json(
            {
                "axe_profile": profile,
                "timeout_ms": int(timeout_ms),
                "axe_config": axe_config,
                "axe_ruleset": axe_ruleset,
            },
            expanded=False,
        )


def _stay_in_settings() -> None:
    """
    Keep the Settings dialog open across reruns triggered by edits.
    IMPORTANT: this is NOT a widget key, so it's always safe to set.
    """
    st.session_state["_settings_open"] = True


# ----------------------------
# IO helpers
# ----------------------------
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _throttle_ok(key: str, min_secs: float = 0.15) -> bool:
    now = time.time()
    last = float(st.session_state.get(key, 0.0))
    if now - last < min_secs:
        return False
    st.session_state[key] = now
    return True


def _provider(mid: str) -> str:
    return mid.split("/")[0] if "/" in mid else "other"


# ----------------------------
# Settings: Models editor
# ----------------------------
def _load_selected_ids_from_file(raw_by_id: dict[str, dict[str, Any]]) -> set[str]:
    obj = _read_json(SELECTED_MODELS_PATH)
    rows = obj.get("data", []) if isinstance(obj, dict) else []
    if not isinstance(rows, list):
        return set()
    out: set[str] = set()
    for r in rows:
        if isinstance(r, dict):
            mid = str(r.get("id") or "").strip()
            if mid and mid in raw_by_id:
                out.add(mid)
    return out


def _save_selected_models_full_objects(
    ids: set[str],
    raw_by_id: dict[str, dict[str, Any]],
    by_id: dict[str, dict[str, str]],
) -> None:
    keep_ids = [mid for mid in ids if mid in raw_by_id]
    keep_ids.sort(key=lambda mid: (_provider(mid).lower(), by_id[mid]["name"].lower(), mid.lower()))
    payload = {"data": [raw_by_id[mid] for mid in keep_ids]}
    _write_json(SELECTED_MODELS_PATH, payload)


def _render_models_editor(reg) -> None:
    st.markdown("#### Models")
    st.caption("Customize which models to test.")

    catalog = reg.models.get("data", []) or []
    raw_by_id: dict[str, dict[str, Any]] = {}
    for obj in catalog:
        if isinstance(obj, dict):
            mid = str(obj.get("id") or "").strip()
            if mid:
                raw_by_id[mid] = obj

    models: list[dict[str, str]] = []
    for mid, obj in raw_by_id.items():
        name = str(obj.get("name") or mid).strip()
        provider = _provider(mid)
        models.append({"id": mid, "name": name, "provider": provider})

    models.sort(key=lambda x: (x["provider"].lower(), x["name"].lower(), x["id"].lower()))
    by_id: dict[str, dict[str, str]] = {m["id"]: m for m in models}
    catalog_ids = set(by_id.keys())

    def _pick_key(mid: str) -> str:
        return f"settings_pick_{mid}"

    if "settings_selected_models" not in st.session_state:
        seeded = _load_selected_ids_from_file(raw_by_id)
        st.session_state["settings_selected_models"] = {mid for mid in seeded if mid in catalog_ids}
        for mid in st.session_state["settings_selected_models"]:
            st.session_state[_pick_key(mid)] = True

    def _autosave_now() -> None:
        _stay_in_settings()
        ids = {x for x in st.session_state.get("settings_selected_models", set()) if x in catalog_ids}
        _save_selected_models_full_objects(ids, raw_by_id, by_id)
        st.session_state["_reg_reload"] = True

    def _sync_one(mid: str) -> None:
        _stay_in_settings()
        checked = bool(st.session_state.get(_pick_key(mid), False))
        if checked:
            st.session_state["settings_selected_models"].add(mid)
        else:
            st.session_state["settings_selected_models"].discard(mid)
        _autosave_now()

    def _selected_sorted() -> list[dict[str, str]]:
        ids = [mid for mid in st.session_state["settings_selected_models"] if mid in by_id]
        out = [by_id[mid] for mid in ids]
        out.sort(key=lambda x: (x["provider"].lower(), x["name"].lower(), x["id"].lower()))
        return out

    sel = _selected_sorted()

    st.markdown(f"**Selected ({len(sel)})**")
    if not sel:
        st.info("No models selected yet.")
    else:
        for m in sel:
            rid = m["id"]
            c1, c2, c3 = st.columns([0.70, 0.18, 0.12], vertical_alignment="center")
            with c1:
                st.markdown(f"**{m['name']}**  \n`{rid}`")
            with c2:
                st.markdown(
                    f"<div style='opacity:0.7; padding-top:4px;'>{m['provider']}</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                if st.button("Remove", key=f"settings_rm_{rid}", use_container_width=True):
                    st.session_state["settings_selected_models"].discard(rid)
                    st.session_state[_pick_key(rid)] = False
                    _autosave_now()
                    st.rerun()

        cc1, _cc2 = st.columns([0.22, 0.78], vertical_alignment="center")
        with cc1:
            if st.button("Clear all", key="settings_clear_all_models", use_container_width=True):
                _stay_in_settings()
                for rid in list(st.session_state["settings_selected_models"]):
                    st.session_state[_pick_key(rid)] = False
                st.session_state["settings_selected_models"].clear()
                _autosave_now()
                st.rerun()

    st.divider()

    st.markdown("**Browse**")
    providers = sorted({m["provider"] for m in models})
    provider_filter = st.multiselect(
        "",
        options=providers,
        default=[],
        placeholder="Filter providers",
        key="settings_provider_filter",
    )

    def matches(m: dict[str, str]) -> bool:
        if provider_filter and m["provider"] not in provider_filter:
            return False
        return True

    shown = [m for m in models if matches(m)]
    st.caption(f"Showing {len(shown)} model(s).")

    for m in shown:
        rid = m["id"]
        selected = rid in st.session_state["settings_selected_models"]
        a, b, c = st.columns([0.10, 0.70, 0.20], vertical_alignment="center")
        with a:
            st.checkbox(
                " ",
                value=selected,
                key=_pick_key(rid),
                label_visibility="collapsed",
                on_change=_sync_one,
                args=(rid,),
            )
        with b:
            st.markdown(f"**{m['name']}**  \n`{rid}`")
        with c:
            st.markdown(
                f"<div style='text-align:right; opacity:0.7; padding-top:4px;'>{m['provider']}</div>",
                unsafe_allow_html=True,
            )


# ----------------------------
# Settings: System prompt editor (autosave via registry save)
# ----------------------------
def _normalize_conditions(obj: dict) -> list[dict]:
    items = obj.get("prompt_conditions", [])
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for c in items:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("condition_id", "")).strip()
        if not cid:
            continue
        out.append(
            {
                "condition_id": cid,
                "name": str(c.get("name", cid)).strip(),
                "system_prompt": str(c.get("system_prompt", "")).strip(),
            }
        )
    return out


def _next_condition_id(existing_ids: set[str]) -> str:
    max_n = 0
    pat = re.compile(r"^cond_(\d{3})$")
    for cid in existing_ids:
        m = pat.match(cid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    n = max_n + 1
    while True:
        cand = f"cond_{n:03d}"
        if cand not in existing_ids:
            return cand
        n += 1


def _build_payload(conds: list[dict]) -> dict[str, Any]:
    final: list[dict[str, str]] = []
    for c in conds:
        cid = c["condition_id"]
        name = st.session_state.get(f"pc_name_{cid}", "").strip()
        prompt = st.session_state.get(f"pc_prompt_{cid}", "").strip()
        final.append({"condition_id": cid, "name": name if name else cid, "system_prompt": prompt})
    return {"prompt_conditions": final}


def _autosave_prompts(conds: list[dict]) -> None:
    _stay_in_settings()
    if not _throttle_ok("_autosave_pc_ts", 0.15):
        return
    save_prompt_conditions(_build_payload(conds))
    st.session_state["_reg_reload"] = True


def _render_system_prompt_editor(reg) -> None:
    st.markdown("#### System Instruction")
    st.caption("Customize the system-level instructions sent to the models.")

    if "prompt_conditions_ui" not in st.session_state:
        conds0 = _normalize_conditions(reg.prompt_conditions or {})
        conds0.sort(key=lambda c: c["condition_id"])
        st.session_state["prompt_conditions_ui"] = conds0

    conds: list[dict] = st.session_state["prompt_conditions_ui"]

    pending_del = st.session_state.pop("_pc_pending_delete", None)
    if pending_del:
        _stay_in_settings()
        cid = str(pending_del).strip()
        conds[:] = [x for x in conds if str(x.get("condition_id", "")).strip() != cid]
        st.session_state.pop(f"pc_name_{cid}", None)
        st.session_state.pop(f"pc_prompt_{cid}", None)
        if _throttle_ok("_autosave_pc_ts", 0.05):
            save_prompt_conditions(_build_payload(conds))
            st.session_state["_reg_reload"] = True
        st.rerun()

    if not conds:
        st.info("No prompt sets yet. Click Add prompt set.")
    else:
        st.subheader(f"Prompt sets ({len(conds)})")
        for i, c in enumerate(list(conds), start=1):
            cid = c["condition_id"]
            badge = f"S{i}"
            name_key = f"pc_name_{cid}"
            prompt_key = f"pc_prompt_{cid}"

            if name_key not in st.session_state:
                st.session_state[name_key] = c.get("name", "")
            if prompt_key not in st.session_state:
                st.session_state[prompt_key] = c.get("system_prompt", "")

            left, right = st.columns([0.86, 0.14], vertical_alignment="center")
            with left:
                title = (st.session_state.get(name_key) or "").strip() or "Untitled"
                st.markdown(f"### `{badge}` {title}")
            with right:
                if st.button("Remove", key=f"pc_rm_{cid}", use_container_width=True):
                    _stay_in_settings()
                    st.session_state["_pc_pending_delete"] = cid
                    st.rerun()

            with st.container(border=True):
                st.text_input("Name", key=name_key, on_change=_autosave_prompts, args=(conds,))
                st.text_area("System prompt", key=prompt_key, height=160, on_change=_autosave_prompts, args=(conds,))
            st.write("")

    st.divider()

    add_col, _ = st.columns([1, 3], vertical_alignment="center")
    with add_col:
        if st.button("Add prompt set", use_container_width=True, key="pc_add_prompt"):
            _stay_in_settings()
            existing = {c["condition_id"] for c in conds}
            new_id = _next_condition_id(existing)
            conds.append({"condition_id": new_id, "name": "New prompt set", "system_prompt": ""})
            st.session_state[f"pc_name_{new_id}"] = "New prompt set"
            st.session_state[f"pc_prompt_{new_id}"] = ""
            _autosave_prompts(conds)
            st.rerun()


# ----------------------------
# Settings: Variants editor (autosave to data/variants.json)
#   - Enabled toggle REMOVED
#   - Add/Remove keeps Settings dialog open
#   - EDITS keep Settings dialog open
# ----------------------------
DEFAULT_VARIANTS = [
    {"variant_id": "G1", "label": "Action command", "template": "Insert a {component} for “{label}”{suffix}"},
    {"variant_id": "G2", "label": "Short specification", "template": "{component}: {label}{suffix}"},
    {"variant_id": "G3", "label": "Descriptive object phrase", "template": "A {component} labeled “{label}”{suffix}"},
    {"variant_id": "G4", "label": "Build instruction", "template": "Build a {component} and label for “{label}”{suffix}"},
    {"variant_id": "G5", "label": "Standards-oriented phrasing", "template": "Accessible {component} for “{label}”{suffix}"},
]
_PREVIEW_EXAMPLE = {"component": "text field", "label": "What color is an orange?", "suffix": " with hint “orange”"}


def _normalize_variants(obj: Any) -> list[dict[str, Any]]:
    raw = obj.get("variants", []) if isinstance(obj, dict) else obj
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for v in raw:
        if not isinstance(v, dict):
            continue
        vid = str(v.get("variant_id", "")).strip()
        if not vid:
            continue
        out.append(
            {
                "variant_id": vid,
                "label": str(v.get("label", vid)).strip(),
                "template": str(v.get("template", "")).strip(),
            }
        )
    return out


def _next_variant_id(existing: list[str]) -> str:
    nums: list[int] = []
    for vid in existing:
        m = re.match(r"^G(\d+)$", str(vid).strip())
        if m:
            nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    cand = f"G{n}"
    while cand in set(existing):
        n += 1
        cand = f"G{n}"
    return cand


def _append_token(key: str, token: str) -> None:
    _stay_in_settings()
    st.session_state[key] = (st.session_state.get(key, "") + token)


def _autosave_variants(rows: list[dict[str, Any]]) -> None:
    _stay_in_settings()
    if not _throttle_ok("_autosave_variants_ts", 0.15):
        return
    payload = {
        "variants": [
            {
                "variant_id": r["variant_id"],
                "label": str(st.session_state.get(f"v_lab_{r['variant_id']}", r.get("label", ""))).strip(),
                "template": str(st.session_state.get(f"v_tpl_{r['variant_id']}", r.get("template", ""))).strip(),
            }
            for r in rows
        ]
    }
    _write_json(VARIANTS_PATH, payload)
    st.session_state["_reg_reload"] = True


def _render_variants_editor(reg) -> None:
    st.markdown("#### Variants")
    st.caption(
        "Customize the linguistic variants template. Each template must have all placeholders so components render correctly."
    )

    with st.container(border=True):
        st.markdown("**Placeholders**")
        st.markdown("- `{component}` — insert the component type (e.g., text field, radio group)")
        st.markdown("- `{label}` — insert the user-facing label or question text")
        st.markdown("- `{suffix}` — insert optional trailing details (e.g., hint text or constraints)")

    if "variants_rows_ui" not in st.session_state:
        file_obj = _read_json(VARIANTS_PATH)
        rows0 = _normalize_variants(file_obj)
        if not rows0:
            rows0 = _normalize_variants(reg.variants if isinstance(reg.variants, dict) else {})
        if not rows0:
            rows0 = [dict(x) for x in DEFAULT_VARIANTS]
        rows0.sort(key=lambda r: (r["variant_id"]))
        st.session_state["variants_rows_ui"] = rows0

    rows: list[dict[str, Any]] = st.session_state["variants_rows_ui"]

    pending_del = st.session_state.pop("_v_pending_delete", None)
    if pending_del:
        _stay_in_settings()
        vid = str(pending_del).strip()
        rows[:] = [r for r in rows if str(r.get("variant_id", "")).strip() != vid]
        st.session_state.pop(f"v_lab_{vid}", None)
        st.session_state.pop(f"v_tpl_{vid}", None)
        _autosave_variants(rows)
        st.rerun()

    st.divider()

    if not rows:
        st.info("No variants yet. Click Add variant.")
    else:
        for r in list(rows):
            vid = r["variant_id"]
            lab_key = f"v_lab_{vid}"
            tpl_key = f"v_tpl_{vid}"

            if lab_key not in st.session_state:
                st.session_state[lab_key] = r.get("label", "")
            if tpl_key not in st.session_state:
                st.session_state[tpl_key] = r.get("template", "")

            hh1, hh2 = st.columns([0.86, 0.14], vertical_alignment="center")
            with hh1:
                title = st.session_state[lab_key] or "Untitled"
                st.markdown(f"### `{vid}` {title}")
            with hh2:
                if st.button("Remove", key=f"v_del_{vid}", type="secondary", use_container_width=True):
                    _stay_in_settings()
                    st.session_state["_v_pending_delete"] = vid
                    st.rerun()

            with st.container(border=True):
                c1, c2 = st.columns([0.38, 0.62], vertical_alignment="top")
                with c1:
                    st.text_input("Name", key=lab_key, on_change=_autosave_variants, args=(rows,))
                    st.caption("Insert placeholders")
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        st.button(
                            "{component}",
                            key=f"v_ins_c_{vid}",
                            on_click=_append_token,
                            args=(tpl_key, "{component}"),
                            use_container_width=True,
                        )
                    with b2:
                        st.button(
                            "{label}",
                            key=f"v_ins_l_{vid}",
                            on_click=_append_token,
                            args=(tpl_key, "{label}"),
                            use_container_width=True,
                        )
                    with b3:
                        st.button(
                            "{suffix}",
                            key=f"v_ins_s_{vid}",
                            on_click=_append_token,
                            args=(tpl_key, "{suffix}"),
                            use_container_width=True,
                        )

                with c2:
                    st.text_area("Template", key=tpl_key, height=90, on_change=_autosave_variants, args=(rows,))
                    st.caption("Example output (fixed example)")
                    tpl = str(st.session_state.get(tpl_key, "") or "")
                    try:
                        preview = tpl.format(**_PREVIEW_EXAMPLE)
                    except Exception:
                        preview = "Invalid template (check braces / placeholder names)."
                    st.code(preview, language="text")
            st.write("")

    st.divider()

    add_col, _ = st.columns([1, 3], vertical_alignment="center")
    with add_col:
        if st.button("Add variant", use_container_width=True, key="v_add"):
            _stay_in_settings()
            vid = _next_variant_id([r["variant_id"] for r in rows])
            rows.append(
                {
                    "variant_id": vid,
                    "label": "New variant",
                    "template": "Insert a {component} for “{label}”{suffix}",
                }
            )
            st.session_state[f"v_lab_{vid}"] = "New variant"
            st.session_state[f"v_tpl_{vid}"] = "Insert a {component} for “{label}”{suffix}"
            _autosave_variants(rows)
            st.rerun()


# ----------------------------
# Settings: Components editor (autosave to data/components.json)
# ----------------------------
def _format_template_safe(tpl: str, fields: dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        return str(fields.get(key, "{" + key + "}"))

    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, tpl)


def _title_case_first(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def _normalize_tests_new_schema(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("test_id", "") or t.get("component_id", "")).strip()
        title = str(t.get("title", "")).strip()
        component = str(t.get("component", "") or t.get("component_phrase", "")).strip()
        label = str(t.get("label", "")).strip()
        suffix = str(t.get("suffix", "")).strip()

        fields = t.get("fields", {})
        if isinstance(fields, dict):
            if not component:
                component = str(fields.get("component", "") or fields.get("component_phrase", "")).strip()
            if not label:
                label = str(fields.get("label", "")).strip()
            if not suffix:
                suffix = str(fields.get("suffix", "")).strip()

        if not tid:
            continue
        if not title and component:
            title = _title_case_first(component)

        out.append({"test_id": tid, "title": title, "component": component, "label": label, "suffix": suffix})
    return out


def _next_test_id(existing: list[str]) -> str:
    nums: list[int] = []
    for tid in existing:
        m = re.match(r"^C(\d+)$", str(tid).strip())
        if m:
            nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return f"C{n}"


def _load_variants_for_generation(reg) -> list[dict[str, str]]:
    file_obj = _read_json(VARIANTS_PATH)
    rows = _normalize_variants(file_obj)
    if not rows:
        rows = _normalize_variants(reg.variants if isinstance(reg.variants, dict) else {})
    if not rows:
        rows = [dict(x) for x in DEFAULT_VARIANTS]
    rows.sort(key=lambda r: str(r.get("variant_id", "")).strip())
    out: list[dict[str, str]] = []
    for r in rows:
        out.append(
            {
                "variant_id": str(r.get("variant_id", "")).strip(),
                "label": str(r.get("label", "")).strip(),
                "template": str(r.get("template", "")).strip(),
            }
        )
    return [r for r in out if r["variant_id"]]


def _autosave_components(tests: list[dict[str, Any]]) -> None:
    _stay_in_settings()
    if not _throttle_ok("_autosave_components_ts", 0.15):
        return
    payload_now: list[dict[str, Any]] = []
    for t in tests:
        tid = str(t.get("test_id", "")).strip()
        if not tid:
            continue
        title = str(t.get("title", "")).strip()
        component = str(t.get("component", "")).strip()
        label = str(t.get("label", "")).strip()
        suffix2 = str(t.get("suffix", "")).strip()

        if not title and component:
            title = _title_case_first(component)
        if suffix2 and not suffix2.startswith(" "):
            suffix2 = " " + suffix2

        payload_now.append(
            {"test_id": tid, "title": title or tid, "component": component, "label": label, "suffix": suffix2}
        )
    _write_json(COMPONENTS_PATH, {"tests": payload_now})
    st.session_state["_reg_reload"] = True


def _render_components_editor(reg) -> None:
    st.markdown("#### Components")
    st.caption(
        "Customize test component list. Define each test component and its label text. Variants generate task prompts using this information."
    )

    variants = _load_variants_for_generation(reg)

    obj = _read_json(COMPONENTS_PATH)
    tests_raw = obj.get("tests", []) if isinstance(obj, dict) else []
    tests_clean = _normalize_tests_new_schema(tests_raw)

    if "settings_components_tests" not in st.session_state:
        st.session_state["settings_components_tests"] = [dict(x) for x in tests_clean]

    tests: list[dict[str, Any]] = st.session_state["settings_components_tests"]

    if "_components_picker_ver" not in st.session_state:
        st.session_state["_components_picker_ver"] = 0
    picker_key = f"settings_components_selectbox_v{int(st.session_state['_components_picker_ver'])}"

    if not tests:
        st.info("No tests yet. Click Add test.")
        if st.button("Add test", type="primary", key="settings_components_add_first", use_container_width=True):
            _stay_in_settings()
            tid = "C1"
            new_test = {"test_id": tid, "title": "New test", "component": "", "label": "", "suffix": ""}
            st.session_state["settings_components_tests"] = [new_test]
            st.session_state["settings_components_picked"] = tid
            st.session_state[f"sc_title_{tid}"] = "New test"
            st.session_state[f"sc_comp_{tid}"] = ""
            st.session_state[f"sc_lab_{tid}"] = ""
            st.session_state[f"sc_suf_{tid}"] = ""
            _autosave_components([new_test])
            st.session_state["_components_picker_ver"] += 1
            st.rerun()
        return

    picker_labels: list[str] = []
    label_to_tid: dict[str, str] = {}
    id_to_idx: dict[str, int] = {}

    for i, t in enumerate(tests):
        tid = str(t.get("test_id", "")).strip()
        name = (t.get("title") or "").strip() or "Untitled"
        lab = f"{tid} {name}"
        picker_labels.append(lab)
        label_to_tid[lab] = tid
        id_to_idx[tid] = i

    picked_tid = st.session_state.get("settings_components_picked")
    if not picked_tid or picked_tid not in id_to_idx:
        picked_tid = str(tests[0].get("test_id", "")).strip()
        st.session_state["settings_components_picked"] = picked_tid

    default_label = next((lab for lab in picker_labels if label_to_tid[lab] == picked_tid), picker_labels[0])
    select_index = picker_labels.index(default_label)

    picked_label = st.selectbox("Select test", options=picker_labels, index=select_index, key=picker_key)
    picked_tid = label_to_tid[picked_label]
    st.session_state["settings_components_picked"] = picked_tid

    idx = id_to_idx[picked_tid]
    test = tests[idx]

    st.divider()
    st.subheader(f"{picked_tid} {test.get('title', 'Untitled')}")

    k_title = f"sc_title_{picked_tid}"
    k_comp = f"sc_comp_{picked_tid}"
    k_lab = f"sc_lab_{picked_tid}"
    k_suf = f"sc_suf_{picked_tid}"

    if k_title not in st.session_state:
        st.session_state[k_title] = (test.get("title") or "").strip()
    if k_comp not in st.session_state:
        st.session_state[k_comp] = (test.get("component") or "").strip()
    if k_lab not in st.session_state:
        st.session_state[k_lab] = (test.get("label") or "").strip()
    if k_suf not in st.session_state:
        st.session_state[k_suf] = (test.get("suffix") or "").strip()

    def _on_components_edit() -> None:
        _stay_in_settings()
        test["title"] = (st.session_state.get(k_title, "") or "").strip()
        test["component"] = (st.session_state.get(k_comp, "") or "").strip()
        test["label"] = (st.session_state.get(k_lab, "") or "").strip()
        test["suffix"] = (st.session_state.get(k_suf, "") or "").strip()
        tests[idx] = dict(test)
        st.session_state["settings_components_tests"] = tests
        _autosave_components(tests)

    st.text_input("Title", key=k_title, placeholder="e.g., Text Field with Placeholder", on_change=_on_components_edit)
    st.text_input("Component", key=k_comp, placeholder="e.g., text field", on_change=_on_components_edit)
    st.text_input("Label", key=k_lab, placeholder="e.g., What color is an orange?", on_change=_on_components_edit)
    st.text_input("Suffix (optional)", key=k_suf, placeholder="e.g., with hint “Rex”", on_change=_on_components_edit)

    st.divider()
    st.subheader("Generated variants")

    if not variants:
        st.info("No variants available.")
    else:
        suffix = (test.get("suffix") or "").strip()
        if suffix and not suffix.startswith(" "):
            suffix = " " + suffix

        fields = {
            "component": (test.get("component") or "").strip(),
            "label": (test.get("label") or "").strip(),
            "suffix": suffix,
        }

        for v in variants:
            vid = v["variant_id"]
            vlabel = v.get("label", vid)
            tpl = v.get("template", "")
            txt = _format_template_safe(tpl, fields)

            c1, c2 = st.columns([0.25, 0.75])
            with c1:
                st.markdown(f"**{vid}**")
                st.caption(vlabel)
            with c2:
                st.code(txt, language="text")

    st.divider()
    a1, a2, _ = st.columns([1, 1, 1])

    with a1:
        if st.button("Add test", use_container_width=True, key="settings_components_add"):
            _stay_in_settings()
            existing = [str(t.get("test_id", "")).strip() for t in tests if str(t.get("test_id", "")).strip()]
            tid = _next_test_id(existing)
            new_test = {"test_id": tid, "title": "New test", "component": "", "label": "", "suffix": ""}
            tests.append(new_test)
            st.session_state["settings_components_tests"] = tests
            st.session_state["settings_components_picked"] = tid
            st.session_state[f"sc_title_{tid}"] = "New test"
            st.session_state[f"sc_comp_{tid}"] = ""
            st.session_state[f"sc_lab_{tid}"] = ""
            st.session_state[f"sc_suf_{tid}"] = ""
            _autosave_components(tests)
            st.session_state["_components_picker_ver"] += 1
            st.rerun()

    with a2:
        if st.button("Delete", use_container_width=True, key="settings_components_delete"):
            _stay_in_settings()
            tests.pop(idx)
            st.session_state["settings_components_tests"] = tests
            st.session_state.pop("settings_components_picked", None)
            _autosave_components(tests)
            st.session_state["_components_picker_ver"] += 1
            st.rerun()


# ----------------------------
# Settings: Schema editor (auto-sync components list from Components tests)
# ----------------------------
def _render_schema_editor(reg) -> None:
    st.markdown("#### Schema")
    st.caption("Define how accessibility checks map to each component. This schema determines how generated outputs are evaluated.")

    def _scale_max(score_obj: dict[str, Any]) -> int:
        try:
            return int((score_obj.get("scoring_scale") or {}).get("max", 2))
        except Exception:
            return 2

    def _get_check_defs(score_obj: dict[str, Any]) -> dict[str, Any]:
        d = score_obj.get("check_definitions") or {}
        return d if isinstance(d, dict) else {}

    def _get_components(score_obj: dict[str, Any]) -> list[dict[str, Any]]:
        comps = score_obj.get("components") or []
        return comps if isinstance(comps, list) else []

    def _all_check_ids(score_obj: dict[str, Any]) -> list[str]:
        defs = _get_check_defs(score_obj)
        ids = [str(k).strip() for k in defs.keys() if str(k).strip()]
        ids.sort()
        return ids

    def _component_label(c: dict[str, Any]) -> str:
        cid = str(c.get("id", "")).strip()
        nm = str(c.get("name", "")).strip()
        return f"{cid} {nm}" if nm else cid

    def _find_component_index(components: list[dict[str, Any]], comp_id: str) -> int:
        comp_id = (comp_id or "").strip()
        for i, c in enumerate(components):
            if str(c.get("id", "")).strip() == comp_id:
                return i
        return -1

    def _bucket_title(score_obj, key: str) -> str:
        anchors = score_obj.get("scoring_scale", {}).get("anchors", {})
        text = str(anchors.get(key, "")).strip()
        label = text
        detail = ""
        if ":" in text:
            label, detail = text.split(":", 1)
        elif "(" in text and text.endswith(")"):
            label, detail = text[:-1].split("(", 1)
        label = label.strip()
        detail = detail.strip()
        if detail:
            return f"{key} — {label}: {detail}"
        return f"{key} — {label}"

    def _clean_example_line(s: str) -> str:
        s = (s or "").strip()
        if s.startswith("- "):
            s = s[2:].strip()
        return s

    def _parse_examples(notes_raw: Any) -> dict[str, list[str]]:
        if not isinstance(notes_raw, str):
            notes_raw = ""
        lines = [_clean_example_line(ln) for ln in notes_raw.splitlines()]
        lines = [ln for ln in lines if ln]
        sections: dict[str, list[str]] = {"What it checks": [], "How it’s helpful": [], "Example": []}
        current: str | None = None
        for ln in lines:
            low = ln.lower().strip()
            if low in {"what it checks:", "what it checks"}:
                current = "What it checks"
                continue
            if low in {"how it’s helpful:", "how it's helpful:", "how it’s helpful", "how it's helpful"}:
                current = "How it’s helpful"
                continue
            if low in {"example:", "examples:", "example", "examples"}:
                current = "Example"
                continue
            if current is None:
                sections["Example"].append(ln)
            else:
                sections[current].append(ln)
        return sections

    def _load_components_tests_for_schema() -> list[dict[str, Any]]:
        obj = _read_json(COMPONENTS_PATH)
        tests_raw = obj.get("tests", []) if isinstance(obj, dict) else []
        tests = _normalize_tests_new_schema(tests_raw)

        def _sort_key(t: dict[str, Any]) -> tuple[int, str]:
            tid = str(t.get("test_id", "")).strip()
            m = re.match(r"^C(\d+)$", tid)
            return (int(m.group(1)) if m else 10**9, tid)

        tests.sort(key=_sort_key)
        return tests

    def _sync_score_components_from_tests(score_obj: dict[str, Any], tests: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        comps_old = _get_components(score_obj)
        old_by_id: dict[str, dict[str, Any]] = {}
        for c in comps_old:
            if isinstance(c, dict):
                cid = str(c.get("id", "")).strip()
                if cid:
                    old_by_id[cid] = c

        new_components: list[dict[str, Any]] = []
        for t in tests:
            tid = str(t.get("test_id", "")).strip()
            title = str(t.get("title", "")).strip() or tid
            if not tid:
                continue
            prev = old_by_id.get(tid, {})
            checks = prev.get("checks", []) if isinstance(prev, dict) else []
            if not isinstance(checks, list):
                checks = []
            checks = [str(x).strip() for x in checks if str(x).strip()]
            new_components.append({"id": tid, "name": title, "checks": checks})

        old_norm = [
            {
                "id": str(c.get("id", "")).strip(),
                "name": str(c.get("name", "")).strip(),
                "checks": sorted([str(x).strip() for x in (c.get("checks") or []) if str(x).strip()]),
            }
            for c in comps_old
            if isinstance(c, dict)
        ]
        new_norm = [
            {
                "id": c["id"],
                "name": c["name"],
                "checks": sorted([str(x).strip() for x in (c.get("checks") or []) if str(x).strip()]),
            }
            for c in new_components
        ]
        changed = json.dumps(old_norm, sort_keys=True, ensure_ascii=False) != json.dumps(new_norm, sort_keys=True, ensure_ascii=False)

        out = dict(score_obj)
        out["components"] = new_components
        return out, changed

    score_file = _read_json(SCORE_PATH)
    score = score_file if score_file else (reg.score if isinstance(reg.score, dict) else {})
    if not score:
        st.error("Could not load data/score.json or registry score.")
        return
    if _scale_max(score) != 2:
        st.warning("scoring_scale.max is not 2. This UI expects 0/1/2.")

    tests = _load_components_tests_for_schema()
    score_synced, changed = _sync_score_components_from_tests(score, tests)
    if changed:
        _write_json(SCORE_PATH, score_synced)
    score = score_synced

    check_defs = _get_check_defs(score)
    components = _get_components(score)
    check_ids = _all_check_ids(score)

    tab_map, tab_checks = st.tabs(["Assign checks to components", "Check definitions"])

    with tab_map:
        st.subheader("Assign checks to components")
        st.caption("Pick a component, then select which check IDs apply.")

        if not components:
            st.error("No components. Add tests in Components first.")
            return
        if not check_ids:
            st.error("No check_definitions found in score.json.")
            return

        labels = [_component_label(c) for c in components]
        label_to_id = {labels[i]: str(components[i].get("id", "")).strip() for i in range(len(labels))}

        if "schema_comp_id" not in st.session_state:
            st.session_state["schema_comp_id"] = str(components[0].get("id", "")).strip()

        current_id = str(st.session_state.get("schema_comp_id", "")).strip()
        if _find_component_index(components, current_id) < 0:
            current_id = str(components[0].get("id", "")).strip()
            st.session_state["schema_comp_id"] = current_id

        default_label = next((lab for lab in labels if label_to_id.get(lab, "") == current_id), labels[0])
        default_index = labels.index(default_label)

        if "_schema_ms_ver" not in st.session_state:
            st.session_state["_schema_ms_ver"] = 0

        def _on_comp_change() -> None:
            _stay_in_settings()
            st.session_state["_schema_ms_ver"] += 1

        picked_label = st.selectbox(
            "Component",
            options=labels,
            index=default_index,
            key="schema_component_picker",
            on_change=_on_comp_change,
        )
        comp_id = label_to_id[picked_label]
        st.session_state["schema_comp_id"] = comp_id

        idx = _find_component_index(components, comp_id)
        if idx < 0:
            st.error("Component not found.")
            return

        current_checks = components[idx].get("checks") or []
        if not isinstance(current_checks, list):
            current_checks = []
        current_checks = [str(x).strip() for x in current_checks if str(x).strip()]
        current_checks = [x for x in current_checks if x in set(check_ids)]

        ms_key = f"schema_checks_ms_{comp_id}_v{int(st.session_state['_schema_ms_ver'])}"

        def _save_schema_mapping_for_current() -> None:
            _stay_in_settings()
            if not _throttle_ok("_autosave_schema_map_ts", 0.15):
                return

            chosen_now = st.session_state.get(ms_key, [])
            if not isinstance(chosen_now, list):
                chosen_now = []
            chosen_now = [str(x).strip() for x in chosen_now if str(x).strip()]
            chosen_now = [x for x in chosen_now if x in set(check_ids)]

            latest = _read_json(SCORE_PATH) or dict(score)
            latest_synced, _ = _sync_score_components_from_tests(latest, tests)

            comps2 = latest_synced.get("components", [])
            if not isinstance(comps2, list):
                comps2 = []

            j = _find_component_index(comps2, comp_id)
            if j >= 0:
                comps2[j]["checks"] = chosen_now

            latest_synced["components"] = comps2
            _write_json(SCORE_PATH, latest_synced)
            st.session_state["_reg_reload"] = True

        chosen = st.multiselect(
            "Checks for this component",
            options=check_ids,
            default=current_checks,
            key=ms_key,
            on_change=_save_schema_mapping_for_current,
        )

        m1, m2 = st.columns([1, 1], vertical_alignment="center")
        with m1:
            st.metric("Selected checks", len(chosen))
        with m2:
            st.metric("Max points", len(chosen) * 2)

        cbtn1, _ = st.columns([1, 6], vertical_alignment="center")

        with st.expander("Selected check titles", expanded=False):
            if not chosen:
                st.write("—")
            else:
                for cid in chosen:
                    d = check_defs.get(cid, {}) if isinstance(check_defs.get(cid, {}), dict) else {}
                    title = str(d.get("title", "")).strip() or cid
                    st.write(f"{cid}: {title}")

    # with tab_checks:
    #     st.subheader("Check definitions")
    #     st.caption("Reference definitions for scoring rubric.")

    #     if not check_ids:
    #         st.info("No check_definitions found.")
    #         return

    #     picked_check = st.selectbox("Check ID", options=check_ids, key="schema_check_picker")
    #     d = check_defs.get(picked_check, {}) if isinstance(check_defs.get(picked_check, {}), dict) else {}

    #     title = str(d.get("title", "")).strip() or picked_check
    #     wcag = d.get("wcag") or []
    #     if isinstance(wcag, str):
    #         wcag = [wcag]
    #     if not isinstance(wcag, list):
    #         wcag = []
    #     wcag = [str(x).strip() for x in wcag if str(x).strip()]

    #     decision = d.get("decision_rules") or {}
    #     if not isinstance(decision, dict):
    #         decision = {}

    #     sections = _parse_examples(d.get("notes", ""))

    #     st.markdown(f"### {picked_check}")
    #     st.caption(title)

    #     meta_l, meta_r = st.columns([2, 3], vertical_alignment="top")

    #     with meta_l:
    #         with st.container(border=True):
    #             st.markdown("**WCAG references**")
    #             st.write(", ".join(wcag) if wcag else "—")

    #         with st.container(border=True):
    #             st.markdown("**Notes**")

    #             def _render_section(name: str, items: list[str]) -> None:
    #                 if not items:
    #                     return
    #                 st.write("")
    #                 st.markdown(f"**{name}:**")
    #                 for item in items:
    #                     if "<" in item and ">" in item:
    #                         st.code(item, language="text")
    #                     else:
    #                         st.write(item)

    #             any_content = any(len(v) for v in sections.values())
    #             if not any_content:
    #                 st.write("—")
    #             else:
    #                 _render_section("What it checks", sections["What it checks"])
    #                 _render_section("How it’s helpful", sections["How it’s helpful"])
    #                 _render_section("Example", sections["Example"])

    #     with meta_r:
    #         st.markdown("**Scoring rules**")

    #         def _show_bucket(k: str) -> None:
    #             label = _bucket_title(score, k)
    #             txt = str(decision.get(k, decision.get(int(k), "")) or "").strip()
    #             with st.container(border=True):
    #                 st.markdown(f"**{label}**")
    #                 st.code(txt if txt else "—", language="text")

    #         _show_bucket("2")
    #         _show_bucket("1")
    #         _show_bucket("0")

    with tab_checks:
        st.subheader("Check definitions")
        st.caption("Reference definitions for scoring rubric.")

        if not check_ids:
            st.info("No check_definitions found.")
            return

        picked_check = st.selectbox("Check ID", options=check_ids, key="schema_check_picker")
        d = check_defs.get(picked_check, {}) if isinstance(check_defs.get(picked_check, {}), dict) else {}

        title = str(d.get("title", "")).strip() or picked_check
        wcag = d.get("wcag") or []
        if isinstance(wcag, str):
            wcag = [wcag]
        if not isinstance(wcag, list):
            wcag = []
        wcag = [str(x).strip() for x in wcag if str(x).strip()]

        decision = d.get("decision_rules") or {}
        if not isinstance(decision, dict):
            decision = {}

        sections = _parse_examples(d.get("notes", ""))

        st.markdown(f"### {picked_check}")
        st.caption(title)

        # --- WCAG (single column, no box) ---
        st.markdown("**WCAG references**")
        st.write(", ".join(wcag) if wcag else "—")

        st.divider()

        # --- Notes (single column, minimal chrome) ---
        with st.expander("Notes", expanded=True):
            def _render_section(name: str, items: list[str]) -> None:
                if not items:
                    return
                st.write("")
                st.markdown(f"**{name}:**")
                for item in items:
                    if "<" in item and ">" in item:
                        st.code(item, language="text")
                    else:
                        st.write(item)

            any_content = any(len(v) for v in sections.values())
            if not any_content:
                st.write("—")
            else:
                _render_section("What it checks", sections["What it checks"])
                _render_section("How it’s helpful", sections["How it’s helpful"])
                _render_section("Example", sections["Example"])

        st.divider()

        # --- Scoring rules (tabs instead of 3 bordered boxes) ---
        st.markdown("**Scoring rules**")

        def _show_bucket(k: str) -> None:
            label = _bucket_title(score, k)
            txt = str(decision.get(k, decision.get(int(k), "")) or "").strip()
            st.markdown(f"**{label}**")
            st.code(txt if txt else "—", language="text")

        t2, t1, t0 = st.tabs(["2", "1", "0"])
        with t2:
            _show_bucket("2")
        with t1:
            _show_bucket("1")
        with t0:
            _show_bucket("0")

# ----------------------------
# Settings dialog
# ----------------------------
@st.dialog("Settings")
def _open_settings(reg) -> None:
    st.caption("All changes auto-update and auto-save.")
    tabs = st.tabs(["Models", "System Instructions", "Variants", "Components", "Schema", "Axe"])
    with tabs[0]:
        _render_models_editor(reg)
    with tabs[1]:
        _render_system_prompt_editor(reg)
    with tabs[2]:
        _render_variants_editor(reg)
    with tabs[3]:
        _render_components_editor(reg)
    with tabs[4]:
        _render_schema_editor(reg)
    with tabs[5]:
        _render_axe_settings()


# ----------------------------
# Load registries
# ----------------------------
ensure_defaults()
reg = load_all()

# ----------------------------
# Axe settings defaults (run-safe even if Settings was never opened)
# ----------------------------
st.session_state.setdefault("axe_profile", "form_fragment_rules")
st.session_state.setdefault("axe_config_obj", None)
st.session_state.setdefault("axe_ruleset_obj", None)
st.session_state.setdefault("axe_timeout_ms", 10_000)


# ----------------------------
# Fixed top navbar (must run BEFORE the navbar content)
# ----------------------------
st.markdown(
    """
    <style>
      /* Hide Streamlit's default header so our navbar is the only top chrome */
      header[data-testid="stHeader"]{
        height: 0px !important;
        visibility: hidden !important;
      }
      div[data-testid="stToolbar"]{
        height: 0px !important;
        visibility: hidden !important;
      }

      /*
        Pin the Streamlit block that contains our marker.
        This avoids the "open <div> then render widgets" nesting issue.
      */
      div[data-testid="stVerticalBlock"] > div:has(#app-navbar-marker){
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 1000 !important;

        background: rgba(255,255,255,0.98) !important;
        
        backdrop-filter: saturate(180%) blur(8px) !important;

        padding: 0.55rem 1.0rem !important;
        margin: 0 !important;
      }

      /* Push the main content below the fixed navbar */
      .block-container{
        padding-top: 5.3rem !important;
      }

      /* Navbar title styling */
      .app-navbar-title{
        font-size: 1.15rem;
        font-weight: 650;
        margin: 0;
        line-height: 1.25;
      }

      /* Tighten column gaps inside the fixed navbar block */
      div[data-testid="stVerticalBlock"] > div:has(#app-navbar-marker) [data-testid="column"]{
        padding-top: 0 !important;
        padding-bottom: 0 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Navbar content (logo + title + settings)
# ----------------------------
with st.container():
    # Marker used by CSS :has(...) selector to pin this entire block
    st.markdown('<div id="app-navbar-marker"></div>', unsafe_allow_html=True)

    nav_logo, nav_title, nav_btn = st.columns([0.16, 0.64, 0.20], vertical_alignment="center")

    with nav_logo:
        st.image("assets/logo.svg", width=100)

    with nav_title:
        st.markdown(
            '<div class="app-navbar-title">Prompt Accessibility Controlled Evaluation</div>',
            unsafe_allow_html=True,
        )

    with nav_btn:
        st.markdown(
            """
            <style>
            /* push container toward top-right */
            #st-key-home_open_settings {
                display: flex !important;
                justify-content: flex-end !important;
                align-items: center !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        clicked = st.button("⚙ Settings", key="home_open_settings", type="tertiary")

        # Make it look like an icon button (no border/background)
        st.markdown(
            """
            <script>
            const btn = window.parent.document.querySelector('#st-key-home_open_settings button');
            if (btn) {
                btn.style.border = 'none';
                btn.style.background = 'transparent';
                btn.style.boxShadow = 'none';
                btn.style.padding = '0';
                btn.style.fontSize = '1.35rem';
                btn.style.lineHeight = '1';
            }
            </script>
            """,
            unsafe_allow_html=True,
        )

        if clicked:
            st.session_state["_settings_open"] = True
            st.rerun()

# ----------------------------
# Front-page intro (scrolls normally under the navbar)
# ----------------------------

st.caption("Executes controlled factorial AI generations of HTML form components across models, system prompts, and linguistic variants")
st.caption("Scores each output using a normalized accessibility schema rubric and axe-core")
st.caption("Records run-level results for quantitative comparison")


# Open only on click, and ALSO keep open across edits (via _stay_in_settings())
if st.session_state.get("_settings_open", False):
    _open_settings(reg)
    st.session_state["_settings_open"] = False

# If any settings editor wrote data, reload registry for Run/Results immediately
if st.session_state.get("_reg_reload", False):
    st.session_state["_reg_reload"] = False
    ensure_defaults()
    reg = load_all()


# ----------------------------
# Inline Run + Results
# ----------------------------
st.markdown("### Benchmark workspace")
tab_run, tab_results = st.tabs(["Run", "Results"])


# ============
# Run
# ============
def _load_selected_models_for_run() -> list[dict[str, Any]]:
    obj = _read_json(SELECTED_MODELS_PATH)
    rows = obj.get("data", []) if isinstance(obj, dict) else []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            mid = str(r.get("id") or "").strip()
            if mid:
                out.append(r)
    return out


def _list_runs() -> list[Path]:
    if not RUNS_ROOT.exists():
        return []
    runs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


with tab_run:
    st.subheader("Run")
    st.caption("Uses Settings registries. This tab chooses a subset for the run.")

    ensure_defaults()
    reg = load_all()

    client = OpenRouterClient()
    api_status = client.preflight()

    if not api_status.has_key:
        st.error("OPENROUTER_API_KEY is missing.")
    if api_status.has_key and not api_status.reachable:
        st.error(f"OpenRouter unreachable: {api_status.error or 'unknown error'}")

    prompt_conditions = reg.prompt_conditions.get("prompt_conditions", []) or []
    variants = reg.variants.get("variants", []) or []
    tests = reg.components.get("tests", []) or []

    # st.markdown("### Configure")

    selected_models = _load_selected_models_for_run()
    if not selected_models:
        st.warning("No models selected. Open Settings → Models and pick at least one model.")
        st.stop()

    selected_models_sorted = sorted(
        selected_models,
        key=lambda m: (
            _provider(str(m.get("id") or "")).lower(),
            str(m.get("name") or "").lower(),
            str(m.get("id") or "").lower(),
        ),
    )

    model_label_to_id: dict[str, str] = {}
    default_labels: list[str] = []
    for m in selected_models_sorted:
        mid = str(m.get("id") or "").strip()
        name = str(m.get("name") or mid).strip()
        label = f"{name} [{mid}]"
        model_label_to_id[label] = mid
        default_labels.append(label)

    sel_model_labels = st.multiselect(
        "Models",
        options=list(model_label_to_id.keys()),
        default=default_labels,
        key="run_models",
    )
    sel_model_ids = [model_label_to_id[x] for x in sel_model_labels]
    st.caption(f"Using {len(sel_model_ids)} model(s).")

    cond_label_to_id: dict[str, str] = {}
    for c in prompt_conditions:
        cid = (c.get("condition_id") or "").strip()
        if not cid:
            continue
        name = (c.get("name") or cid).strip()
        cond_label_to_id[f"{name} [{cid}]"] = cid
    cond_labels = list(cond_label_to_id.keys())

    sel_cond_labels = st.multiselect(
        "System prompts",
        options=cond_labels,
        default=cond_labels,
        key="run_prompt_conditions",
    )
    sel_cond_ids = [cond_label_to_id[x] for x in sel_cond_labels]

    variant_label_to_id: dict[str, str] = {}
    enabled_variant_labels: list[str] = []
    for v in variants:
        vid = (v.get("variant_id") or "").strip()
        if not vid:
            continue
        lab = (v.get("label") or vid).strip()
        key = f"{lab} [{vid}]"
        variant_label_to_id[key] = vid
        enabled_variant_labels.append(key)

    variant_labels = list(variant_label_to_id.keys())
    sel_var_labels = st.multiselect(
        "Variants",
        options=variant_labels,
        default=enabled_variant_labels,
        key="run_variants",
    )
    sel_var_ids = [variant_label_to_id[x] for x in sel_var_labels]

    component_titles = [t.get("title", f"Component {i+1}") for i, t in enumerate(tests)]
    component_idx_all = list(range(len(component_titles)))

    def _format_comp(i: int) -> str:
        return component_titles[i]

    sel_components = st.multiselect(
        "Components",
        options=component_idx_all,
        default=component_idx_all,
        format_func=_format_comp,
        key="run_components",
    )

    repetitions = st.number_input("Repetitions per cell", 1, 50, 10, 1, key="run_reps")
    do_score_col, do_axe_col = st.columns([1, 1], vertical_alignment="center")

    with do_score_col:
        do_score = st.checkbox("Accessibility score (schema)", value=True, key="run_do_score")

    with do_axe_col:
        do_axe = st.checkbox("Axe score", value=True, key="run_do_axe")

    with st.expander("Advanced", expanded=False):
        # max_tokens = st.number_input("max_tokens", 200, 4000, 1200, 100, key="run_max_tokens")
        max_tokens = st.number_input(
            "Maximum response length (max tokens)",
            200,
            4000,
            600,
            100,
            key="run_max_tokens",
            help="Limits how long the model’s output can be."
        )
        # temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.05, key="run_temperature")
        temperature = st.slider(
        "Response randomness (temperature)",
            0.0,
            1.5,
            0.7,
            0.05,
            key="run_temperature",
            help="Lower = more predictable and consistent. Higher = more varied and creative."
        )
        # top_p = st.slider("Response diversity (top_p)", 0.1, 1.0, 0.95, 0.01, key="run_top_p")
        top_p = st.slider(
            "Response diversity (top_p)",
            0.1,
            1.0,
            0.95,
            0.01,
            key="run_top_p",
            help="Lower = safer and more predictable. Higher = more varied wording."
        )
        rate_limit = st.number_input("Delay between calls (sec)", 0.0, 5.0, 0.3, 0.1, key="run_rate_limit")

    run_name = st.text_input("Run name", value=f"run_{int(time.time())}", key="run_name")

    n_cells = len(sel_model_ids) * len(sel_cond_ids) * len(sel_var_ids) * len(sel_components) * int(repetitions)
    st.info(f"Planned generations: **{n_cells:,}**")

    can_run = True
    if not api_status.has_key or not api_status.reachable:
        can_run = False
    if not sel_model_ids or not sel_cond_ids or not sel_var_ids or not sel_components:
        can_run = False

    if "run_state" not in st.session_state:
        st.session_state["run_state"] = "idle"  # "idle" | "running"
    if "run_cancel" not in st.session_state:
        st.session_state["run_cancel"] = False

    def should_cancel() -> bool:
        return bool(st.session_state.get("run_cancel", False))

    st.markdown(
        """
<style>
div[data-testid="stHorizontalBlock"] { gap: 0.35rem !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) div.stButton{
  display:flex !important;
  justify-content:flex-end !important;
  margin-right:0 !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div.stButton{
  display:flex !important;
  justify-content:flex-start !important;
  margin-left:0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    cbtn1, cbtn2, _sp = st.columns([1, 1, 8], gap="small")
    start = False
    stop = False

    with cbtn1:
        if st.session_state["run_state"] == "idle":
            start = st.button("▶ Run", type="primary", disabled=not can_run, key="run_start_btn")
        else:
            st.button("▶ Run", type="primary", disabled=True, key="run_start_btn_disabled")

    with cbtn2:
        if st.session_state["run_state"] == "running":
            stop = st.button("⏹ Stop", type="secondary", key="run_stop_btn")
        else:
            st.button("⏹ Stop", type="secondary", disabled=True, key="run_stop_btn_disabled")

    if stop and st.session_state["run_state"] == "running":
        st.session_state["run_cancel"] = True
        st.warning("Stopping… will finish the current call, then stop.")
        st.rerun()

    if start:
        st.session_state["run_state"] = "running"
        st.session_state["run_cancel"] = False
        st.rerun()

    if st.session_state["run_state"] == "running":
        cfg = RunConfig(
            run_name=run_name,
            out_root=OUT_ROOT,
            model_ids=sel_model_ids,
            condition_ids=sel_cond_ids,
            variant_ids=sel_var_ids,
            component_indices=sel_components,
            repetitions=int(repetitions),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            rate_limit=float(rate_limit),
            do_score=bool(do_score),
            do_axe=bool(do_axe),
            axe_profile=str(st.session_state.get("axe_profile", "form_fragment_rules")),
            axe_config=st.session_state.get("axe_config_obj", None),
            axe_ruleset=st.session_state.get("axe_ruleset_obj", None),
            axe_timeout_ms=int(st.session_state.get("axe_timeout_ms", 10_000)),
        )

        st.caption("Run config")
        st.json(asdict(cfg), expanded=False)

        prog = st.progress(0)
        status = st.empty()

        def on_progress(done: int, total: int, msg: str) -> None:
            if total > 0:
                prog.progress(min(1.0, done / total))
            status.write(msg)

        try:
            run_dir = run_benchmark(
                reg,
                cfg,
                client=client,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )

            if should_cancel():
                st.warning(f"Run stopped. Saved to: {run_dir}")
            else:
                st.success(f"Run complete. Saved to: {run_dir}")

            st.session_state["last_run_dir"] = str(run_dir)

            res_path = Path(run_dir) / "results.csv"
            per_path = Path(run_dir) / "per_check.csv"
            cdl1, cdl2 = st.columns(2)
            with cdl1:
                if res_path.exists():
                    st.download_button("Download results.csv", data=res_path.read_bytes(), file_name="results.csv")
            with cdl2:
                if per_path.exists():
                    st.download_button("Download per_check.csv", data=per_path.read_bytes(), file_name="per_check.csv")
        finally:
            st.session_state["run_state"] = "idle"
            st.session_state["run_cancel"] = False

# ===============
# Results
# ===============
with tab_results:
    st.subheader("Results")

    def _need_altair() -> None:
        if alt is None:
            st.error("Altair is not installed. Run: pip install altair")
            st.stop()

    def _need_numpy() -> None:
        if np is None:
            st.error("NumPy is not installed. Run: pip install numpy")
            st.stop()

    _need_altair()
    _need_numpy()

    # ----------------------------
    # Global Altair styling
    # ----------------------------
    Y01_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def _title_center(text: str) -> alt.TitleParams:
        return alt.TitleParams(text=text, anchor="middle")


    def _axis_cat(title: str, angle: int = -35) -> alt.Axis:
        return alt.Axis(
          title=title,
          labelAngle=angle,
          labelLimit=2000,      # was 300
          labelPadding=10,      # a bit more breathing room
          labelOverlap=False,   # don't auto-drop labels
        )

    def _chart_pad(top: int = 40, right: int = 12, bottom: int = 120, left: int = 12) -> dict:
        return {"top": top, "right": right, "bottom": bottom, "left": left}
    # def _base_config(chart: alt.Chart) -> alt.Chart:
    #     return (
    #         chart.configure_title(anchor="middle", align="center", fontSize=16, offset=14)
    #         .configure_axis(
    #             titleFontSize=13,
    #             labelFontSize=12,
    #             grid=True,
    #             tickSize=4,
    #             titlePadding=10,
    #             labelPadding=6,
    #             labelLimit=300,
    #         )
    #         .configure_legend(titleFontSize=12, labelFontSize=12, orient="bottom", padding=10)
    #         .configure_view(stroke=None)
    #     )

    # def _axis_cat(title: str, angle: int = -35) -> alt.Axis:
    #     return alt.Axis(title=title, labelAngle=angle, labelLimit=300, labelOverlap="greedy")

    def _base_config(chart: alt.Chart) -> alt.Chart:
        return (
            chart.configure_title(anchor="middle", align="center", fontSize=16, offset=14)
            .configure_axis(
                titleFontSize=13,
                labelFontSize=12,
                grid=True,
                tickSize=4,
                titlePadding=10,
                labelPadding=6,
                labelLimit=300,
            )
            .configure_legend(titleFontSize=12, labelFontSize=12, orient="bottom", padding=10)
            .configure_view(stroke=None)
        )
    


    def _axis_y01(title: str) -> alt.Axis:
        return alt.Axis(title=title, values=Y01_TICKS, format=".1f", grid=True, labelOverlap=False, labelFlush=False)

    def _y_enc01(field: str, title: str) -> alt.Y:
        return alt.Y(
            f"{field}:Q",
            title=title,
            scale=alt.Scale(domain=[0.0, 1.0], zero=True, nice=False),
            axis=_axis_y01(title),
        )

    def _chart_pad(top: int = 40, right: int = 12, bottom: int = 55, left: int = 12) -> dict:
        return {"top": top, "right": right, "bottom": bottom, "left": left}

    def _apply_layout(chart: alt.Chart, title: str, *, pad: Optional[dict] = None) -> alt.Chart:
        return chart.properties(title=_title_center(title), padding=(pad or _chart_pad()))

    # ----------------------------
    # Utilities
    # ----------------------------
    def _ensure_trial_column(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "trial" in df.columns:
            return df
        if "rep_idx" in df.columns:
            return df.rename(columns={"rep_idx": "trial"})
        return df

    def _coerce_ok_bool(df: pd.DataFrame) -> pd.DataFrame:
        if "ok" not in df.columns:
            return df
        if df["ok"].dtype == bool:
            return df
        s = df["ok"].astype(str).str.strip().str.lower()
        df["ok"] = s.map(
            {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "t": True, "f": False}
        )
        return df

    def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _strip_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df

    def _apply_multiselect(df: pd.DataFrame, col: str, label: str, default_all: bool = True) -> pd.DataFrame:
        if col not in df.columns:
            return df
        opts = df[col].dropna().astype(str).str.strip()
        opts = [x for x in opts.unique().tolist() if x != ""]
        opts = sorted(opts, key=lambda x: x)
        if not opts:
            return df
        default = opts if default_all else []
        picked = st.multiselect(label, options=opts, default=default)
        if not picked:
            return df.iloc[0:0]
        picked_set = set(picked)
        return df[df[col].astype(str).str.strip().isin(picked_set)]

    def run_kpis(res_df: pd.DataFrame) -> dict[str, str]:
        if res_df is None or res_df.empty:
            return {"rows": "0", "pct_ok": "0.0%", "mean_norm_ok": "-", "total_cost": "0.0000"}
        rows = int(len(res_df))
        ok_df = res_df
        if "ok" in res_df.columns:
            ok_df = res_df[res_df["ok"] == True]  # noqa: E712
        pct_ok = (len(ok_df) / rows) if rows else 0.0

        mean_norm_ok = "-"
        if not ok_df.empty and "norm_score" in ok_df.columns:
            vals = pd.to_numeric(ok_df["norm_score"], errors="coerce").dropna()
            if not vals.empty:
                mean_norm_ok = f"{float(vals.mean()):.3f}"

        total_cost = "0.0000"
        if "cost" in res_df.columns:
            vals = pd.to_numeric(res_df["cost"], errors="coerce").dropna()
            total_cost = f"{float(vals.sum()):.4f}" if not vals.empty else "0.0000"

        return {"rows": str(rows), "pct_ok": f"{pct_ok*100:.1f}%", "mean_norm_ok": mean_norm_ok, "total_cost": total_cost}

    def _mean_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        tmp = df.copy()
        tmp["norm_score"] = pd.to_numeric(tmp["norm_score"], errors="coerce")
        tmp = tmp.dropna(subset=["norm_score"])
        if tmp.empty:
            return pd.DataFrame()

        g = tmp.groupby(group_cols)["norm_score"].agg(["mean", "std", "count"]).reset_index()
        g = g.rename(columns={"count": "n"})
        g["std"] = g["std"].fillna(0.0)

        g["sd_lo"] = g["mean"] - g["std"]
        g["sd_hi"] = g["mean"] + g["std"]

        g["se"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
        g["se_lo"] = g["mean"] - g["se"]
        g["se_hi"] = g["mean"] + g["se"]

        g["ci95"] = 1.96 * g["se"]
        g["lo"] = g["mean"] - g["ci95"]
        g["hi"] = g["mean"] + g["ci95"]
        return g

    def _mean_stats_value(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
        tmp = df.copy()
        if value_col not in tmp.columns:
            return pd.DataFrame()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna(subset=[value_col])
        if tmp.empty:
            return pd.DataFrame()

        g = tmp.groupby(group_cols)[value_col].agg(["mean", "std", "count"]).reset_index()
        g = g.rename(columns={"count": "n"})
        g["std"] = g["std"].fillna(0.0)

        g["sd_lo"] = g["mean"] - g["std"]
        g["sd_hi"] = g["mean"] + g["std"]

        g["se"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
        g["se_lo"] = g["mean"] - g["se"]
        g["se_hi"] = g["mean"] + g["se"]

        g["ci95"] = 1.96 * g["se"]
        g["lo"] = g["mean"] - g["ci95"]
        g["hi"] = g["mean"] + g["ci95"]
        return g

    def _build_name_maps_from_registry(reg) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        model_map: dict[str, str] = {}
        models = (reg.models or {}).get("data", []) if isinstance(reg.models, dict) else []
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict):
                    mid = str(m.get("id") or "").strip()
                    nm = str(m.get("name") or mid).strip()
                    if mid:
                        model_map[mid] = nm or mid

        cond_map: dict[str, str] = {}
        conds = (reg.prompt_conditions or {}).get("prompt_conditions", []) if isinstance(reg.prompt_conditions, dict) else []
        if isinstance(conds, list):
            for c in conds:
                if isinstance(c, dict):
                    cid = str(c.get("condition_id") or "").strip()
                    nm = str(c.get("name") or cid).strip()
                    if cid:
                        cond_map[cid] = nm or cid

        var_map: dict[str, str] = {}
        vars_ = (reg.variants or {}).get("variants", []) if isinstance(reg.variants, dict) else []
        if isinstance(vars_, list):
            for v in vars_:
                if isinstance(v, dict):
                    vid = str(v.get("variant_id") or "").strip()
                    lab = str(v.get("label") or vid).strip()
                    if vid:
                        var_map[vid] = lab or vid

        return model_map, cond_map, var_map

    def _ensure_display_columns(df: pd.DataFrame, reg) -> pd.DataFrame:
        model_map, cond_map, var_map = _build_name_maps_from_registry(reg)
        out = df.copy()

        if "model_name" not in out.columns:
            if "model_id" in out.columns:
                out["model_name"] = out["model_id"].astype(str).map(
                    lambda x: model_map.get(str(x).strip(), str(x).strip())
                )
        else:
            out["model_name"] = out["model_name"].astype(str).str.strip()

        if "condition_name" not in out.columns:
            if "condition_id" in out.columns:
                out["condition_name"] = out["condition_id"].astype(str).map(
                    lambda x: cond_map.get(str(x).strip(), str(x).strip())
                )
        else:
            out["condition_name"] = out["condition_name"].astype(str).str.strip()

        if "variant_label" not in out.columns:
            if "variant_id" in out.columns:
                out["variant_label"] = out["variant_id"].astype(str).map(
                    lambda x: var_map.get(str(x).strip(), str(x).strip())
                )
        else:
            out["variant_label"] = out["variant_label"].astype(str).str.strip()

        return out

    COLOR_SCHEMES = ["tableau10", "set2", "dark2", "category10", "accent"]
    DEFAULT_SCHEME = "tableau10"

    CAP_SIZE = 10
    CAP_THICKNESS = 2
    RULE_THICKNESS = 2

    BAR_OPACITY = 0.92
    BAR_RADIUS = 4

    def _center_block():
        return st.columns([1, 10, 1])

    def _pretty_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
        styler = df.style.format(precision=3)
        styler = styler.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center"), ("font-weight", "600")]},
                {"selector": "td", "props": [("text-align", "center")]},
                {"selector": "table", "props": [("width", "100%")]},
            ]
        )
        return styler

    def _error_caps_layer(
        base: alt.Chart,
        x_enc: alt.X,
        y_lo_field: str,
        y_hi_field: str,
        x_offset_enc: Optional[alt.XOffset] = None,
        color_enc: Optional[Any] = None,
    ) -> alt.LayerChart:
        enc_common: dict[str, Any] = {"x": x_enc}
        if x_offset_enc is not None:
            enc_common["xOffset"] = x_offset_enc
        if color_enc is not None:
            enc_common["color"] = color_enc

        rule = base.mark_rule(size=RULE_THICKNESS).encode(
            **enc_common,
            y=alt.Y(f"{y_lo_field}:Q"),
            y2=alt.Y2(f"{y_hi_field}:Q"),
        )
        tick_lo = base.mark_tick(size=CAP_SIZE, thickness=CAP_THICKNESS, orient="horizontal").encode(
            **enc_common,
            y=alt.Y(f"{y_lo_field}:Q"),
        )
        tick_hi = base.mark_tick(size=CAP_SIZE, thickness=CAP_THICKNESS, orient="horizontal").encode(
            **enc_common,
            y=alt.Y(f"{y_hi_field}:Q"),
        )
        return rule + tick_lo + tick_hi

    def _interval_fields(error_mode: str) -> tuple[str, str, str]:
        if error_mode == "sd":
            return ("sd_lo", "sd_hi", "SD")
        if error_mode == "se":
            return ("se_lo", "se_hi", "SE")
        if error_mode == "ci95":
            return ("lo", "hi", "95% CI")
        return ("", "", "")

    def _grouped_bar_with_error(
        stats: pd.DataFrame,
        x_col: str,
        series_col: Optional[str],
        title: str,
        error_mode: str,
        scheme: str,
        height: int = 420,
        x_title: Optional[str] = None,
        series_title: Optional[str] = None,
        y_title: str = "Accessibility score (normalized)",
        label_angle: int = -35,
        extra_tooltips: Optional[list[alt.Tooltip]] = None,
    ) -> None:
        if stats.empty:
            st.caption("No data.")
            return

        pad = _chart_pad(top=46, bottom=(95 if abs(label_angle) >= 20 else 65), left=14, right=14)
        base = alt.Chart(stats).properties(height=height)

        axis_x = _axis_cat(title=(x_title or x_col), angle=label_angle)
        lo_f, hi_f, err_lab = _interval_fields(error_mode)

        def _tt_common() -> list[alt.Tooltip]:
            tts = [
                alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                alt.Tooltip("std:Q", title="SD", format=".3f"),
                alt.Tooltip("se:Q", title="SE", format=".3f"),
                alt.Tooltip("ci95:Q", title="95% CI half-width", format=".3f"),
                alt.Tooltip("n:Q", title="n"),
            ]
            if err_lab and lo_f and hi_f:
                tts.insert(1, alt.Tooltip(f"{lo_f}:Q", title=f"{err_lab} low", format=".3f"))
                tts.insert(2, alt.Tooltip(f"{hi_f}:Q", title=f"{err_lab} high", format=".3f"))
            if extra_tooltips:
                tts = extra_tooltips + tts
            return tts

        if series_col:
            x_enc = alt.X(
                f"{x_col}:N",
                title=(x_title or x_col),
                axis=axis_x,
                scale=alt.Scale(paddingInner=0.25, paddingOuter=0.15),
            )
            xoff = alt.XOffset(f"{series_col}:N")

            col_enc = alt.Color(
                f"{series_col}:N",
                title=(series_title or series_col),
                scale=alt.Scale(scheme=scheme),
                legend=alt.Legend(orient="bottom"),
            )

            bars = base.mark_bar(
                opacity=BAR_OPACITY,
                cornerRadiusTopLeft=BAR_RADIUS,
                cornerRadiusTopRight=BAR_RADIUS,
            ).encode(
                x=x_enc,
                xOffset=xoff,
                y=_y_enc01("mean", y_title),
                color=col_enc,
                tooltip=[
                    alt.Tooltip(f"{x_col}:N", title=(x_title or x_col)),
                    alt.Tooltip(f"{series_col}:N", title=(series_title or series_col)),
                    *_tt_common(),
                ],
            )

            if error_mode != "none" and lo_f and hi_f:
                cap_color = alt.Color(f"{series_col}:N", scale=alt.Scale(scheme=scheme), legend=None)
                caps = _error_caps_layer(
                    base=base,
                    x_enc=alt.X(f"{x_col}:N", axis=axis_x, scale=alt.Scale(paddingInner=0.25, paddingOuter=0.15)),
                    y_lo_field=lo_f,
                    y_hi_field=hi_f,
                    x_offset_enc=xoff,
                    color_enc=cap_color,
                )
                chart = _apply_layout(bars + caps, title, pad=pad)
                st.altair_chart(_base_config(chart), use_container_width=True)
                return

            chart = _apply_layout(bars, title, pad=pad)
            st.altair_chart(_base_config(chart), use_container_width=True)
            return

        x_enc = alt.X(
            f"{x_col}:N",
            title=(x_title or x_col),
            axis=axis_x,
            scale=alt.Scale(paddingInner=0.25, paddingOuter=0.15),
        )

        color_by_category = alt.Color(
            f"{x_col}:N",
            title=None,
            scale=alt.Scale(scheme=scheme),
            legend=None,
        )

        bars = base.mark_bar(
            opacity=BAR_OPACITY,
            cornerRadiusTopLeft=BAR_RADIUS,
            cornerRadiusTopRight=BAR_RADIUS,
        ).encode(
            x=x_enc,
            y=_y_enc01("mean", y_title),
            color=color_by_category,
            tooltip=[alt.Tooltip(f"{x_col}:N", title=(x_title or x_col)), *_tt_common()],
        )

        if error_mode != "none" and lo_f and hi_f:
            caps = _error_caps_layer(
                base=base,
                x_enc=alt.X(f"{x_col}:N", axis=axis_x, scale=alt.Scale(paddingInner=0.25, paddingOuter=0.15)),
                y_lo_field=lo_f,
                y_hi_field=hi_f,
                color_enc=color_by_category,
            )
            chart = _apply_layout(bars + caps, title, pad=pad)
            st.altair_chart(_base_config(chart), use_container_width=True)
            return

        chart = _apply_layout(bars, title, pad=pad)
        st.altair_chart(_base_config(chart), use_container_width=True)

    def _safe_slider_topk(n_items: int, default_k: int = 20, cap: int = 60) -> int:
        if n_items <= 0:
            return 0
        if n_items <= 100:
            return n_items
        lo = 5
        hi = min(cap, n_items)
        return int(st.slider("Components shown (top K by mean)", lo, hi, min(default_k, hi)))

    st.caption("Loads results.csv from last run by default.")

    source = st.radio(
        "Results source",
        ["Last run", "Pick a run folder", "Upload results.csv"],
        horizontal=True,
        index=0,
        key="results_source",
    )

    csv_bytes: Optional[bytes] = None
    csv_label: str = ""

    if source == "Last run":
        last = st.session_state.get("last_run_dir")
        if last:
            p = Path(last) / "results.csv"
            if p.exists():
                csv_bytes = p.read_bytes()
                csv_label = str(p)
            else:
                st.warning(f"Last run has no results.csv: {p}")
        else:
            st.info("No last run yet. Run something first, or pick a run folder, or upload results.csv.")

    elif source == "Pick a run folder":
        runs = _list_runs()
        if not runs:
            st.info("No runs found under output/_runs/.")
        else:
            labels = [
                f"{p.name}  ·  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))}"
                for p in runs
            ]
            label_to_path = {labels[i]: runs[i] for i in range(len(runs))}
            pick = st.selectbox("Run folder", options=labels, index=0, key="results_pick_run")
            rp = label_to_path[pick]
            p = rp / "results.csv"
            if p.exists():
                csv_bytes = p.read_bytes()
                csv_label = str(p)
            else:
                st.error(f"Missing results.csv in: {rp}")

    else:
        up = st.file_uploader("Upload results.csv", type=["csv"], accept_multiple_files=False, key="results_uploader")
        if up is not None:
            csv_bytes = up.getvalue()
            csv_label = up.name

    if not csv_bytes:
        st.stop()

    try:
        res_df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    except Exception as e:
        st.error(f"Could not read CSV ({csv_label}): {e}")
        st.stop()

    res_df = _ensure_trial_column(res_df)
    res_df = _strip_text(
        res_df,
        [
            "model_id",
            "model_name",
            "condition_id",
            "condition_name",
            "variant_id",
            "variant_label",
            "component_id",
            "component_title",
        ],
    )
    res_df = _coerce_ok_bool(res_df)
    res_df = _coerce_numeric(res_df, ["trial", "norm_score", "axe_score", "cost", "raw_ core", "max_score"])

    ensure_defaults()
    reg = load_all()
    res_df = _ensure_display_columns(res_df, reg)

    if res_df.empty:
        st.info("CSV is empty.")
        st.stop()

    k = run_kpis(res_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", k["rows"])
    c2.metric("% ok", k["pct_ok"])
    c3.metric("Mean norm_score (ok)", k["mean_norm_ok"])
    c4.metric("Total cost", k["total_cost"])

    st.divider()
    st.subheader("Filter")
    df = res_df.copy()

    if "ok" in df.columns:
        ok_only = st.checkbox("Only ok rows", value=True, key="results_ok_only")
        if ok_only:
            df = df[df["ok"] == True]  # noqa: E712

    left, right = st.columns(2)
    with left:
        df = _apply_multiselect(df, "model_name" if "model_name" in df.columns else "model_id", "Model", default_all=True)
        df = _apply_multiselect(
            df,
            "condition_name" if "condition_name" in df.columns else "condition_id",
            "System prompts",
            default_all=True,
        )
    with right:
        df = _apply_multiselect(df, "variant_label" if "variant_label" in df.columns else "variant_id", "Variant", default_all=True)
        df = _apply_multiselect(df, "component_title", "Component", default_all=True)

    st.caption(f"Filtered rows: {len(df)}")
    if ("model_id" in df.columns) and (df["model_id"].notna().any()):
        st.caption(f"Unique models after filters: {int(df['model_id'].nunique(dropna=True))}")

    if df.empty:
        st.warning("Nothing left after filters.")
        st.stop()

    st.divider()
    st.subheader("Plot options")
    scheme = st.selectbox(
        "Color scheme",
        options=COLOR_SCHEMES,
        index=COLOR_SCHEMES.index(DEFAULT_SCHEME),
        key="results_color_scheme",
    )
    error_mode = st.radio("Error bars", ["sd", "se", "ci95", "none"], horizontal=True, index=0, key="results_error_mode")

    st.divider()

    df_ok = df
    if "ok" in df_ok.columns:
        df_ok = df_ok[df_ok["ok"] == True]  # noqa: E712

    # ----------------------------
    # Tabs: accessibility score vs axe score
    # ----------------------------
    tab_acc, tab_axe = st.tabs(["Accessibility score (schema)", "Axe score"])

    # ============================
    # Accessibility score (norm_score) — EXACT SAME CONTENT
    # ============================
   
    with tab_acc:
        st.subheader("Accessibility Score (schema)")
        st.caption(
            "Each component has K rubric checks scored 0–2. "
            "Raw score = sum of check scores. "
            "Normalized score = raw score / (2 × K), giving a value between 0 and 1."
        )
        
        st.divider()
        
        need_schema = {"norm_score", "raw_score", "max_score"}
        if not need_schema.issubset(set(df_ok.columns)):
            st.info(
                "No schema score columns found in results.csv. "
            )
        else:
            st.subheader("Consistency across models")
            st.caption("Accessibility differences across models.")

            if ("model_name" in df_ok.columns) and ("norm_score" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                s_model = _mean_stats(df_ok, ["model_name"]).sort_values("mean", ascending=False)
                _grouped_bar_with_error(
                    s_model,
                    x_col="model_name",
                    series_col=None,
                    title="Accessibility score by model",
                    error_mode=error_mode,
                    scheme=scheme,
                    x_title="Model",
                    y_title="Accessibility score (normalized)",
                    label_angle=-35,
                    height=480,
                )
                st.dataframe(_pretty_table(s_model), use_container_width=True, height=240)
            else:
                st.caption("Need ≥2 models and columns: model_name (or model_id) and norm_score.")

            st.divider()

            st.subheader("System Instructions")
            st.caption("Effect of system prompt on accessibility. ")

            rq1_split = st.checkbox("Split by model", value=False, key="rq1_split_models")

            need_rq1 = {"condition_name", "norm_score"}
            if need_rq1.issubset(set(df_ok.columns)):
                lp, mid, rp = _center_block()
                with mid:
                    if rq1_split and ("model_name" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                        s = _mean_stats(df_ok, ["condition_name", "model_name"])
                        _grouped_bar_with_error(
                            s,
                            x_col="condition_name",
                            series_col="model_name",
                            title="Accessibility score by prompt condition (grouped by model)",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Prompt condition",
                            series_title="Model",
                            y_title="Accessibility score (normalized)",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "Prompt condition", "model_name": "Model"})
                        st.dataframe(_pretty_table(tbl.sort_values(["Prompt condition", "Model"])), use_container_width=True, height=260)
                    else:
                        s = _mean_stats(df_ok, ["condition_name"])
                        _grouped_bar_with_error(
                            s,
                            x_col="condition_name",
                            series_col=None,
                            title="Accessibility score by prompt condition",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Prompt condition",
                            y_title="Accessibility score (normalized)",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "Prompt condition"})
                        st.dataframe(_pretty_table(tbl.sort_values("mean", ascending=False)), use_container_width=True, height=260)
            else:
                st.caption("Need condition_name and norm_score.")

            st.divider()

            st.subheader("Linguistic Variants ")
            st.caption("Effect of linguistic phrasing on accessibility. ")

            rq2_split = st.checkbox("Split by model", value=False, key="rq2_split_models")

            need_rq2 = {"variant_label", "norm_score"}
            if need_rq2.issubset(set(df_ok.columns)):
                if rq2_split and ("model_name" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                    s = _mean_stats(df_ok, ["variant_label", "model_name"])
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col="model_name",
                        title="Accessibility score by variant (grouped by model)",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="Variant",
                        series_title="Model",
                        height=480,
                    )
                    st.dataframe(_pretty_table(s.sort_values(["variant_label", "model_name"])), use_container_width=True, height=300)
                else:
                    s = _mean_stats(df_ok, ["variant_label"])
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col=None,
                        title="Accessibility score by variant (overall)",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="Variant",
                        height=480,
                    )
                    st.dataframe(_pretty_table(s.sort_values("mean", ascending=False)), use_container_width=True, height=300)
            else:
                st.caption("Need variant_label and norm_score.")

            st.divider()

            st.subheader("Prompt × Variant Interaction")
            st.caption("Joint effect of system prompts and linguistic variants. .")

            need = {"condition_name", "variant_label", "norm_score"}
            if need.issubset(set(df_ok.columns)):
                s = _mean_stats(df_ok, ["condition_name", "variant_label"])
                if s.empty:
                    st.caption("No data.")
                else:
                    heat = (
                        alt.Chart(s)
                        .mark_rect()
                        .encode(
                            x=alt.X("variant_label:N", title="Variant", axis=_axis_cat("Variant", angle=-35)),
                            y=alt.Y("condition_name:N", title="Prompt condition", axis=_axis_cat("Prompt condition", angle=0)),
                            color=alt.Color(
                                "mean:Q",
                                title="Mean accessibility score",
                                scale=alt.Scale(scheme="viridis", domain=[0.0, 1.0]),
                            ),
                            tooltip=[
                                alt.Tooltip("condition_name:N", title="Prompt condition"),
                                alt.Tooltip("variant_label:N", title="Variant"),
                                alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                                alt.Tooltip("std:Q", title="SD", format=".3f"),
                                alt.Tooltip("se:Q", title="SE", format=".3f"),
                                alt.Tooltip("ci95:Q", title="95% CI half-width", format=".3f"),
                                alt.Tooltip("n:Q", title="n"),
                            ],
                        )
                        .properties(height=420)
                    )
                    heat = _apply_layout(heat, "Condition × variant interaction", pad=_chart_pad(top=46, bottom=70, left=12, right=12))
                    st.altair_chart(_base_config(heat), use_container_width=True)
                    st.dataframe(_pretty_table(s.sort_values(["condition_name", "variant_label"])), use_container_width=True, height=280)
            else:
                st.caption("Need condition_name, variant_label, norm_score.")

            st.divider()

            st.subheader("Component Accessibility ")
            st.caption("Accessibility differences across HTML form components.")

            if "component_title" in df_ok.columns and "norm_score" in df_ok.columns:
                tmp = df_ok.copy()
                tmp["norm_score"] = pd.to_numeric(tmp["norm_score"], errors="coerce")
                tmp = tmp.dropna(subset=["norm_score"])
                if tmp.empty:
                    st.caption("No data.")
                else:
                    comp_means = tmp.groupby("component_title")["norm_score"].mean().sort_values(ascending=False)
                    top_k = _safe_slider_topk(int(len(comp_means)), default_k=20, cap=60)
                    keep = set(comp_means.head(top_k).index.tolist())
                    tmp = tmp[tmp["component_title"].isin(keep)]
                    rq4_split = st.checkbox("Split by model", value=False, key="rq4_split_checkbox")

                    if rq4_split and ("model_name" in tmp.columns) and (tmp["model_name"].nunique() > 1):
                        s = _mean_stats(tmp, ["component_title", "model_name"])
                        order = comp_means.head(top_k).index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values(["component_title", "model_name"])
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col="model_name",
                            title="Accessibility score by component (grouped by model)",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Component",
                            series_title="Model",
                            y_title="Accessibility score (normalized)",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(s), use_container_width=True, height=320)
                    else:
                        s = _mean_stats(tmp, ["component_title"])
                        order = comp_means.head(top_k).index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values("component_title")
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col=None,
                            title="Accessibility score by component",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Component",
                            y_title="Accessibility score (normalized)",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(s.sort_values("mean", ascending=False)), use_container_width=True, height=320)
            else:
                st.caption("Need component_title and norm_score.")

            st.divider()
            
            st.subheader("Data preview")
            st.dataframe(df.head(400), use_container_width=True, height=380)

            st.divider()
            
    # ============================
    # Axe score (axe_score)
    # ============================
    with tab_axe:
        st.subheader("Axe score")
        
        st.divider()
        
        st.caption(
            "Normalized strict pass rate: passes / (passes + violations + incomplete), giving a value between 0 and 1. "
            "Inapplicable rules are excluded."
        )

        if "axe_score" not in df_ok.columns:
            st.info("No axe_score column found in results.csv.")
        else:         

            st.subheader("Consistency across models")
            st.caption("Axe score differences across models.")

            if ("model_name" in df_ok.columns) and ("axe_score" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                s_model = _mean_stats_value(df_ok, ["model_name"], "axe_score").sort_values("mean", ascending=False)
                _grouped_bar_with_error(
                    s_model,
                    x_col="model_name",
                    series_col=None,
                    title="Axe score by model",
                    error_mode=error_mode,
                    scheme=scheme,
                    x_title="Model",
                    y_title="Axe score (0–1)",
                    label_angle=-35,
                    height=480,
                )
                st.dataframe(_pretty_table(s_model), use_container_width=True, height=240)
            else:
                st.caption("Need ≥2 models and columns: model_name (or model_id) and axe_score.")

            st.divider()
            
            st.subheader("System Instructions ")
            st.caption("Effect of system prompt framing. ")

            axe_rq1_split = st.checkbox("Split by model", value=False, key="axe_rq1_split_models")

            need_rq1_axe = {"condition_name", "axe_score"}
            if need_rq1_axe.issubset(set(df_ok.columns)):
                lp, mid, rp = _center_block()
                with mid:
                    if axe_rq1_split and ("model_name" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                        s = _mean_stats_value(df_ok, ["condition_name", "model_name"], "axe_score")
                        _grouped_bar_with_error(
                            s,
                            x_col="condition_name",
                            series_col="model_name",
                            title="Axe score by prompt condition (grouped by model)",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Prompt condition",
                            series_title="Model",
                            y_title="Axe score",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "Prompt condition", "model_name": "Model"})
                        st.dataframe(_pretty_table(tbl.sort_values(["Prompt condition", "Model"])), use_container_width=True, height=260)
                    else:
                        s = _mean_stats_value(df_ok, ["condition_name"], "axe_score")
                        _grouped_bar_with_error(
                            s,
                            x_col="condition_name",
                            series_col=None,
                            title="Axe score by prompt condition",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Prompt condition",
                            y_title="Axe score (0–1)",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "Prompt condition"})
                        st.dataframe(_pretty_table(tbl.sort_values("mean", ascending=False)), use_container_width=True, height=260)
            else:
                st.caption("Need condition_name and axe_score.")

            st.divider()

            st.subheader("Linguistic Variants ")
            st.caption("Effect of linguistic phrasing. ")

            axe_rq2_split = st.checkbox("Split by model", value=False, key="axe_rq2_split_models")

            need_rq2_axe = {"variant_label", "axe_score"}
            if need_rq2_axe.issubset(set(df_ok.columns)):
                if axe_rq2_split and ("model_name" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                    s = _mean_stats_value(df_ok, ["variant_label", "model_name"], "axe_score")
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col="model_name",
                        title="Axe score by variant (grouped by model)",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="Variant",
                        series_title="Model",
                        y_title="Axe score (0–1)",
                        height=480,
                    )
                    st.dataframe(_pretty_table(s.sort_values(["variant_label", "model_name"])), use_container_width=True, height=300)
                else:
                    s = _mean_stats_value(df_ok, ["variant_label"], "axe_score")
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col=None,
                        title="Axe score by variant (overall)",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="Variant",
                        y_title="Axe score (0–1)",
                        height=480,
                    )
                    st.dataframe(_pretty_table(s.sort_values("mean", ascending=False)), use_container_width=True, height=300)
            else:
                st.caption("Need variant_label and axe_score.")

            st.divider()

            st.subheader("Prompt × Variant Interaction")
            st.caption("Joint effect of system prompts and linguistic variants. .")

            need_axe = {"condition_name", "variant_label", "axe_score"}
            if need_axe.issubset(set(df_ok.columns)):
                s = _mean_stats_value(df_ok, ["condition_name", "variant_label"], "axe_score")
                if s.empty:
                    st.caption("No data.")
                else:
                    heat = (
                        alt.Chart(s)
                        .mark_rect()
                        .encode(
                            x=alt.X("variant_label:N", title="Variant", axis=_axis_cat("Variant", angle=-35)),
                            y=alt.Y("condition_name:N", title="Prompt condition", axis=_axis_cat("Prompt condition", angle=0)),
                            color=alt.Color(
                                "mean:Q",
                                title="Mean axe score",
                                scale=alt.Scale(scheme="viridis", domain=[0.0, 1.0]),
                            ),
                            tooltip=[
                                alt.Tooltip("condition_name:N", title="Prompt condition"),
                                alt.Tooltip("variant_label:N", title="Variant"),
                                alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                                alt.Tooltip("std:Q", title="SD", format=".3f"),
                                alt.Tooltip("se:Q", title="SE", format=".3f"),
                                alt.Tooltip("ci95:Q", title="95% CI half-width", format=".3f"),
                                alt.Tooltip("n:Q", title="n"),
                            ],
                        )
                        .properties(height=420)
                    )
                    heat = _apply_layout(heat, "Condition × variant interaction (axe_score)", pad=_chart_pad(top=46, bottom=70, left=12, right=12))
                    st.altair_chart(_base_config(heat), use_container_width=True)
                    st.dataframe(_pretty_table(s.sort_values(["condition_name", "variant_label"])), use_container_width=True, height=280)
            else:
                st.caption("Need condition_name, variant_label, axe_score.")

            st.divider()

            st.subheader("Component ")
            st.caption("Axe score differences across HTML form components.")

            if "component_title" in df_ok.columns and "axe_score" in df_ok.columns:
                tmp = df_ok.copy()
                tmp["axe_score"] = pd.to_numeric(tmp["axe_score"], errors="coerce")
                tmp = tmp.dropna(subset=["axe_score"])
                if tmp.empty:
                    st.caption("No data.")
                else:
                    comp_means = tmp.groupby("component_title")["axe_score"].mean().sort_values(ascending=False)
                    top_k = _safe_slider_topk(int(len(comp_means)), default_k=20, cap=60)
                    keep = set(comp_means.head(top_k).index.tolist())
                    tmp = tmp[tmp["component_title"].isin(keep)]
                    axe_rq4_split = st.checkbox("Split by model", value=False, key="axe_rq4_split_checkbox")

                    if axe_rq4_split and ("model_name" in tmp.columns) and (tmp["model_name"].nunique() > 1):
                        s = _mean_stats_value(tmp, ["component_title", "model_name"], "axe_score")
                        order = comp_means.head(top_k).index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values(["component_title", "model_name"])
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col="model_name",
                            title="Axe score by component (grouped by model)",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Component",
                            series_title="Model",
                            y_title="Axe score (0–1)",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(s), use_container_width=True, height=320)
                    else:
                        s = _mean_stats_value(tmp, ["component_title"], "axe_score")
                        order = comp_means.head(top_k).index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values("component_title")
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col=None,
                            title="Axe score by component",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="Component",
                            y_title="Axe score (0–1)",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(s.sort_values("mean", ascending=False)), use_container_width=True, height=320)
            else:
                st.caption("Need component_title and axe_score.")

            st.divider()
            
            st.subheader("Data preview")
            st.dataframe(df.head(400), use_container_width=True, height=380)
            
            st.divider()
            
st.caption("Runs save under output/_runs/<run_name>/. Results loads that run’s results.csv (and optional per_check.csv for schema).")
