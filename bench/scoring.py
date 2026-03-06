from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


@dataclass
class CheckScore:
    check_id: str
    score: int
    rationale: str


# ---------------------------
# Meta contract
# ---------------------------
#
# scoring.py is meta-first. String-based inference is OFF by default.
# Opt in per-variant via component_meta['allow_component_inference']=True.
EXPECTED_META_KEYS = {
    "test_id",
    "title",
    "label",
    "component",
    "suffix",
    "allow_component_inference",
    "forbid_script_style",
    "primary_control_id",
    "expected_primary_kind",
    "input_type_required",
    "expected_input_types",
    "expected_autocomplete_tokens",
    "description_ids",
    "required_expected",
    "pattern_expected",
    "pattern_guidance_ids",
    "error_ids",
    "group_validation_expected",
    "group_description_expected",
    "status_ids",
    "group_required",
    "group_container_id",
    "radio_group_expected",
    "option_inputs_expected",
    "disabled_expected",
    "readonly_expected",
    "file_input_expected",
    "file_required_attributes",
    "button_expected",
    "expected_button_type",
    "expected_accept_tokens",
    "required_symbol_explanation_required",
}


# ---------------------------
# Meta derivation (deterministic, from components.json fields)
# ---------------------------

NOTE_RE = re.compile(r'with note [“"](.*?)[”"]', re.I)
ERROR_RE = re.compile(r'with a validation error [“"](.*?)[”"]', re.I)
PLACEHOLDER_RE = re.compile(r'with placeholder [“"](.*?)[”"]', re.I)
HINT_RE = re.compile(r'with hint [“"](.*?)[”"]', re.I)
AUTOCOMP_RE = re.compile(r'autocomplete\s*=\s*([a-zA-Z0-9\-_]+)', re.I)
OPTIONS_RE = re.compile(r'with options\s+(.+)$', re.I)
ACCEPTING_RE = re.compile(r'accepting\s+(.+)$', re.I)
RANGE_RE = re.compile(r'with range input\s+([0-9]+)\s*[–-]\s*([0-9]+)', re.I)
CHARCOUNT_RE = re.compile(r'character counter\s*\(max\s*([0-9]+)\s*characters?\)', re.I)


def _norm_test_id(v: str) -> str:
    """
    Normalize test ids like:
      - "C4"  -> "c04"
      - "C04" -> "c04"
      - "c04" -> "c04"
      - "c004" -> "c04"
      - anything else -> lowercased trimmed
    """
    s = (v or "").strip().lower()
    if not s:
        return ""
    m = re.fullmatch(r"c\s*0*([0-9]+)", s)
    if m:
        n = int(m.group(1))
        return f"c{n:02d}"
    return s


def _maybe_set(meta: Dict[str, Any], key: str, val: Any) -> None:
    if key not in meta or meta.get(key) in (None, "", [], {}):
        meta[key] = val


def _meta_list(meta: Dict[str, Any], key: str) -> List[str]:
    v = meta.get(key)
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for x in v:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out


def _is_group_component(comp_lower: str) -> bool:
    """
    Group-ish components where the "relevant target" may be a group container.
    """
    c = comp_lower or ""
    return (
        "radio group" in c
        or "checkbox group" in c
        or "toggle switch group" in c
        or "group of toggle switches" in c
    )


def _derive_meta_from_components_fields(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic meta generation when you pass components.json fields through meta.

    Expected inputs inside meta_in (any subset):
      - test_id (e.g., "C6" / "C06" / "c06")
      - component (e.g., "required email field")
      - suffix (the exact suffix string from components.json)
      - title, label (optional; not required for ids)

    Output:
      - fills canonical ids like cXX-control / cXX-desc / cXX-error / cXX-status / cXX-group
      - fills flags used by checks (required_expected, pattern_expected, etc.)
      - fills expected_primary_kind, expected_input_types, expected_autocomplete_tokens, etc.

    IMPORTANT:
      - This only works if your prompt enforces the same canonical ids in the generated HTML.
      - It never overwrites explicit values you already provided (component_meta wins).
      - It is meta-first: string-based inference is OFF by default. Opt in per-variant with
        allow_component_inference=True if you want fallback inference.
    """
    meta = dict(meta_in or {})

    tid = _norm_test_id(str(meta.get("test_id") or ""))
    comp = str(meta.get("component") or "").strip().lower()
    suffix = str(meta.get("suffix") or "").strip()
    title = str(meta.get("title") or "").strip().lower()

    # When True, infer semantics (types/kinds/flags) from component/title strings.
    # Default False aligns with score.json implementation_notes.meta_source_of_truth.
    allow_component_inference = bool(meta.get("allow_component_inference") is True)

    if tid:
        base = tid  # e.g., "c06"
        _maybe_set(meta, "primary_control_id", f"{base}-control")

    # ---- primary kind (native element) ----
    if allow_component_inference:
        if "textarea" in comp:
            _maybe_set(meta, "expected_primary_kind", "textarea")
        elif "select" in comp:
            _maybe_set(meta, "expected_primary_kind", "select")
        elif "button" in comp:
            _maybe_set(meta, "expected_primary_kind", "button")
            _maybe_set(meta, "button_expected", True)
            if "submit" in comp:
                _maybe_set(meta, "expected_button_type", "submit")
            if "reset" in comp:
                _maybe_set(meta, "expected_button_type", "reset")
        else:
            _maybe_set(meta, "expected_primary_kind", "input")

    # ---- required ----
    if allow_component_inference:
        if "required" in comp:
            _maybe_set(meta, "required_expected", True)

    # ---- autocomplete ----
    m = AUTOCOMP_RE.search(suffix)
    if m:
        tok = m.group(1).strip()
        _maybe_set(meta, "expected_autocomplete_tokens", [tok])

    # ---- description/note -> cX-desc ----
    m_note = NOTE_RE.search(suffix)
    if m_note and tid:
        _maybe_set(meta, "description_ids", [f"{tid}-desc"])

        # If the component is a group and has a note, description association should
        # target the group container (fieldset/role=group/cXX-group), not an individual option.
        if allow_component_inference and _is_group_component(comp):
            _maybe_set(meta, "group_description_expected", True)

    # ---- errors -> cX-error + status ----
    m_err = ERROR_RE.search(suffix)
    if m_err and tid:
        _maybe_set(meta, "error_ids", [f"{tid}-error"])
        _maybe_set(meta, "status_ids", [f"{tid}-status"])

        # Group validation targeting for group components with validation error
        if allow_component_inference and _is_group_component(comp):
            _maybe_set(meta, "group_validation_expected", True)

    # ---- placeholder / hint ----
    m_ph = PLACEHOLDER_RE.search(suffix) or HINT_RE.search(suffix)
    if m_ph:
        _maybe_set(meta, "expected_placeholder", m_ph.group(1).strip())

    # ---- pattern ----
    if allow_component_inference:
        # Make this deterministic even if the caller doesn't pass title:
        # - If title indicates pattern OR test_id is C04 (canonical c04), set pattern_expected.
        if title.endswith("with pattern") or tid == "c04":
            _maybe_set(meta, "pattern_expected", True)
            if tid:
                _maybe_set(meta, "pattern_guidance_ids", [f"{tid}-desc"])

    # ---- character counter/status -> cX-status ----
    m_cc = CHARCOUNT_RE.search(suffix)
    if m_cc and tid:
        _maybe_set(meta, "status_ids", [f"{tid}-status"])

    # ---- groups (radio/checkbox/toggle switch group) ----
    if allow_component_inference:
        if "radio group" in comp:
            _maybe_set(meta, "radio_group_expected", True)
            _maybe_set(meta, "option_inputs_expected", True)
            _maybe_set(meta, "group_required", True)
            if tid:
                _maybe_set(meta, "group_container_id", f"{tid}-group")

        if "checkbox group" in comp:
            _maybe_set(meta, "option_inputs_expected", True)
            _maybe_set(meta, "group_required", True)
            if tid:
                _maybe_set(meta, "group_container_id", f"{tid}-group")

        # "group of toggle switches" behaves like a group for scoring purposes
        if "toggle switch group" in comp or "group of toggle switches" in comp:
            _maybe_set(meta, "option_inputs_expected", True)
            _maybe_set(meta, "group_required", True)
            if tid:
                _maybe_set(meta, "group_container_id", f"{tid}-group")

    # ---- file upload ----
    if allow_component_inference:
        if "file upload" in comp:
            _maybe_set(meta, "file_input_expected", True)
            _maybe_set(meta, "expected_primary_kind", "input")
            _maybe_set(meta, "expected_input_types", ["file"])
            _maybe_set(meta, "input_type_required", True)

            if "multiple file upload" in comp:
                _maybe_set(meta, "file_required_attributes", ["multiple"])

            m_acc = ACCEPTING_RE.search(suffix)
            if m_acc:
                raw = m_acc.group(1).strip()
                toks = [t.strip() for t in re.split(r"\s*,\s*", raw) if t.strip()]
                _maybe_set(meta, "expected_accept_tokens", toks)
                req = list(_meta_list(meta, "file_required_attributes"))
                if "accept" not in [x.lower() for x in req]:
                    req.append("accept")
                meta["file_required_attributes"] = req

    # ---- single checkbox + toggle switch ----
    if allow_component_inference:
        # C27/C28 ("checkbox") and C54/C55 ("toggle switch") must be enforced as native input[type=checkbox]
        if comp == "checkbox" or comp == "toggle switch":
            _maybe_set(meta, "expected_primary_kind", "input")
            _maybe_set(meta, "expected_input_types", ["checkbox"])
            _maybe_set(meta, "input_type_required", True)

        # ---- input[type] expectations (deterministic, based on component string) ----
        if "email field" in comp:
            _maybe_set(meta, "expected_input_types", ["email"])
            _maybe_set(meta, "input_type_required", True)

        if "phone number field" in comp:
            _maybe_set(meta, "expected_input_types", ["tel"])
            _maybe_set(meta, "input_type_required", True)

        if comp.startswith("url field") or "url field" in comp:
            _maybe_set(meta, "expected_input_types", ["url"])
            _maybe_set(meta, "input_type_required", True)

        if "search field" in comp:
            _maybe_set(meta, "expected_input_types", ["search"])
            _maybe_set(meta, "input_type_required", True)

        if "password field" in comp:
            _maybe_set(meta, "expected_input_types", ["password"])
            _maybe_set(meta, "input_type_required", True)

        if "number field" in comp:
            _maybe_set(meta, "expected_input_types", ["number"])
            _maybe_set(meta, "input_type_required", True)

        if "range slider" in comp:
            _maybe_set(meta, "expected_input_types", ["range"])
            _maybe_set(meta, "input_type_required", True)

        if "color input" in comp or "color picker" in comp:
            _maybe_set(meta, "expected_input_types", ["color"])
            _maybe_set(meta, "input_type_required", True)

        if "date field" in comp:
            _maybe_set(meta, "expected_input_types", ["date"])
            _maybe_set(meta, "input_type_required", True)

        if "time field" in comp and "datetime-local" not in comp:
            _maybe_set(meta, "expected_input_types", ["time"])
            _maybe_set(meta, "input_type_required", True)

        if "datetime-local" in comp:
            _maybe_set(meta, "expected_input_types", ["datetime-local"])
            _maybe_set(meta, "input_type_required", True)

        if "credit card field" in comp:
            _maybe_set(meta, "expected_input_types", ["text"])
            _maybe_set(meta, "input_type_required", True)

        if "disabled" in comp:
            _maybe_set(meta, "disabled_expected", True)

        if "read-only" in comp or "readonly" in comp:
            _maybe_set(meta, "readonly_expected", True)

    return meta


# ---------------------------
# Parsing helpers
# ---------------------------

def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _is_htmlish(html: str) -> bool:
    if "```" in html or "~~~" in html:
        return False
    return bool(re.search(r"<[a-zA-Z][^>]*>", html))


def _nonempty_text(t) -> bool:
    if t is None:
        return False
    return bool((t.get_text(" ", strip=True) or "").strip())


def _all_ids(s: BeautifulSoup) -> List[str]:
    return [t.get("id") for t in s.find_all(attrs={"id": True}) if t.get("id")]


def _id_set(s: BeautifulSoup) -> set:
    return set(_all_ids(s))


def _primary_controls(s: BeautifulSoup) -> List[Any]:
    return s.find_all(["input", "select", "textarea", "button"])


def _primary_control(s: BeautifulSoup, meta: Dict[str, Any]) -> Optional[Any]:
    pcid = (meta.get("primary_control_id") or "").strip()
    if pcid:
        t = s.find(id=pcid)
        if t and t.name in {"input", "select", "textarea", "button"}:
            return t
    ctrls = _primary_controls(s)
    return ctrls[0] if ctrls else None


def _aria_describedby_ids(control_tag) -> List[str]:
    ref = control_tag.get("aria-describedby")
    if not ref:
        return []
    return [x for x in ref.split() if x]


def _aria_labelledby_ids(control_tag) -> List[str]:
    ref = control_tag.get("aria-labelledby")
    if not ref:
        return []
    return [x for x in ref.split() if x]


def _aria_errormessage_ids(control_tag) -> List[str]:
    ref = control_tag.get("aria-errormessage")
    if not ref:
        return []
    return [x for x in ref.split() if x]


def _label_for_text(s: BeautifulSoup, control_id: str) -> Optional[str]:
    lab = s.find("label", attrs={"for": control_id})
    if not lab:
        return None
    return lab.get_text(" ", strip=True) or ""


def _implicit_label_text(control_tag) -> Optional[str]:
    lab = control_tag.find_parent("label")
    if not lab:
        return None
    return lab.get_text(" ", strip=True) or ""


def _labelledby_text(s: BeautifulSoup, control_tag) -> str:
    ids = _aria_labelledby_ids(control_tag)
    if not ids:
        return ""
    out: List[str] = []
    for rid in ids:
        t = s.find(id=rid)
        if t:
            txt = t.get_text(" ", strip=True)
            if txt:
                out.append(txt)
    return " ".join(out).strip()


def _has_required_tokens(s: BeautifulSoup) -> bool:
    txt = s.get_text(" ", strip=True)
    if re.search(r"\brequired\b", txt, re.I):
        return True
    if "*" in txt:
        return True
    return False


def _has_star(s: BeautifulSoup) -> bool:
    return "*" in s.get_text(" ", strip=True)


def _has_required_word(s: BeautifulSoup) -> bool:
    return bool(re.search(r"\brequired\b", s.get_text(" ", strip=True), re.I))


def _meta_true(meta: Dict[str, Any], key: str) -> bool:
    return bool(meta.get(key) is True)


def _otherwise_2_if_not_applicable(applicable: bool) -> Tuple[bool, Optional[Tuple[int, str]]]:
    if applicable:
        return True, None
    return False, (2, "Not applicable; score=2 by rule.")


def _assoc_refs_with_source(s: BeautifulSoup) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for lab in s.find_all("label", attrs={"for": True}):
        v = (lab.get("for") or "").strip()
        if v:
            out.append((v, "label_for"))
    for t in s.find_all(attrs={"aria-labelledby": True}):
        for rid in (t.get("aria-labelledby") or "").split():
            rid = rid.strip()
            if rid:
                out.append((rid, "aria-labelledby"))
    for t in s.find_all(attrs={"aria-describedby": True}):
        for rid in (t.get("aria-describedby") or "").split():
            rid = rid.strip()
            if rid:
                out.append((rid, "aria-describedby"))
    for t in s.find_all(attrs={"aria-errormessage": True}):
        for rid in (t.get("aria-errormessage") or "").split():
            rid = rid.strip()
            if rid:
                out.append((rid, "aria-errormessage"))
    return out


def _expected_primary_kind(meta: Dict[str, Any]) -> str:
    v = (meta.get("expected_primary_kind") or "").strip().lower()
    return v if v in {"input", "select", "textarea", "button"} else ""


def _expects_input_types(meta: Dict[str, Any]) -> List[str]:
    return [x.lower() for x in _meta_list(meta, "expected_input_types")]


def _expected_input_type_required(meta: Dict[str, Any]) -> bool:
    return bool(meta.get("input_type_required") is True)


def _has_conflicting_role_or_state_on_native(control_tag) -> bool:
    if control_tag is None:
        return False
    return control_tag.has_attr("role")


def _any_non_native_control_emulation(s: BeautifulSoup) -> bool:
    fake_roles = {
        "button", "checkbox", "radio", "switch", "textbox", "combobox",
        "listbox", "slider", "spinbutton", "searchbox",
    }
    for t in s.find_all(attrs={"role": True}):
        role = (t.get("role") or "").strip().lower()
        if role in fake_roles and t.name not in {"input", "select", "textarea", "button"}:
            return True
    return False


def _find_group_container(s: BeautifulSoup, meta: Dict[str, Any]):
    gid = (meta.get("group_container_id") or "").strip()
    if gid:
        t = s.find(id=gid)
        if t:
            return t
    fs = s.find("fieldset")
    if fs:
        return fs
    g = s.find(attrs={"role": re.compile(r"^group$", re.I)})
    if g:
        return g
    return None


def _relevant_target_for_association(s: BeautifulSoup, meta: Dict[str, Any]) -> Optional[Any]:
    # FIX: use group container for either group-level validation or group-level description
    if bool(meta.get("group_validation_expected") is True) or bool(meta.get("group_description_expected") is True):
        return _find_group_container(s, meta)
    return _primary_control(s, meta)


def _any_wrapper_disabled_or_aria_disabled(s: BeautifulSoup, control: Any) -> bool:
    for t in s.find_all(True):
        if t is control:
            continue
        if t.has_attr("disabled"):
            return True
        if (t.get("aria-disabled") or "").strip().lower() == "true":
            return True
    return False


def _any_wrapper_readonly_or_aria_readonly(s: BeautifulSoup, control: Any) -> bool:
    for t in s.find_all(True):
        if t is control:
            continue
        if t.has_attr("readonly"):
            return True
        if (t.get("aria-readonly") or "").strip().lower() == "true":
            return True
    return False


def _any_misplaced_aria_invalid(s: BeautifulSoup, target: Any) -> bool:
    for t in s.find_all(attrs={"aria-invalid": True}):
        if t is target:
            continue
        return True
    return False


def _has_required_conflict_on_control(c: Any) -> bool:
    if c is None:
        return False
    ar = (c.get("aria-required") or "").strip().lower()
    if c.has_attr("required") and ar and ar != "true":
        return True
    if ar == "false" and c.has_attr("required"):
        return True
    return False


def _normalize_generic_error_text(tx: str) -> str:
    t = (tx or "").strip().lower()
    t = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------
# Public API
# ---------------------------

def score_checks(
    html: str,
    component_checks: List[str],
    score_spec: Dict[str, Any],
    component_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, List[CheckScore]]:
    """
    Returns: (raw_sum, max_sum, per_check_scores)

    Deterministic meta:
      - If you pass components.json fields inside component_meta (test_id/component/suffix/title),
        this function derives canonical ids + expectations automatically.
      - Explicit meta values you pass still win (we only fill missing keys).
    """
    defs = score_spec.get("check_definitions", {}) or {}
    scale_max = int((score_spec.get("scoring_scale", {}) or {}).get("max", 2))

    meta0 = component_meta or {}
    meta = _derive_meta_from_components_fields(meta0)

    s = _soup(html)
    per: List[CheckScore] = []

    for cid in component_checks:
        if cid not in defs:
            per.append(CheckScore(cid, 0, "Check not defined in score.json."))
            continue

        fn = CHECK_IMPL.get(cid)
        if not fn:
            per.append(CheckScore(cid, 0, "Check not implemented in scoring.py."))
            continue

        sc, why = fn(html, s, meta)
        sc = int(max(0, min(scale_max, sc)))
        per.append(CheckScore(cid, sc, why))

    raw = sum(x.score for x in per)
    mx = scale_max * len(component_checks)
    return raw, mx, per


# ---------------------------
# Check implementations (0..2)
# ---------------------------

def DOC_VALID_HTML(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    if not _is_htmlish(html):
        return 0, "Not HTML-only or contains markdown fences / no tags."
    if s.find() is None:
        return 0, "Parser found no elements."

    forbid = bool(meta.get("forbid_script_style") is True)
    if s.find(["script", "style"]):
        if forbid:
            return 0, "Contains forbidden <script>/<style> under constraints."
        return 1, "Contains <script>/<style>."

    return 2, "Parseable HTML fragment; no forbidden elements."


def ID_UNIQUENESS(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    ids = _all_ids(s)
    if not ids:
        return 2, "No ids present."
    seen = set()
    dup = set()
    for x in ids:
        if x in seen:
            dup.add(x)
        seen.add(x)
    if not dup:
        return 2, "All ids unique."

    refs = [rid for rid, _src in _assoc_refs_with_source(s)]
    if any(d in refs for d in dup):
        return 0, f"Duplicate ids used in associations: {sorted(list(dup))[:5]}"
    return 1, f"Duplicate ids exist but not referenced: {sorted(list(dup))[:5]}"


def REF_TARGET_EXISTS(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    ids = _id_set(s)
    refs = _assoc_refs_with_source(s)
    if not refs:
        return 2, "No association references used."

    missing: List[Tuple[str, str]] = []
    empty_name: List[str] = []

    for rid, src in refs:
        if rid not in ids:
            missing.append((rid, src))
            continue
        if src == "aria-labelledby":
            t = s.find(id=rid)
            if not _nonempty_text(t):
                empty_name.append(rid)
        if src == "aria-errormessage":
            t = s.find(id=rid)
            if not _nonempty_text(t):
                empty_name.append(rid)

    missing_ids = sorted(set(r for r, _ in missing))
    empty_ids = sorted(set(empty_name))
    unique_ref_ids = sorted(set(r for r, _ in refs))

    if not missing_ids and not empty_ids:
        return 2, "All association references resolve and referenced naming/error targets have non-empty text."

    if missing_ids and set(missing_ids) == set(unique_ref_ids):
        return 0, f"All association refs missing: {missing_ids[:5]}"

    parts: List[str] = []
    if missing_ids:
        parts.append(f"missing ids: {missing_ids[:5]}")
    if empty_ids:
        parts.append(f"referenced targets empty: {empty_ids[:5]}")
    return 1, "; ".join(parts)


def NAME_ROLE_VALUE_NATIVE(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    c = _primary_control(s, meta)
    if not c:
        if _any_non_native_control_emulation(s):
            return 0, "No native primary form control; found non-native role-based emulation."
        return 0, "No native primary form control exists."

    if _has_conflicting_role_or_state_on_native(c):
        return 1, "Primary native form control has a role override."

    expected_kind = _expected_primary_kind(meta)
    if expected_kind and c.name != expected_kind:
        return 0, f"Primary control kind mismatch (expected {expected_kind}, found {c.name})."

    if _meta_true(meta, "file_input_expected"):
        if c.name != "input":
            return 0, f"File input expected but primary control is {c.name}."
        t = (c.get("type") or "").strip().lower()
        if t != "file":
            return 0, f"File input expected but input[type] is {t or '(missing)'}."
        return 2, "Primary control uses native input[type=file] and no role override."

    if _meta_true(meta, "radio_group_expected") or _meta_true(meta, "option_inputs_expected"):
        radios = s.find_all("input", attrs={"type": re.compile(r"^radio$", re.I)})
        checks = s.find_all("input", attrs={"type": re.compile(r"^checkbox$", re.I)})
        if _meta_true(meta, "radio_group_expected") and len(radios) < 2:
            if _any_non_native_control_emulation(s):
                return 0, "Radio group expected but native radio inputs missing; found role-based emulation."
            return 0, "Radio group expected but native radio inputs missing (<2)."
        if _meta_true(meta, "option_inputs_expected") and len(radios) + len(checks) == 0:
            if _any_non_native_control_emulation(s):
                return 0, "Option inputs expected but missing; found role-based emulation."
            return 0, "Option inputs expected but none found."
        return 2, "Native option inputs present and no role override on primary control."

    if c.name == "input":
        expected_types = _expects_input_types(meta)
        t_raw = (c.get("type") or "").strip().lower()
        if _expected_input_type_required(meta) and not t_raw:
            return 1, "input[type] required by variant but missing on primary control."
        if expected_types and t_raw and t_raw not in set(expected_types):
            return 1, f"Native input present but input[type] not aligned with expected types: {t_raw}"
        return 2, "Primary native form control present with no role override."

    return 2, "Primary native form control present with no role override."


def LABEL_ASSOCIATION(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    c = _primary_control(s, meta)
    if not c:
        if s.find("label") or s.find(attrs={"aria-labelledby": True}):
            return 1, "Label/aria-labelledby exists somewhere but no primary control found."
        return 0, "No primary control and no labeling mechanism present."

    cid = (c.get("id") or "").strip()
    if cid:
        t = _label_for_text(s, cid)
        if t is not None:
            return (2, "<label for> association with non-empty text.") if t.strip() else (1, "<label for> exists but label text empty.")

    imp = _implicit_label_text(c)
    if imp is not None:
        return (2, "Implicit label nesting with non-empty text.") if imp.strip() else (1, "Implicit label nesting but text empty.")

    labelledby_ids = _aria_labelledby_ids(c)
    if labelledby_ids:
        txt = _labelledby_text(s, c)
        if txt.strip():
            return 2, "aria-labelledby yields non-empty label text."
        if any(s.find(id=rid) is not None for rid in labelledby_ids):
            return 1, "aria-labelledby targets exist but contribute no text."
        return 1, "aria-labelledby present but referenced ids are missing."

    if s.find("label") or s.find(attrs={"aria-labelledby": True}):
        return 1, "Label/aria-labelledby exists but not a valid association on primary control."
    return 0, "No labeling mechanism present."


def LABEL_VISIBLE_OR_EQUIV(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    text_sources: List[str] = []
    cid = (c.get("id") or "").strip()
    if cid:
        t = _label_for_text(s, cid)
        if t is not None:
            text_sources.append(t)
    imp = _implicit_label_text(c)
    if imp is not None:
        text_sources.append(imp)
    lb = _labelledby_text(s, c)
    if lb:
        text_sources.append(lb)

    if any((x or "").strip() for x in text_sources):
        return 2, "Non-empty visible label/labelledby text exists."

    aria_label = (c.get("aria-label") or "").strip()
    placeholder = (c.get("placeholder") or "").strip()

    if placeholder and not aria_label:
        return 0, "No label text; placeholder appears to be the only labeling mechanism."

    if aria_label:
        return 1, "Uses aria-label as the only name source (no visible label text)."
    return 0, "No non-empty label text, labelledby text, or aria-label."


def DESCRIPTION_ASSOCIATION(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    desc_ids_expected = _meta_list(meta, "description_ids")
    applicable = len(desc_ids_expected) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    target = _relevant_target_for_association(s, meta)
    if not target:
        return 0, "No relevant target found (primary control or group container)."

    describedby = _aria_describedby_ids(target)
    if not describedby:
        return 0, "aria-describedby missing on relevant target."

    missing = [rid for rid in describedby if s.find(id=rid) is None]
    if missing:
        return 1, f"aria-describedby includes missing ids: {sorted(set(missing))[:5]}"

    expected_set = set(desc_ids_expected)
    if not any(rid in expected_set for rid in describedby):
        return 1, "aria-describedby present but includes none of declared description ids."

    any_nonempty = False
    any_exists = False
    for did in desc_ids_expected:
        t = s.find(id=did)
        if t is not None:
            any_exists = True
            if _nonempty_text(t):
                any_nonempty = True
                break

    if not any_exists:
        return 1, "Declared description_ids provided but no matching elements exist in fragment."
    if not any_nonempty:
        return 1, "Declared description element(s) exist but text is empty/whitespace."
    return 2, "aria-describedby includes declared description id and description text is non-empty."


def REQUIRED_STATE(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "required_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    if _has_required_conflict_on_control(c):
        return 1, "Required encoding present but conflicting (required with aria-required not 'true')."

    if c.has_attr("required"):
        return 2, "Primary control uses required attribute."
    if (c.get("aria-required") or "").strip().lower() == "true":
        return 2, "Primary control uses aria-required='true'."

    if _has_required_tokens(s):
        return 1, "Required indicated by token/symbol but not encoded on primary control."
    return 0, "No required encoding and no required token/symbol."


def REQUIRED_CUE_EXPLAINED(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "required_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    if not _has_star(s):
        return 2, "Required field encoded; no '*' symbol used."

    def _star_hidden() -> bool:
        for t in s.find_all(True):
            txt = (t.get_text("", strip=True) or "")
            if "*" in txt:
                if (t.get("aria-hidden") or "").strip().lower() == "true":
                    return True
        return False

    hidden = _star_hidden()

    explanation_required = bool(meta.get("required_symbol_explanation_required") is True)
    has_required = _has_required_word(s)

    if explanation_required and not has_required:
        return 1, "Symbol '*' present but explanation text ('required') is missing under constraints."

    if hidden:
        return 2, "Symbol '*' present and hidden from AT via aria-hidden."
    return 1, "Symbol '*' present but not hidden from AT (recommend aria-hidden='true')."


def AUTOCOMPLETE_PRESENT(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    expected = _meta_list(meta, "expected_autocomplete_tokens")
    applicable = len(expected) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    if not c.has_attr("autocomplete"):
        return 0, "autocomplete missing."

    val = (c.get("autocomplete") or "").strip()
    if not val or val.lower() in {"on", "off"}:
        return 1, "autocomplete present but empty or on/off."
    return 2, f"autocomplete present: {val}"


def AUTOCOMPLETE_VALID_TOKEN(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    expected = _meta_list(meta, "expected_autocomplete_tokens")
    applicable = len(expected) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    if not c.has_attr("autocomplete"):
        return 0, "autocomplete missing."

    val = (c.get("autocomplete") or "").strip()
    if not val:
        return 1, "autocomplete empty/whitespace-only."
    if val.lower() in {"on", "off"}:
        return 1, f"autocomplete is on/off: {val}"

    if val in set(expected):
        return 2, f"autocomplete matches expected token: {val}"
    return 0, f"autocomplete present but not in expected tokens: {val}"


def INPUT_TYPE_MATCHES_PURPOSE(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    expected_types = [x.lower() for x in _meta_list(meta, "expected_input_types")]
    applicable = len(expected_types) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    if c.name != "input":
        return 0, f"Primary control is not an input (found {c.name})."

    t_raw = (c.get("type") or "").strip().lower()
    t = t_raw if t_raw else "text"

    if t in set(expected_types):
        return 2, f"input[type] matches expected: {t}"

    if t == "text" and "text" not in set(expected_types):
        return 1, "input[type] is 'text' (explicit or implicit) as a generic fallback."

    return 0, f"input[type] does not match expected types: {t} (expected one of {expected_types})"


def CONSTRAINT_PATTERN_EXPLAINED(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "pattern_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    if not c.has_attr("pattern"):
        return 0, "pattern_expected true but primary control has no pattern attribute."

    guidance_ids = _meta_list(meta, "pattern_guidance_ids")
    if len(guidance_ids) == 0:
        return 0, "pattern present but pattern_guidance_ids missing/empty."

    describedby = _aria_describedby_ids(c)
    if not describedby:
        return 1, "pattern present and guidance_ids declared but aria-describedby missing."

    missing = [rid for rid in describedby if s.find(id=rid) is None]
    if missing:
        return 1, f"aria-describedby includes missing ids: {sorted(set(missing))[:5]}"

    if not any(rid in set(guidance_ids) for rid in describedby):
        return 1, "aria-describedby present but includes none of declared guidance ids."

    any_nonempty = False
    any_exists = False
    for gid in guidance_ids:
        t = s.find(id=gid)
        if t is not None:
            any_exists = True
            if _nonempty_text(t):
                any_nonempty = True
                break

    if not any_exists:
        return 0, "pattern guidance ids declared but no guidance element exists."
    if not any_nonempty:
        return 1, "guidance element exists but text is empty/whitespace."
    return 2, "pattern present and aria-describedby includes declared guidance id with non-empty guidance text."


def ERROR_PRESENT(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    error_ids = _meta_list(meta, "error_ids")
    applicable = len(error_ids) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    found = []
    for eid in error_ids:
        t = s.find(id=eid)
        if t is not None:
            found.append(t)

    if not found:
        return 0, "No declared error element id exists."

    texts = [(t.get_text(" ", strip=True) or "").strip() for t in found]
    nonempty = [tx for tx in texts if tx]
    if not nonempty:
        return 1, "Declared error element(s) exist but text is empty/whitespace."

    for tx in nonempty:
        if _normalize_generic_error_text(tx) != "error":
            return 2, f"Declared error element exists with specific text: {tx}"
    return 1, f"Declared error element text is overly generic: {nonempty[0]}"


def ERROR_ASSOCIATED(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    error_ids = _meta_list(meta, "error_ids")
    applicable = len(error_ids) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    target = _relevant_target_for_association(s, meta)
    if not target:
        return 0, "No relevant target found (primary control or group container)."

    describedby = _aria_describedby_ids(target)
    errormsg = _aria_errormessage_ids(target)

    if not describedby and not errormsg:
        return 0, "No aria-describedby or aria-errormessage on relevant target."

    def _missing(ids: List[str]) -> List[str]:
        return [rid for rid in ids if s.find(id=rid) is None]

    missing_db = _missing(describedby)
    missing_em = _missing(errormsg)

    if missing_db or missing_em:
        parts: List[str] = []
        if missing_db:
            parts.append(f"aria-describedby missing ids: {sorted(set(missing_db))[:5]}")
        if missing_em:
            parts.append(f"aria-errormessage missing ids: {sorted(set(missing_em))[:5]}")
        return 1, "; ".join(parts)

    if any(rid in set(error_ids) for rid in describedby + errormsg):
        return 2, "Error association includes at least one declared error id (describedby or errormessage)."
    return 1, "Error association present but includes none of declared error ids."


def ERROR_INVALID_STATE(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    error_ids = _meta_list(meta, "error_ids")
    applicable = len(error_ids) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    group_level = bool(meta.get("group_validation_expected") is True)

    c = _primary_control(s, meta)
    g = _find_group_container(s, meta) if group_level else None

    if group_level:
        if g is None:
            return 0, "Group-level validation expected but no group container found."

        gv = (g.get("aria-invalid") or "").strip().lower()
        if gv == "true":
            return 2, "aria-invalid='true' present on group container."
        if gv:
            return 1, f"aria-invalid present but not 'true' on group container: {gv}"

        if c is not None:
            cv = (c.get("aria-invalid") or "").strip().lower()
            if cv == "true" or cv:
                return 1, "aria-invalid set on primary control but missing on group container (misplaced for group-level validation)."

        if _any_misplaced_aria_invalid(s, g):
            return 1, "aria-invalid present on a non-relevant element while missing on the group container."
        return 0, "Group-level validation expected but aria-invalid missing on group container."

    if c is None:
        return 0, "No primary control found."

    v = (c.get("aria-invalid") or "").strip().lower()
    if v == "true":
        return 2, "aria-invalid='true' present on primary control."
    if v:
        return 1, f"aria-invalid present but not 'true' on primary control: {v}"

    if _any_misplaced_aria_invalid(s, c):
        return 1, "aria-invalid present on a non-relevant element while missing on the primary control."
    return 0, "aria-invalid missing on primary control."


def STATUS_ANNOUNCE(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    status_ids = _meta_list(meta, "status_ids")
    applicable = len(status_ids) > 0
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    nodes = []
    for sid in status_ids:
        t = s.find(id=sid)
        if t is not None:
            nodes.append(t)

    if not nodes:
        return 0, "No declared status element id exists."

    def _is_live(n) -> bool:
        role = (n.get("role") or "").strip().lower()
        if role in {"status", "alert"}:
            return True
        if n.has_attr("aria-live"):
            v = (n.get("aria-live") or "").strip()
            return bool(v)
        return False

    if any(_is_live(n) for n in nodes):
        return 2, "Declared status element exists and uses a live-region mechanism."
    return 1, "Declared status element exists but lacks role=status/alert and aria-live is missing/empty."


def GROUP_FIELDSET_LEGEND(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "group_required")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    fs = s.find("fieldset")
    if fs:
        leg = fs.find("legend")
        if leg and (leg.get_text(" ", strip=True) or "").strip():
            return 2, "fieldset + non-empty legend present."
        return 1, "fieldset present but legend missing/empty."

    g = s.find(attrs={"role": re.compile(r"^group$", re.I)})
    if not g:
        return 0, "No fieldset and no role=group fallback present."

    aria_label = (g.get("aria-label") or "").strip()
    if aria_label:
        return 1, "role=group present with aria-label fallback (no fieldset/legend)."

    lb_ids = (g.get("aria-labelledby") or "").split()
    for rid in [x.strip() for x in lb_ids if x.strip()]:
        t = s.find(id=rid)
        if t is not None and _nonempty_text(t):
            return 1, "role=group present with aria-labelledby fallback (no fieldset/legend)."

    return 0, "role=group present but accessible name is missing/empty (not a usable fallback)."


def RADIO_SAME_NAME(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "radio_group_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    radios = s.find_all("input", attrs={"type": re.compile(r"^radio$", re.I)})
    if len(radios) < 2:
        return 0, "Radio group expected but fewer than 2 radio inputs found."

    names = [(r.get("name") or "").strip() for r in radios]
    if all(names) and len(set(names)) == 1:
        return 2, f"All radios share the same non-empty name: {names[0]}"
    if any(names):
        return 1, "Some radio names exist but are missing/empty or not identical."
    return 0, "All radio name attributes missing/empty."


def EACH_OPTION_LABELED(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "option_inputs_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    opts = s.find_all("input", attrs={"type": re.compile(r"^(radio|checkbox)$", re.I)})
    if not opts:
        return 0, "Option inputs expected but none found."

    def _option_has_label(o) -> bool:
        oid = (o.get("id") or "").strip()
        if oid:
            lab = s.find("label", attrs={"for": oid})
            if lab is not None and _nonempty_text(lab):
                return True
        labp = o.find_parent("label")
        if labp is not None and _nonempty_text(labp):
            return True
        lb_ids = _aria_labelledby_ids(o)
        if lb_ids:
            for rid in lb_ids:
                t = s.find(id=rid)
                if t is not None and _nonempty_text(t):
                    return True
        aria_label = (o.get("aria-label") or "").strip()
        if aria_label:
            return True
        return False

    ok_count = sum(1 for o in opts if _option_has_label(o))
    if ok_count == len(opts):
        return 2, "Every option input has a programmatic label association."
    if ok_count > 0:
        return 1, f"Some options labeled, others missing ({ok_count}/{len(opts)})."
    return 0, "No option input has any valid label association."


def DISABLED_STATE(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "disabled_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    if c.has_attr("disabled"):
        return 2, "disabled attribute present."
    if (c.get("aria-disabled") or "").strip().lower() == "true":
        return 1, "aria-disabled='true' present but disabled missing."

    if _any_wrapper_disabled_or_aria_disabled(s, c):
        return 1, "Disabled intent encoded on a non-control element (wrapper) but missing on the primary control."
    return 0, "No disabled encoding present on the primary control."


def READONLY_STATE(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "readonly_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    c = _primary_control(s, meta)
    if not c:
        return 0, "No primary control found."

    if c.has_attr("readonly"):
        return 2, "readonly attribute present."
    if (c.get("aria-readonly") or "").strip().lower() == "true":
        return 1, "aria-readonly='true' present but readonly missing."

    if _any_wrapper_readonly_or_aria_readonly(s, c):
        return 1, "Readonly intent encoded on a non-control element (wrapper) but missing on the primary control."
    return 0, "No readonly encoding present on the primary control."


def FILE_INPUT_SEMANTICS(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "file_input_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    fi = s.find("input", attrs={"type": re.compile(r"^file$", re.I)})
    if fi is None:
        return 0, "Missing input[type=file]."

    required_attrs = _meta_list(meta, "file_required_attributes")
    expected_accept = [x.strip() for x in _meta_list(meta, "expected_accept_tokens")]

    def _parse_accept(val: str) -> List[str]:
        return [t.strip().lower() for t in (val or "").split(",") if t.strip()]

    def _looks_like_accept_token(tok: str) -> bool:
        if tok.startswith(".") and len(tok) > 1 and re.fullmatch(r"\.[a-z0-9]+", tok):
            return True
        if "/" in tok:
            return bool(re.fullmatch(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+|[a-z0-9!#$&^_.+-]+/\*|\*/\*", tok))
        return False

    def _image_exts() -> set:
        return {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg"}

    missing: List[str] = []
    for a in required_attrs:
        a_l = a.strip().lower()
        if not fi.has_attr(a_l):
            missing.append(a_l)
            continue
        if a_l == "accept":
            v = (fi.get("accept") or "").strip()
            if not v:
                missing.append("accept(empty)")
                continue

            tokens = _parse_accept(v)
            if not tokens:
                missing.append("accept(empty)")
                continue

            if any(not _looks_like_accept_token(t) for t in tokens):
                return 1, f"accept present but contains malformed token(s): {tokens[:5]}"

            if expected_accept:
                tokset = set(tokens)

                if "image/*" in tokset and all(x.lower() in _image_exts() for x in expected_accept if x.startswith(".")):
                    pass
                elif "application/pdf" in tokset and all(x.lower() == ".pdf" for x in expected_accept if x.startswith(".")):
                    pass
                else:
                    if "*/*" not in tokset:
                        exp_norm = [x.lower() for x in expected_accept]
                        if any(x not in tokset for x in exp_norm):
                            return 1, f"accept present but does not cover expected tokens: expected={exp_norm[:8]} actual={tokens[:8]}"

    if missing:
        return 1, f"input[type=file] present but missing/empty required attributes: {missing[:5]}"
    return 2, "input[type=file] present with all declared required attributes."


def BUTTON_TYPE_AND_NAME(html: str, s: BeautifulSoup, meta: Dict[str, Any]) -> Tuple[int, str]:
    applicable = _meta_true(meta, "button_expected")
    ok, early = _otherwise_2_if_not_applicable(applicable)
    if not ok:
        return early  # type: ignore[return-value]

    expected_type = (meta.get("expected_button_type") or "").strip().lower()

    btn = s.find("button")
    inp = None
    if btn is None:
        inp = s.find("input", attrs={"type": re.compile(r"^(submit|reset|button)$", re.I)})

    if btn is None and inp is None:
        return 0, "No <button> or input[type=submit|reset|button] exists."

    if btn is not None:
        text_name = (btn.get_text(" ", strip=True) or "").strip()
        aria_name = (btn.get("aria-label") or "").strip()
        if not text_name and not aria_name:
            return 0, "Button has no accessible name (no text, no aria-label)."

        actual_type = (btn.get("type") or "").strip().lower()
        if expected_type and actual_type != expected_type:
            return 1, f"<button> present but type mismatch (expected {expected_type}, got {actual_type or '(missing)'})."

        if text_name:
            return 2, f"<button> present with non-empty text name: {text_name}"
        return 1, "Button name only via aria-label (text empty)."

    assert inp is not None
    actual_type = (inp.get("type") or "").strip().lower()
    val = (inp.get("value") or "").strip()
    aria_name = (inp.get("aria-label") or "").strip()

    if not val and not aria_name:
        return 0, "Input button has no accessible name (no value, no aria-label)."

    if expected_type and actual_type != expected_type:
        return 1, f"Input button present but type mismatch (expected {expected_type}, got {actual_type or '(missing)'})."

    if val:
        return 2, f"Input button present with non-empty value name: {val}"
    return 1, "Input button name only via aria-label (value empty)."


# ---------------------------
# Check registry
# ---------------------------

CHECK_IMPL = {
    "AUTOCOMPLETE_PRESENT": AUTOCOMPLETE_PRESENT,
    "AUTOCOMPLETE_VALID_TOKEN": AUTOCOMPLETE_VALID_TOKEN,
    "BUTTON_TYPE_AND_NAME": BUTTON_TYPE_AND_NAME,
    "CONSTRAINT_PATTERN_EXPLAINED": CONSTRAINT_PATTERN_EXPLAINED,
    "DESCRIPTION_ASSOCIATION": DESCRIPTION_ASSOCIATION,
    "DISABLED_STATE": DISABLED_STATE,
    "DOC_VALID_HTML": DOC_VALID_HTML,
    "EACH_OPTION_LABELED": EACH_OPTION_LABELED,
    "ERROR_ASSOCIATED": ERROR_ASSOCIATED,
    "ERROR_INVALID_STATE": ERROR_INVALID_STATE,
    "ERROR_PRESENT": ERROR_PRESENT,
    "FILE_INPUT_SEMANTICS": FILE_INPUT_SEMANTICS,
    "GROUP_FIELDSET_LEGEND": GROUP_FIELDSET_LEGEND,
    "ID_UNIQUENESS": ID_UNIQUENESS,
    "INPUT_TYPE_MATCHES_PURPOSE": INPUT_TYPE_MATCHES_PURPOSE,
    "LABEL_ASSOCIATION": LABEL_ASSOCIATION,
    "LABEL_VISIBLE_OR_EQUIV": LABEL_VISIBLE_OR_EQUIV,
    "NAME_ROLE_VALUE_NATIVE": NAME_ROLE_VALUE_NATIVE,
    "RADIO_SAME_NAME": RADIO_SAME_NAME,
    "READONLY_STATE": READONLY_STATE,
    "REF_TARGET_EXISTS": REF_TARGET_EXISTS,
    "REQUIRED_CUE_EXPLAINED": REQUIRED_CUE_EXPLAINED,
    "REQUIRED_STATE": REQUIRED_STATE,
    "STATUS_ANNOUNCE": STATUS_ANNOUNCE,
}

__all__ = ["score_checks", "CheckScore", "CHECK_IMPL"]