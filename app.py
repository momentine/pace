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
# from bench.azure import AzureFoundryClient as OpenRouterClient
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


# ----------------------------
# Always-on visual theme
# ----------------------------
st.markdown(
    """
<style>
/* PACE white + cheeseburger-ish soft UI */
:root{
  --pace-bg: #ffffff;
  --pace-ink: #24211d;
  --pace-muted: rgba(36,33,29,.66);
  --pace-blue: #7ea7f2;
  --pace-blue-soft: rgba(126,167,242,.15);
  --pace-blue-line: rgba(126,167,242,.34);
  --pace-cream: #fffaf0;
}

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
main{
  background: #ffffff !important;
  background-color: #ffffff !important;
  background-image: none !important;
  color: var(--pace-ink) !important;
}

header[data-testid="stHeader"],
div[data-testid="stToolbar"]{
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  visibility: hidden !important;
  height: 0 !important;
}

/* readable page width */
.block-container{
  max-width: 1120px !important;
  padding-top: 2.2rem !important;
  padding-left: 2.4rem !important;
  padding-right: 2.4rem !important;
  padding-bottom: 3rem !important;
  background: transparent !important;
}

/* cheerful rounded readable type */
html, body, .stApp, input, textarea, select,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
label, p, li{
  font-family: "Avenir Next", "Nunito", "SF Pro Rounded", "Segoe UI", system-ui, sans-serif !important;
  color: var(--pace-ink) !important;
}

/* larger base text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] p,
label,
li{
  font-size: 1rem !important;
  line-height: 1.55 !important;
}

/* hand-drawn feeling for headings */
h1, h2, h3, h4,
.pace-wordmark,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4{
  font-family: "Bradley Hand", "Chalkboard SE", "Segoe Print", "Comic Sans MS", "Avenir Next", system-ui, sans-serif !important;
  font-weight: 650 !important;
  color: var(--pace-ink) !important;
  letter-spacing: .01em !important;
}
[data-testid="stMarkdownContainer"] h1{ font-size: 2.35rem !important; }
[data-testid="stMarkdownContainer"] h2{ font-size: 1.9rem !important; }
[data-testid="stMarkdownContainer"] h3{ font-size: 1.55rem !important; }
[data-testid="stMarkdownContainer"] h4{ font-size: 1.3rem !important; }

.pace-wordmark{
  font-size: 2.35rem !important;
  line-height: 1 !important;
}
.pace-subtitle{
  font-size: .98rem !important;
  color: var(--pace-muted) !important;
}

/* no lines unless Streamlit absolutely needs structure */
hr,
[data-testid="stDivider"],
[data-testid="stMarkdownContainer"] hr,
div[data-baseweb="tab-border"],
div[data-baseweb="tab-highlight"]{
  display: none !important;
  border: 0 !important;
  height: 0 !important;
  box-shadow: none !important;
}

/* tabs like soft pills */
div[data-baseweb="tab-list"]{
  border: 0 !important;
  box-shadow: none !important;
  gap: .52rem !important;
  margin: .5rem 0 1.2rem 0 !important;
}
button[data-baseweb="tab"],
div[role="tab"]{
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: .5rem .9rem !important;
  font-size: 1.04rem !important;
  color: var(--pace-muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"],
div[role="tab"][aria-selected="true"]{
  background: var(--pace-blue-soft) !important;
  color: var(--pace-ink) !important;
  border-radius: 999px !important;
}

/* settings tabs: axe/schema far right */
div[data-baseweb="modal"] div[data-baseweb="tab-list"]{
  display: flex !important;
  width: 100% !important;
}
div[data-baseweb="modal"] div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-of-type(5),
div[data-baseweb="modal"] div[data-baseweb="tab-list"] div[role="tab"]:nth-of-type(5){
  margin-left: auto !important;
}

/* inputs: cheeseburger-ish friendly rounded controls */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] div[role="combobox"],
div[data-testid="stFileUploader"] section,
div[data-testid="stNumberInput"] input{
  background: #ffffff !important;
  border: 1.5px solid var(--pace-blue-line) !important;
  border-radius: 16px !important;
  box-shadow: none !important;
  min-height: 2.75rem !important;
  font-size: 1rem !important;
}

/* multiselect tags */
div[data-baseweb="tag"]{
  background: #f4f8ff !important;
  border: 0 !important;
  border-radius: 999px !important;
  font-size: .96rem !important;
  color: #ffffff !important;
}
div[data-baseweb="tag"] span,
div[data-baseweb="tag"] svg,
div[data-baseweb="tag"] *{
  color: #ffffff !important;
  fill: #ffffff !important;
}

/* buttons */
div.stButton > button,
button[kind="secondary"],
button[kind="primary"],
button[kind="tertiary"],
div[data-testid="stDownloadButton"] button{
  border: 0 !important;
  border-radius: 999px !important;
  background: var(--pace-blue-soft) !important;
  color: var(--pace-ink) !important;
  min-height: 2.65rem !important;
  padding: .6rem 1.1rem !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
  box-shadow: none !important;
}
div.stButton > button:hover,
div[data-testid="stDownloadButton"] button:hover{
  background: rgba(126,167,242,.24) !important;
}

/* keep pencil visible */
#st-key-home_open_settings{
  position: fixed !important;
  top: 1.55rem !important;
  right: calc((100vw - min(1120px, 100vw))/2 + 1.9rem) !important;
  z-index: 10000 !important;
}
#st-key-home_open_settings button{
  width: 3.55rem !important;
  height: 3.55rem !important;
  min-height: 3.55rem !important;
  padding: 0 !important;
  border-radius: 999px !important;
  background: #ffffff !important;
  box-shadow: 0 5px 18px rgba(36,33,29,.12) !important;
  font-family: "Bradley Hand", "Chalkboard SE", "Segoe Print", system-ui !important;
  font-size: 1.8rem !important;
}

/* remove boxy surfaces */
div[data-testid="stContainer"],
div[data-testid="stForm"],
details[data-testid="stExpander"],
div[data-testid="stPopoverBody"],
div[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"]{
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
}

/* metrics more readable */
div[data-testid="stMetric"]{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
div[data-testid="stMetric"] label{
  font-size: .95rem !important;
  color: var(--pace-muted) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{
  font-size: 1.55rem !important;
}

/* alerts and code */
div[data-testid="stAlert"]{
  border: 0 !important;
  border-radius: 18px !important;
  background: #f4f8ff !important;
}
pre, code{
  border: 0 !important;
  border-radius: 12px !important;
  background: #f7f9ff !important;
  font-size: .98rem !important;
}
</style>
""",
    unsafe_allow_html=True,
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
hr{ opacity: 0.24; }

/* Analog paper-like surface */
.stApp{ background: #ffffff !important; background-image: none !important; }

.block-container{
  max-width: 1080px !important;
}

/* Small, calm text rhythm */
p, li, label{
  font-size-adjust: none;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stCaptionContainer"] p{
  font-size: 0.88rem !important;
}

/* Keep surfaces light and low-chrome unless Streamlit needs contrast */
div[data-testid="stContainer"],
details[data-testid="stExpander"],
div[data-testid="stForm"]{
  box-shadow: none !important;
}



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
# Paper notebook override
# ----------------------------
st.markdown(
    """
<style>
:root{
  --paper:#fbfaf4;
  --paper-soft:#f6f1e6;
  --ink:#332f27;
  --ink-light:rgba(51,47,39,.62);
  --pencil:rgba(97,83,58,.18);
}

.stApp{ background: #ffffff !important; background-image: none !important; }

.block-container{
  max-width: 980px !important;
  padding-top: 2.4rem !important;
  padding-left: 2.2rem !important;
  padding-right: 2.2rem !important;
}

html, body, [class*="css"], .stApp, input, textarea, select{
  font-family: ui-rounded, "SF Pro Rounded", "Avenir Next", "Nunito", "Comic Sans MS", system-ui, sans-serif !important;
}

h1, h2, h3, h4,
.pace-wordmark,
.app-navbar-title,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4{
  font-family: "Bradley Hand", "Marker Felt", "Comic Sans MS", ui-rounded, system-ui, sans-serif !important;
  font-weight: 500 !important;
  letter-spacing: .01em !important;
}

.pace-hero{
  margin: .35rem 0 1.15rem 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}
.pace-wordmark{
  font-size: 1.9rem !important;
  line-height: 1 !important;
  color: rgba(42,37,29,.86) !important;
}
.pace-subtitle{
  margin-top: .25rem !important;
  font-size: .76rem !important;
  color: var(--ink-light) !important;
  letter-spacing: .03em !important;
}

/* No visible divider lines */
hr,
div[data-testid="stDivider"],
[data-testid="stMarkdownContainer"] hr{
  display:none !important;
  border:0 !important;
  height:0 !important;
  margin:0 !important;
}

/* Remove tab divider feel */
div[data-baseweb="tab-list"]{
  border:0 !important;
  box-shadow:none !important;
  gap:.28rem !important;
  margin-bottom:.8rem !important;
}
button[data-baseweb="tab"], div[role="tab"]{
  border:0 !important;
  background: transparent !important;
  box-shadow:none !important;
  font-size:.84rem !important;
  color: rgba(51,47,39,.58) !important;
  padding:.35rem .58rem !important;
}
button[data-baseweb="tab"][aria-selected="true"], div[role="tab"][aria-selected="true"]{
  color: rgba(51,47,39,.92) !important;
  background: rgba(234,222,199,.45) !important;
  border-radius:999px !important;
}

/* Settings tabs: real tabs, axe/schema drift right */
div[data-baseweb="modal"] div[data-baseweb="tab-list"]{
  display:flex !important;
  width:100% !important;
}
div[data-baseweb="modal"] div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-of-type(5),
div[data-baseweb="modal"] div[data-baseweb="tab-list"] div[role="tab"]:nth-of-type(5){
  margin-left:auto !important;
}

/* Floating pencil: cute but quiet */
#st-key-home_open_settings{
  position: fixed !important;
  top: .75rem !important;
  right: .9rem !important;
  z-index: 10000 !important;
}
#st-key-home_open_settings button{
  width: 2.15rem !important;
  height: 2.15rem !important;
  min-height: 2.15rem !important;
  padding:0 !important;
  border:0 !important;
  border-radius:999px !important;
  background: rgba(255,253,247,.72) !important;
  box-shadow: 0 6px 18px rgba(69,58,37,.08) !important;
  color: rgba(55,47,35,.78) !important;
  font-size: 1.08rem !important;
}
#st-key-home_open_settings button:hover{
  background: rgba(238,228,206,.75) !important;
}

/* Paper controls: smaller, less boxy */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] div[role="combobox"]{
  background: rgba(255,253,247,.56) !important;
  border: 1px solid rgba(91,77,53,.13) !important;
  border-radius: 9px !important;
  min-height: 2.05rem !important;
  font-size: .86rem !important;
  box-shadow:none !important;
}
label, div[data-testid="stCaptionContainer"] p{
  font-size:.78rem !important;
  color: rgba(51,47,39,.56) !important;
}

/* Buttons as small paper chips */
div.stButton > button,
button[kind="secondary"],
button[kind="primary"],
div[data-testid="stDownloadButton"] button{
  border:0 !important;
  border-radius: 999px !important;
  min-height: 2.1rem !important;
  padding: .35rem .75rem !important;
  font-size:.84rem !important;
  box-shadow:none !important;
}

/* Remove heavy cards and outlines */
div[data-testid="stContainer"],
div[data-testid="stForm"],
details[data-testid="stExpander"],
div[data-testid="stPopoverBody"]{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

/* Soft note boxes for warnings/info only */
div[data-testid="stAlert"]{
  border:0 !important;
  border-radius: 12px !important;
  background: rgba(242,232,210,.42) !important;
  color: rgba(51,47,39,.82) !important;
}

/* Metrics should feel like margin notes */
div[data-testid="stMetric"]{
  background: transparent !important;
  border:0 !important;
  box-shadow:none !important;
}
div[data-testid="stMetric"] label{
  font-size:.72rem !important;
}

/* Dataframes/charts: less hard edge */
div[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"]{
  border:0 !important;
  box-shadow:none !important;
  background: transparent !important;
}
pre, code{
  background: rgba(238,228,206,.42) !important;
  border:0 !important;
  border-radius: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)



# ----------------------------
# Final notebook UI pass: flatter, quieter, fewer lines
# ----------------------------
st.markdown(
    """
<style>
:root{
  --pace-paper: #fbfaf4;
  --pace-ink: #302b24;
  --pace-muted: rgba(48,43,36,.54);
  --pace-faint: rgba(48,43,36,.075);
  --pace-wash: rgba(232,222,202,.34);
}

.stApp{ background: #ffffff !important; background-image: none !important; }

h1, h2, h3, h4,
.pace-wordmark,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4{
  font-family: "Bradley Hand", "Chalkboard SE", "Segoe Print", "Avenir Next", system-ui, sans-serif !important;
  font-weight: 500 !important;
  letter-spacing: .005em !important;
}

.pace-wordmark{
  font-size: 1.7rem !important;
  color: rgba(48,43,36,.78) !important;
}
.pace-subtitle{
  font-size: .68rem !important;
  color: var(--pace-muted) !important;
  text-transform: lowercase !important;
  letter-spacing: .06em !important;
}

/* remove line-heavy Streamlit chrome */
hr,
div[data-testid="stDivider"],
[data-testid="stMarkdownContainer"] hr,
div[data-baseweb="tab-border"],
div[data-baseweb="tab-highlight"],
[data-testid="stHeader"],
[data-testid="stToolbar"]{
  display:none !important;
  border:0 !important;
  box-shadow:none !important;
}

/* tabs as soft handwritten labels, no rule underneath */
div[data-baseweb="tab-list"]{
  border:0 !important;
  box-shadow:none !important;
  gap:.38rem !important;
  margin:.25rem 0 1rem 0 !important;
}
button[data-baseweb="tab"], div[role="tab"]{
  border:0 !important;
  background:transparent !important;
  box-shadow:none !important;
  padding:.32rem .56rem !important;
  color:rgba(48,43,36,.54) !important;
  font-size:.83rem !important;
}
button[data-baseweb="tab"][aria-selected="true"], div[role="tab"][aria-selected="true"]{
  color:rgba(48,43,36,.90) !important;
  background:var(--pace-wash) !important;
  border-radius:999px !important;
}

/* Settings: keep axe/schema on far right with real tabs only */
div[data-baseweb="modal"] div[data-baseweb="tab-list"]{
  display:flex !important;
  width:100% !important;
}
div[data-baseweb="modal"] div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-of-type(5),
div[data-baseweb="modal"] div[data-baseweb="tab-list"] div[role="tab"]:nth-of-type(5){
  margin-left:auto !important;
}

/* floating pencil only, no nav */
#st-key-home_open_settings{
  position: fixed !important;
  top: .75rem !important;
  right: .9rem !important;
  z-index: 10000 !important;
}
#st-key-home_open_settings button{
  width: 2rem !important;
  height: 2rem !important;
  min-height: 2rem !important;
  padding:0 !important;
  border:0 !important;
  border-radius:999px !important;
  background:transparent !important;
  box-shadow:none !important;
  color:rgba(48,43,36,.62) !important;
  font-family:"Bradley Hand", "Chalkboard SE", system-ui !important;
  font-size:1.05rem !important;
}
#st-key-home_open_settings button:hover{
  background:rgba(232,222,202,.38) !important;
  color:rgba(48,43,36,.86) !important;
}

/* inputs: no boxes unless needed */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] div[role="combobox"],
div[data-testid="stFileUploader"] section{
  background:rgba(255,253,247,.44) !important;
  border:0 !important;
  border-radius:10px !important;
  box-shadow: inset 0 -1px 0 rgba(48,43,36,.13) !important;
  font-size:.84rem !important;
}

/* remove card/box feeling everywhere */
div[data-testid="stContainer"],
div[data-testid="stForm"],
details[data-testid="stExpander"],
div[data-testid="stPopoverBody"],
div[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"]{
  border:0 !important;
  box-shadow:none !important;
  background:transparent !important;
}

/* expanders should not look like compartments */
details[data-testid="stExpander"] summary{
  padding:.2rem 0 !important;
  color:var(--pace-muted) !important;
}

/* buttons: soft text chips */
div.stButton > button,
button[kind="secondary"],
button[kind="primary"],
div[data-testid="stDownloadButton"] button{
  border:0 !important;
  border-radius:999px !important;
  background:rgba(232,222,202,.42) !important;
  box-shadow:none !important;
  min-height:1.9rem !important;
  padding:.32rem .72rem !important;
  font-size:.82rem !important;
  color:rgba(48,43,36,.76) !important;
}

/* metrics as tiny annotations */
div[data-testid="stMetric"]{
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
  padding:0 !important;
}
div[data-testid="stMetric"] label{
  color:var(--pace-muted) !important;
  font-size:.70rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{
  font-size:1.12rem !important;
  color:rgba(48,43,36,.82) !important;
}

label, div[data-testid="stCaptionContainer"] p{
  color:var(--pace-muted) !important;
  font-size:.76rem !important;
}

pre, code{
  border:0 !important;
  border-radius:9px !important;
  background:rgba(232,222,202,.36) !important;
}
</style>
""",
    unsafe_allow_html=True,
)



# ----------------------------
# Streamlit icon font restore
# ----------------------------
st.markdown(
    """
<style>
/* Do not let the app font override Streamlit Material Symbols. */
[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] *,
span[class*="material-symbol"],
span[class*="material-icons"],
span[class*="Material"],
i[class*="material-symbol"],
i[class*="material-icons"],
i[class*="Material"],
svg[data-testid*="stIcon"],
button [data-testid*="stIcon"],
button [data-testid*="Icon"]{
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons", sans-serif !important;
  font-weight: normal !important;
  font-style: normal !important;
  line-height: 1 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  white-space: nowrap !important;
  direction: ltr !important;
  -webkit-font-feature-settings: "liga" !important;
  font-feature-settings: "liga" !important;
  -webkit-font-smoothing: antialiased !important;
}

/* Give expander summaries their normal text font without touching icon spans. */
details[data-testid="stExpander"] summary,
details[data-testid="stExpander"] summary p{
  font-family: "Avenir Next", "Nunito", "SF Pro Rounded", "Segoe UI", system-ui, sans-serif !important;
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
    st.markdown("#### axe")
    st.caption("choose the automated checker scope")

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

    st.write("")
    st.markdown("**run mode**")

    if profile == "form_fragment_rules":
        st.write("rule list · PACE default")
        st.write(f"{len(DEFAULT_FORM_FRAGMENT_RULES)} rule IDs")
        with st.expander("rule IDs", expanded=False):
            st.code("\n".join(DEFAULT_FORM_FRAGMENT_RULES))

    elif profile == "all_rules":
        st.write("all axe rules")

    else:
        tag = {
            "wcag2a_tags": "wcag2a",
            "wcag2aa_tags": "wcag2aa",
            "wcag21aa_tags": "wcag21aa",
        }[profile]
        st.write(f"tag · `{tag}`")

    with st.expander("config", expanded=False):
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
    st.markdown("#### models")
    st.caption("choose models")

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

    st.write("")
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
    st.markdown("#### instructions")
    st.caption("edit system instructions")

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

    st.write("")
    add_col, _ = st.columns([1, 3], vertical_alignment="center")
    with add_col:
        if st.button("Addinstruction set", use_container_width=True, key="pc_add_prompt"):
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
    {"variant_id": "G1", "label": "Action command", "template": "Insert a {component} for “{label}”{attribute}"},
    {"variant_id": "G2", "label": "Short specification", "template": "{component}: {label}{attribute}"},
    {"variant_id": "G3", "label": "Descriptive object phrase", "template": "A {component} labeled “{label}”{attribute}"},
    {"variant_id": "G4", "label": "Build instruction", "template": "Build a {component} and label for “{label}”{attribute}"},
    {"variant_id": "G5", "label": "Standards-oriented phrasing", "template": "Accessible {component} for “{label}”{attribute}"},
]
_PREVIEW_EXAMPLE = {"component": "text field", "label": "What color is an orange?", "attribute": " with hint “orange”"}


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
    st.markdown("#### variants")
    st.caption("edit prompt variants")

    with st.container(border=True):
        st.markdown("**Placeholders**")
        st.markdown("- `{component}` — insert the component type (e.g., text field, radio group)")
        st.markdown("- `{label}` — insert the user-facing label or question text")
        st.markdown("- `{attribute}` — insert optional trailing details (e.g., hint text or constraints)")

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

    st.write("")
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
                            "{attribute}",
                            key=f"v_ins_s_{vid}",
                            on_click=_append_token,
                            args=(tpl_key, "{attribute}"),
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

    st.write("")
    add_col, _ = st.columns([1, 3], vertical_alignment="center")
    with add_col:
        if st.button("Add variant", use_container_width=True, key="v_add"):
            _stay_in_settings()
            vid = _next_variant_id([r["variant_id"] for r in rows])
            rows.append(
                {
                    "variant_id": vid,
                    "label": "New variant",
                    "template": "Insert a {component} for “{label}”{attribute}",
                }
            )
            st.session_state[f"v_lab_{vid}"] = "New variant"
            st.session_state[f"v_tpl_{vid}"] = "Insert a {component} for “{label}”{attribute}"
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
        attribute = str(t.get("attribute", "")).strip()

        fields = t.get("fields", {})
        if isinstance(fields, dict):
            if not component:
                component = str(fields.get("component", "") or fields.get("component_phrase", "")).strip()
            if not label:
                label = str(fields.get("label", "")).strip()
            if not attribute:
                attribute = str(fields.get("attribute", "")).strip()

        if not tid:
            continue
        if not title and component:
            title = _title_case_first(component)

        out.append({"test_id": tid, "title": title, "component": component, "label": label, "attribute": attribute})
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
        attribute2 = str(t.get("attribute", "")).strip()

        if not title and component:
            title = _title_case_first(component)
        if attribute2 and not attribute2.startswith(" "):
            attribute2 = " " + attribute2

        payload_now.append(
            {"test_id": tid, "title": title or tid, "component": component, "label": label, "attribute": attribute2}
        )
    _write_json(COMPONENTS_PATH, {"tests": payload_now})
    st.session_state["_reg_reload"] = True


def _render_components_editor(reg) -> None:
    st.markdown("#### components")
    st.caption("edit form components")

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
            new_test = {"test_id": tid, "title": "New test", "component": "", "label": "", "attribute": ""}
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

    st.write("")
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
        st.session_state[k_suf] = (test.get("attribute") or "").strip()

    def _on_components_edit() -> None:
        _stay_in_settings()
        test["title"] = (st.session_state.get(k_title, "") or "").strip()
        test["component"] = (st.session_state.get(k_comp, "") or "").strip()
        test["label"] = (st.session_state.get(k_lab, "") or "").strip()
        test["attribute"] = (st.session_state.get(k_suf, "") or "").strip()
        tests[idx] = dict(test)
        st.session_state["settings_components_tests"] = tests
        _autosave_components(tests)

    st.text_input("Title", key=k_title, placeholder="e.g., Text Field with Placeholder", on_change=_on_components_edit)
    st.text_input("component", key=k_comp, placeholder="e.g., text field", on_change=_on_components_edit)
    st.text_input("Label", key=k_lab, placeholder="e.g., What color is an orange?", on_change=_on_components_edit)
    st.text_input("attribute (optional)", key=k_suf, placeholder="e.g., with hint “orange”", on_change=_on_components_edit)

    st.write("")
    st.subheader("Generated variants")

    if not variants:
        st.info("No variants available.")
    else:
        attribute = (test.get("attribute") or "").strip()
        if attribute and not attribute.startswith(" "):
            attribute = " " + attribute

        fields = {
            "component": (test.get("component") or "").strip(),
            "label": (test.get("label") or "").strip(),
            "attribute": attribute,
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

    st.write("")
    a1, a2, _ = st.columns([1, 1, 1])

    with a1:
        if st.button("Add test", use_container_width=True, key="settings_components_add"):
            _stay_in_settings()
            existing = [str(t.get("test_id", "")).strip() for t in tests if str(t.get("test_id", "")).strip()]
            tid = _next_test_id(existing)
            new_test = {"test_id": tid, "title": "New test", "component": "", "label": "", "attribute": ""}
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
    st.markdown("#### schema")
    st.caption("connect rubric checks to each component")

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

    tab_map, tab_checks = st.tabs(["map checks", "check library"])

    with tab_map:
        st.subheader("map checks")
        st.caption("pick a component, then choose its rubric checks")

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
            "component",
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
            "checks",
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

        with st.expander("selected checks", expanded=False):
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
    #             st.markdown("**WCAG**")
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
    #         st.markdown("**score rules**")

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
        st.subheader("check library")
        st.caption("rubric details")

        if not check_ids:
            st.info("No check_definitions found.")
            return

        picked_check = st.selectbox("check", options=check_ids, key="schema_check_picker")
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

        st.write("")
        # --- Notes (single column, minimal chrome) ---
        with st.expander("notes", expanded=True):
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

        st.write("")
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
@st.dialog("settings")
def _open_settings(reg) -> None:
    st.caption("auto-saves")
    st.markdown('<div id="settings-tabs-marker"></div>', unsafe_allow_html=True)
    tabs = st.tabs([
        "models",
        "instructions",
        "components",
        "variants",
        "axe",
        "schema",
    ])
    with tabs[0]:
        _render_models_editor(reg)
    with tabs[1]:
        _render_system_prompt_editor(reg)
    with tabs[2]:
        _render_components_editor(reg)
    with tabs[3]:
        _render_variants_editor(reg)
    with tabs[4]:
        _render_axe_settings()
    with tabs[5]:
        _render_schema_editor(reg)


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
# Paper-style page chrome
# ----------------------------
st.markdown(
    """
    <style>
      /* Hide Streamlit's default chrome. The app should feel like a quiet paper workspace. */
      header[data-testid="stHeader"],
      div[data-testid="stToolbar"]{
        height: 0px !important;
        visibility: hidden !important;
      }

      /* Whole-page paper feel */
      html, body, [data-testid="stAppViewContainer"], .stApp{ background: #ffffff !important; background-image: none !important; }

      .block-container{
        max-width: 1080px !important;
        padding-top: 1.55rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
      }

      h1, h2, h3, h4,
      div[data-testid="stMarkdownContainer"] h1,
      div[data-testid="stMarkdownContainer"] h2,
      div[data-testid="stMarkdownContainer"] h3{
        font-family: "Bradley Hand", "Chalkboard SE", "Segoe Print", "Avenir Next", system-ui, sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.005em !important;
        color: rgba(47, 42, 35, 0.92) !important;
      }

      p, li, label{
        letter-spacing: 0.005em;
      }

      /* Small floating settings control, not a navbar */
      div[data-testid="stVerticalBlock"] > div:has(#floating-settings-marker){
        position: fixed !important;
        top: 0.75rem !important;
        right: 1rem !important;
        z-index: 1000 !important;
        width: auto !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
      }

      #st-key-home_open_settings{
        display: flex !important;
        justify-content: flex-end !important;
      }

      #st-key-home_open_settings button{
        border: 0 !important;
        background: rgba(251, 250, 244, 0.72) !important;
        box-shadow: none !important;
        color: rgba(47, 42, 35, 0.70) !important;
        padding: 0.25rem 0.45rem !important;
        min-height: 1.8rem !important;
        border-radius: 999px !important;
        font-family: "Bradley Hand", "Chalkboard SE", "Segoe Print", system-ui, sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
      }

      #st-key-home_open_settings button:hover{
        background: rgba(235, 226, 205, 0.72) !important;
        color: rgba(47, 42, 35, 0.92) !important;
      }

      /* Small, left-aligned wordmark. No card, no border. */
      .pace-hero{
        margin: 0.25rem 0 1.15rem 0;
        padding: 0;
        text-align: left;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
      }

      .pace-wordmark{
        font-family: "Bradley Hand", "Chalkboard SE", "Segoe Print", "Comic Sans MS", cursive;
        font-size: clamp(1.55rem, 3vw, 2.35rem);
        font-weight: 500;
        line-height: 0.98;
        letter-spacing: 0.01em;
        color: rgba(47, 42, 35, 0.82);
      }

      .pace-subtitle{
        margin-top: 0.12rem;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: lowercase;
        opacity: 0.50;
      }

      /* Paper-light controls */
      div[data-testid="stForm"],
      details[data-testid="stExpander"],
      div[data-testid="stPopoverBody"]{
        background: rgba(255, 253, 247, 0.46) !important;
        border: 1px solid rgba(71, 60, 44, 0.08) !important;
        box-shadow: none !important;
      }

      div[data-testid="stContainer"]{
        background: transparent !important;
        box-shadow: none !important;
      }

      div[data-baseweb="input"] input,
      div[data-baseweb="textarea"] textarea,
      div[data-baseweb="select"] > div,
      div[data-testid="stMultiSelect"] div[role="combobox"]{
        background: rgba(255, 253, 247, 0.64) !important;
        border-color: rgba(71, 60, 44, 0.16) !important;
        box-shadow: none !important;
        border-radius: 10px !important;
        font-size: 0.90rem !important;
      }

      div.stButton > button,
      button[kind="secondary"],
      button[kind="primary"]{
        border-radius: 999px !important;
        box-shadow: none !important;
        font-size: 0.88rem !important;
      }

      /* Dialog feels like a sheet of paper */
      div[data-baseweb="modal"] [role="dialog"],
      div[data-testid="stDialog"] [role="dialog"]{
        background: #fbfaf4 !important;
        border: 1px solid rgba(71, 60, 44, 0.10) !important;
        box-shadow: 0 18px 45px rgba(47, 42, 35, 0.10) !important;
        border-radius: 12px !important;
      }

      /* Real right alignment for Axe + Schema in Settings. No spacer tabs. */
      div[data-baseweb="modal"] div[data-baseweb="tab-list"]{
        display: flex !important;
        width: 100% !important;
        gap: 0.22rem !important;
      }

      div[data-baseweb="modal"] div[data-baseweb="tab-list"] button[data-baseweb="tab"]:nth-of-type(5),
      div[data-baseweb="modal"] div[data-baseweb="tab-list"] div[role="tab"]:nth-of-type(5){
        margin-left: auto !important;
      }

      div[data-baseweb="modal"] button[data-baseweb="tab"],
      div[data-baseweb="modal"] div[role="tab"]{
        font-size: 0.82rem !important;
        border-radius: 999px !important;
        padding: 0.42rem 0.72rem !important;
        font-weight: 500 !important;
      }

      /* Dataframes and code should feel lighter */
      pre, code{
        border-radius: 10px !important;
        background: rgba(245, 239, 225, 0.62) !important;
      }

      hr{
        opacity: 0.20 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Final layout tune: logo header, visible pencil, quiet headings, centered error bars
# ----------------------------
st.markdown(
    """
<style>
/* visible but still small pencil */
div[data-testid="stVerticalBlock"] > div:has(#floating-settings-marker){
  top: 1.05rem !important;
  right: 1.55rem !important;
}
#st-key-home_open_settings button{
  width: 2.55rem !important;
  height: 2.55rem !important;
  min-height: 2.55rem !important;
  font-size: 1.32rem !important;
  background: rgba(255,253,247,.84) !important;
  color: rgba(48,43,36,.78) !important;
  box-shadow: 0 4px 18px rgba(48,43,36,.07) !important;
}
#st-key-home_open_settings button:hover{
  background: rgba(236,226,205,.78) !important;
  color: rgba(48,43,36,.94) !important;
}

/* logo row: no nav bar, no rule, just a small mark */
.pace-logo-row{
  display:flex;
  align-items:center;
  gap:.55rem;
  margin:.1rem 0 1.2rem 0;
}
.pace-logo-img img{
  width:54px !important;
  max-width:54px !important;
}
.pace-wordmark{
  font-size:1.58rem !important;
  line-height:1 !important;
}
.pace-subtitle{
  font-size:.66rem !important;
  margin-top:.08rem !important;
}

/* make headings distinct without boxes or lines */
.note-section{
  margin:.15rem 0 .65rem 0;
}
.note-kicker{
  font-family:"Bradley Hand", "Chalkboard SE", "Segoe Print", system-ui, sans-serif !important;
  font-size:1.08rem;
  color:rgba(48,43,36,.86);
  letter-spacing:.01em;
  margin-bottom:.1rem;
}
.note-hint{
  font-size:.72rem;
  color:rgba(48,43,36,.48);
  margin-bottom:.55rem;
}
.note-inline{
  display:flex;
  align-items:center;
  gap:.45rem;
  flex-wrap:wrap;
}

/* remove remaining hard compartment outlines */
[data-testid="stFileUploader"] section,
div[data-testid="stForm"],
details[data-testid="stExpander"],
div[data-testid="stPopoverBody"]{
  border:0 !important;
  box-shadow:none !important;
  background:transparent !important;
}

/* softer checkbox/radio scale */
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label{
  font-size:.82rem !important;
}

/* settings dialog should breathe without looking boxed */
div[data-baseweb="modal"] [role="dialog"],
div[data-testid="stDialog"] [role="dialog"]{
  border:0 !important;
  box-shadow:0 20px 52px rgba(48,43,36,.11) !important;
}
</style>
""",
    unsafe_allow_html=True,
)



# ----------------------------
# Placement tune: larger pencil + tighter run buttons
# ----------------------------
st.markdown(
    """
<style>
/* settings pencil: larger, lower, and pulled back into the page */
div[data-testid="stVerticalBlock"] > div:has(#floating-settings-marker){
  top: 3.2rem !important;
  right: max(5.0rem, calc((100vw - 1080px) / 2 + 4.6rem)) !important;
  z-index: 10000 !important;
}
#st-key-home_open_settings button{
  width: 3.6rem !important;
  height: 3.6rem !important;
  min-height: 3.6rem !important;
  padding: 0 !important;
  font-size: 1.9rem !important;
  background: rgba(255,253,247,.78) !important;
  color: rgba(48,43,36,.82) !important;
  box-shadow: 0 6px 22px rgba(48,43,36,.08) !important;
}
#st-key-home_open_settings button:hover{
  background: rgba(236,226,205,.78) !important;
  color: rgba(48,43,36,.95) !important;
}

/* keep the logo row airy, but leave room for the pencil */
.pace-logo-row, .pace-hero{
  margin-top: .25rem !important;
}

/* run/stop buttons: compact pair instead of drifting apart */
#run-buttons-anchor + div[data-testid="stHorizontalBlock"]{
  gap: .18rem !important;
}
#run-buttons-anchor + div[data-testid="stHorizontalBlock"] div.stButton > button{
  min-width: 3.85rem !important;
  padding-left: .85rem !important;
  padding-right: .85rem !important;
}

/* compact run selectors */
div[data-testid="stCheckbox"]{
  margin-bottom: -0.25rem !important;
}
div[data-testid="stCheckbox"] label p{
  font-size: .82rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Floating settings control + small intro
# ----------------------------
with st.container():
    st.markdown('<div id="floating-settings-marker"></div>', unsafe_allow_html=True)
    clicked = st.button("✎", key="home_open_settings", type="tertiary", help="settings")
    if clicked:
        st.session_state["_settings_open"] = True
        st.rerun()

hero_icon, hero_text, _hero_space = st.columns([0.08, 0.48, 0.44], vertical_alignment="center")
with hero_icon:
    if Path("assets/logo.svg").exists():
        st.image("assets/logo.svg", width=54)
with hero_text:
    st.markdown(
        """
        <div class="pace-hero">
          <div class="pace-wordmark">PACE</div>
          <div class="pace-subtitle">prompt accessibility controlled evaluation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
st.markdown("### bench notebook")
tab_run, tab_results = st.tabs(["run", "results"])


# # ============
# # Run
# # ============
# def _load_selected_models_for_run() -> list[dict[str, Any]]:
#     obj = _read_json(SELECTED_MODELS_PATH)
#     rows = obj.get("data", []) if isinstance(obj, dict) else []
#     if not isinstance(rows, list):
#         return []
#     out: list[dict[str, Any]] = []
#     for r in rows:
#         if isinstance(r, dict):
#             mid = str(r.get("id") or "").strip()
#             if mid:
#                 out.append(r)
#     return out


# def _list_runs() -> list[Path]:
#     if not RUNS_ROOT.exists():
#         return []
#     runs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
#     runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
#     return runs


# with tab_run:
#     st.subheader("Run")
#     st.caption("Uses Settings registries. This tab chooses a subset for the run.")

#     ensure_defaults()
#     reg = load_all()

#     # client = OpenRouterClient()
#     # api_status = client.preflight()

#     # if not api_status.has_key:
#     #     st.error("OPENROUTER_API_KEY is missing.")
#     # if api_status.has_key and not api_status.reachable:
#     #     st.error(f"OpenRouter unreachable: {api_status.error or 'unknown error'}")

#     client = OpenRouterClient()
#     api_status = client.preflight()

#     if not api_status.has_key:
#         st.error("AZURE_FOUNDRY_API_KEY is missing.")
#     if api_status.has_key and not api_status.reachable:
#         st.error(f"Azure Foundry unreachable: {api_status.error or 'unknown error'}")

#     prompt_conditions = reg.prompt_conditions.get("prompt_conditions", []) or []
#     variants = reg.variants.get("variants", []) or []
#     tests = reg.components.get("tests", []) or []

#     # st.markdown("### Configure")

#     selected_models = _load_selected_models_for_run()
#     if not selected_models:
#         st.warning("No models selected. Open Settings → Models and pick at least one model.")
#         st.stop()

#     selected_models_sorted = sorted(
#         selected_models,
#         key=lambda m: (
#             _provider(str(m.get("id") or "")).lower(),
#             str(m.get("name") or "").lower(),
#             str(m.get("id") or "").lower(),
#         ),
#     )

#     model_label_to_id: dict[str, str] = {}
#     default_labels: list[str] = []
#     for m in selected_models_sorted:
#         mid = str(m.get("id") or "").strip()
#         name = str(m.get("name") or mid).strip()
#         label = f"{name} [{mid}]"
#         model_label_to_id[label] = mid
#         default_labels.append(label)

#     sel_model_labels = st.multiselect(
#         "Models",
#         options=list(model_label_to_id.keys()),
#         default=default_labels,
#         key="run_models",
#     )
#     sel_model_ids = [model_label_to_id[x] for x in sel_model_labels]
#     st.caption(f"Using {len(sel_model_ids)} model(s).")

#     cond_label_to_id: dict[str, str] = {}
#     for c in prompt_conditions:
#         cid = (c.get("condition_id") or "").strip()
#         if not cid:
#             continue
#         name = (c.get("name") or cid).strip()
#         cond_label_to_id[f"{name} [{cid}]"] = cid
#     cond_labels = list(cond_label_to_id.keys())

#     sel_cond_labels = st.multiselect(
#         "System Instructions",
#         options=cond_labels,
#         default=cond_labels,
#         key="run_prompt_conditions",
#     )
#     sel_cond_ids = [cond_label_to_id[x] for x in sel_cond_labels]

#     variant_label_to_id: dict[str, str] = {}
#     enabled_variant_labels: list[str] = []
#     for v in variants:
#         vid = (v.get("variant_id") or "").strip()
#         if not vid:
#             continue
#         lab = (v.get("label") or vid).strip()
#         key = f"{lab} [{vid}]"
#         variant_label_to_id[key] = vid
#         enabled_variant_labels.append(key)

#     variant_labels = list(variant_label_to_id.keys())
#     sel_var_labels = st.multiselect(
#         "Request Variants",
#         options=variant_labels,
#         default=enabled_variant_labels,
#         key="run_variants",
#     )
#     sel_var_ids = [variant_label_to_id[x] for x in sel_var_labels]

#     component_titles = [t.get("title", f"Component {i+1}") for i, t in enumerate(tests)]
#     component_idx_all = list(range(len(component_titles)))

#     def _format_comp(i: int) -> str:
#         return component_titles[i]

#     sel_components = st.multiselect(
#         "Components",
#         options=component_idx_all,
#         default=component_idx_all,
#         format_func=_format_comp,
#         key="run_components",
#     )

#     repetitions = st.number_input("Repetitions per cell", 1, 50, 10, 1, key="run_reps")
#     do_score_col, do_axe_col = st.columns([1, 1], vertical_alignment="center")

#     with do_axe_col:
#         do_axe = st.checkbox("Schema", value=True, key="run_do_axe")

#     with do_score_col:
#         do_score = st.checkbox("Axe", value=True, key="run_do_score")

#     with st.expander("Parameters", expanded=False):
#         # max_tokens = st.number_input("max_tokens", 200, 4000, 1200, 100, key="run_max_tokens")
#         max_tokens = st.number_input(
#             "Maximum response length (max tokens)",
#             200,
#             4000,
#             600,
#             100,
#             key="run_max_tokens",
#             help="Limits how long the model’s output can be."
#         )
#         # temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.05, key="run_temperature")
#         temperature = st.slider(
#         "Response randomness (temperature)",
#             0.0,
#             1.5,
#             0.7,
#             0.05,
#             key="run_temperature",
#             help="Lower = more predictable and consistent. Higher = more varied and creative."
#         )
#         # top_p = st.slider("Response diversity (top_p)", 0.1, 1.0, 0.95, 0.01, key="run_top_p")
#         top_p = st.slider(
#             "Response diversity (top_p)",
#             0.1,
#             1.0,
#             0.95,
#             0.01,
#             key="run_top_p",
#             help="Lower = safer and more predictable. Higher = more varied wording."
#         )
#         rate_limit = st.number_input("Delay between calls (sec)", 0.0, 5.0, 0.3, 0.1, key="run_rate_limit")

#     run_name = st.text_input("Run name", value=f"run_{int(time.time())}", key="run_name")

#     n_cells = len(sel_model_ids) * len(sel_cond_ids) * len(sel_var_ids) * len(sel_components) * int(repetitions)
#     st.caption(f"planned generations: **{n_cells:,}**")

#     can_run = True
#     if not api_status.has_key or not api_status.reachable:
#         can_run = False
#     if not sel_model_ids or not sel_cond_ids or not sel_var_ids or not sel_components:
#         can_run = False

#     if "run_state" not in st.session_state:
#         st.session_state["run_state"] = "idle"  # "idle" | "running"
#     if "run_cancel" not in st.session_state:
#         st.session_state["run_cancel"] = False

#     def should_cancel() -> bool:
#         return bool(st.session_state.get("run_cancel", False))

#     st.markdown(
#         """
# <style>
# div[data-testid="stHorizontalBlock"] { gap: 0.35rem !important; }
# div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) div.stButton{
#   display:flex !important;
#   justify-content:flex-end !important;
#   margin-right:0 !important;
# }
# div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div.stButton{
#   display:flex !important;
#   justify-content:flex-start !important;
#   margin-left:0 !important;
# }
# </style>
# """,
#         unsafe_allow_html=True,
#     )

#     cbtn1, cbtn2, _sp = st.columns([1, 1, 8], gap="small")
#     start = False
#     stop = False

#     with cbtn1:
#         if st.session_state["run_state"] == "idle":
#             start = st.button("run", type="primary", disabled=not can_run, key="run_start_btn", use_container_width=True)
#         else:
#             st.button("run", type="primary", disabled=True, key="run_start_btn_disabled", use_container_width=True)

#     with cbtn2:
#         if st.session_state["run_state"] == "running":
#             stop = st.button("stop", type="secondary", key="run_stop_btn", use_container_width=True)
#         else:
#             st.button("stop", type="secondary", disabled=True, key="run_stop_btn_disabled", use_container_width=True)

#     if stop and st.session_state["run_state"] == "running":
#         st.session_state["run_cancel"] = True
#         st.warning("Stopping… will finish the current call, then stop.")
#         st.rerun()

#     if start:
#         st.session_state["run_state"] = "running"
#         st.session_state["run_cancel"] = False
#         st.rerun()

#     if st.session_state["run_state"] == "running":
#         cfg = RunConfig(
#             run_name=run_name,
#             out_root=OUT_ROOT,
#             model_ids=sel_model_ids,
#             condition_ids=sel_cond_ids,
#             variant_ids=sel_var_ids,
#             component_indices=sel_components,
#             repetitions=int(repetitions),
#             max_tokens=int(max_tokens),
#             temperature=float(temperature),
#             top_p=float(top_p),
#             rate_limit=float(rate_limit),
#             do_score=bool(do_score),
#             do_axe=bool(do_axe),
#             axe_profile=str(st.session_state.get("axe_profile", "form_fragment_rules")),
#             axe_config=st.session_state.get("axe_config_obj", None),
#             axe_ruleset=st.session_state.get("axe_ruleset_obj", None),
#             axe_timeout_ms=int(st.session_state.get("axe_timeout_ms", 10_000)),
#         )

#         st.caption("run config")
#         st.json(asdict(cfg), expanded=False)

#         prog = st.progress(0)
#         status = st.empty()

#         def on_progress(done: int, total: int, msg: str) -> None:
#             if total > 0:
#                 prog.progress(min(1.0, done / total))
#             status.write(msg)

#         try:
#             run_dir = run_benchmark(
#                 reg,
#                 cfg,
#                 client=client,
#                 on_progress=on_progress,
#                 should_cancel=should_cancel,
#             )

#             if should_cancel():
#                 st.warning(f"Run stopped. Saved to: {run_dir}")
#             else:
#                 st.success(f"Run complete. Saved to: {run_dir}")

#             st.session_state["last_run_dir"] = str(run_dir)

#             res_path = Path(run_dir) / "results.csv"
#             per_path = Path(run_dir) / "per_check.csv"
#             cdl1, cdl2 = st.columns(2)
#             with cdl1:
#                 if res_path.exists():
#                     st.download_button("results.csv", data=res_path.read_bytes(), file_name="results.csv")
#             with cdl2:
#                 if per_path.exists():
#                     st.download_button("per_check.csv", data=per_path.read_bytes(), file_name="per_check.csv")
#         finally:
#             st.session_state["run_state"] = "idle"
#             st.session_state["run_cancel"] = False

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
    st.markdown("#### run")
    st.caption("choose cells, then launch")

    ensure_defaults()
    reg = load_all()

    # client = OpenRouterClient()
    # api_status = client.preflight()

    # if not api_status.has_key:
    #     st.error("OPENROUTER_API_KEY is missing.")
    # if api_status.has_key and not api_status.reachable:
    #     st.error(f"OpenRouter unreachable: {api_status.error or 'unknown error'}")

    client = OpenRouterClient()
    api_status = client.preflight()

    if not api_status.has_key:
        st.error("AZURE_FOUNDRY_API_KEY is missing.")
    if api_status.has_key and not api_status.reachable:
        st.error(f"Azure Foundry unreachable: {api_status.error or 'unknown error'}")

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

    cond_label_to_id: dict[str, str] = {}
    for c in prompt_conditions:
        cid = (c.get("condition_id") or "").strip()
        if not cid:
            continue
        name = (c.get("name") or cid).strip()
        cond_label_to_id[f"{name} [{cid}]"] = cid
    cond_labels = list(cond_label_to_id.keys())

    component_titles = [t.get("title", f"Component {i+1}") for i, t in enumerate(tests)]
    component_idx_all = list(range(len(component_titles)))

    def _format_comp(i: int) -> str:
        return component_titles[i]

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

    left_note, right_note = st.columns([1.15, 0.85], gap="large", vertical_alignment="top")

    with left_note:
        st.markdown('<div class="note-section"><div class="note-kicker">choose</div><div class="note-hint">models, instructions, components, variants</div></div>', unsafe_allow_html=True)
        use_all_models = st.checkbox("all models", value=True, key="run_all_models")
        if use_all_models:
            sel_model_labels = default_labels
            sel_model_ids = [model_label_to_id[x] for x in sel_model_labels]
            st.caption(f"{len(sel_model_ids)} model(s)")
        else:
            sel_model_labels = st.multiselect(
                "models",
                options=list(model_label_to_id.keys()),
                default=[],
                key="run_models",
                placeholder="choose models",
            )
            sel_model_ids = [model_label_to_id[x] for x in sel_model_labels]
            st.caption(f"{len(sel_model_ids)} model(s)")

        use_all_instructions = st.checkbox("all instructions", value=True, key="run_all_instructions")
        if use_all_instructions:
            sel_cond_labels = cond_labels
            sel_cond_ids = [cond_label_to_id[x] for x in sel_cond_labels]
            st.caption(f"{len(sel_cond_ids)} instruction set(s)")
        else:
            sel_cond_labels = st.multiselect(
                "system instruction",
                options=cond_labels,
                default=[],
                key="run_prompt_conditions",
                placeholder="choose instructions",
            )
            sel_cond_ids = [cond_label_to_id[x] for x in sel_cond_labels]
            st.caption(f"{len(sel_cond_ids)} instruction set(s)")

        use_all_components = st.checkbox("all components", value=True, key="run_all_components")
        if use_all_components:
            sel_components = component_idx_all
            st.caption(f"{len(sel_components)} component(s)")
        else:
            sel_components = st.multiselect(
                "components",
                options=component_idx_all,
                default=[],
                format_func=_format_comp,
                key="run_components",
                placeholder="choose components",
            )
            st.caption(f"{len(sel_components)} component(s)")

        use_all_variants = st.checkbox("all variants", value=True, key="run_all_variants")
        if use_all_variants:
            sel_var_labels = enabled_variant_labels
            sel_var_ids = [variant_label_to_id[x] for x in sel_var_labels]
            st.caption(f"{len(sel_var_ids)} variant(s)")
        else:
            sel_var_labels = st.multiselect(
                "variants",
                options=variant_labels,
                default=[],
                key="run_variants",
                placeholder="choose variants",
            )
            sel_var_ids = [variant_label_to_id[x] for x in sel_var_labels]
            st.caption(f"{len(sel_var_ids)} variant(s)")

    with right_note:
        st.markdown('<div class="note-section"><div class="note-kicker">launch</div><div class="note-hint">name, repeats, scoring</div></div>', unsafe_allow_html=True)
        run_name = st.text_input("name", value=f"run_{int(time.time())}", key="run_name")
        repetitions = st.number_input("repeats", 1, 50, 10, 1, key="run_reps")
        max_tokens = st.number_input(
            "max tokens",
            200,
            4000,
            600,
            100,
            key="run_max_tokens",
            help="Limits how long the model output can be."
        )
        temperature = st.slider(
            "temperature",
            0.0,
            1.5,
            0.7,
            0.05,
            key="run_temperature",
            help="Lower means steadier outputs. Higher means more variation."
        )
        top_p = st.slider(
            "top p",
            0.1,
            1.0,
            0.95,
            0.01,
            key="run_top_p",
            help="Lower means narrower sampling. Higher means more varied wording."
        )
        rate_limit = st.number_input("pause", 0.0, 5.0, 0.3, 0.1, key="run_rate_limit")
        st.caption("score with")
        score_a, score_b = st.columns(2, vertical_alignment="center")
        with score_a:
            do_axe = st.checkbox("axe", value=True, key="run_do_axe")
        with score_b:
            do_score = st.checkbox("schema", value=True, key="run_do_score")

    n_cells = len(sel_model_ids) * len(sel_cond_ids) * len(sel_var_ids) * len(sel_components) * int(repetitions)
    st.caption(f"planned generations: **{n_cells:,}**")

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

    st.markdown('<div id="run-buttons-anchor"></div>', unsafe_allow_html=True)
    cbtn1, cbtn2, _sp = st.columns([0.085, 0.085, 0.83], gap="small")
    start = False
    stop = False

    with cbtn1:
        if st.session_state["run_state"] == "idle":
            start = st.button("run", type="primary", disabled=not can_run, key="run_start_btn", use_container_width=True)
        else:
            st.button("run", type="primary", disabled=True, key="run_start_btn_disabled", use_container_width=True)

    with cbtn2:
        if st.session_state["run_state"] == "running":
            stop = st.button("stop", type="secondary", key="run_stop_btn", use_container_width=True)
        else:
            st.button("stop", type="secondary", disabled=True, key="run_stop_btn_disabled", use_container_width=True)

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

        st.caption("run config")
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
                    st.download_button("results.csv", data=res_path.read_bytes(), file_name="results.csv")
            with cdl2:
                if per_path.exists():
                    st.download_button("per_check.csv", data=per_path.read_bytes(), file_name="per_check.csv")
        finally:
            st.session_state["run_state"] = "idle"
            st.session_state["run_cancel"] = False

# ===============
# Results
# ===============
with tab_results:
    st.markdown("#### results")

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
    #             grid=False,
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
                grid=False,
                tickSize=4,
                titlePadding=10,
                labelPadding=6,
                labelLimit=300,
            )
            .configure_legend(titleFontSize=12, labelFontSize=12, orient="bottom", padding=10)
            .configure_view(stroke=None)
        )
    


    def _axis_y01(title: str) -> alt.Axis:
        return alt.Axis(title=title, values=Y01_TICKS, format=".1f", grid=False, labelOverlap=False, labelFlush=False)

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
        # Keep the filter compact: empty means all, so Streamlit does not render dozens of selected tags.
        picked = st.multiselect(label, options=opts, default=[], placeholder=f"all {label}")
        if not picked:
            return df
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
        g["lo"] = (g["mean"] - g["ci95"]).clip(lower=0.0, upper=1.0)
        g["hi"] = (g["mean"] + g["ci95"]).clip(lower=0.0, upper=1.0)
        g["sd_lo"] = g["sd_lo"].clip(lower=0.0, upper=1.0)
        g["sd_hi"] = g["sd_hi"].clip(lower=0.0, upper=1.0)
        g["se_lo"] = g["se_lo"].clip(lower=0.0, upper=1.0)
        g["se_hi"] = g["se_hi"].clip(lower=0.0, upper=1.0)
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
        g["lo"] = (g["mean"] - g["ci95"]).clip(lower=0.0, upper=1.0)
        g["hi"] = (g["mean"] + g["ci95"]).clip(lower=0.0, upper=1.0)
        g["sd_lo"] = g["sd_lo"].clip(lower=0.0, upper=1.0)
        g["sd_hi"] = g["sd_hi"].clip(lower=0.0, upper=1.0)
        g["se_lo"] = g["se_lo"].clip(lower=0.0, upper=1.0)
        g["se_hi"] = g["se_hi"].clip(lower=0.0, upper=1.0)
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

    CAP_SIZE = 8
    CAP_THICKNESS = 1
    RULE_THICKNESS = 1

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

    def _safe_sort(df: pd.DataFrame, by: str | list[str], ascending: bool | list[bool] = True) -> pd.DataFrame:
        """Sort a dataframe only when the requested column(s) exist.

        Some filtered result tables can be empty or can lose summary columns after
        aggregation. Streamlit should still render the table instead of crashing.
        """
        if df is None or df.empty:
            return df
        cols = [by] if isinstance(by, str) else list(by)
        if not all(col in df.columns for col in cols):
            return df
        try:
            return df.sort_values(by, ascending=ascending)
        except Exception:
            return df

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
        y_title: str = "score",
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
            xoff = alt.XOffset(f"{series_col}:N", scale=alt.Scale(paddingInner=0.08, paddingOuter=0.45))

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
        return int(st.slider("components shown", lo, hi, min(default_k, hi)))

    source = st.radio(
        "results source",
        ["last run", "pick folder", "upload csv"],
        horizontal=True,
        index=0,
        key="results_source",
        label_visibility="collapsed",
        format_func=lambda x: {"last run": "latest", "pick folder": "folder", "upload csv": "csv"}.get(x, x),
    )

    csv_bytes: Optional[bytes] = None
    csv_label: str = ""

    if source == "last run":
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

    elif source == "pick folder":
        runs = _list_runs()
        if not runs:
            st.info("No runs found under output/_runs/.")
        else:
            labels = [
                f"{p.name}  ·  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))}"
                for p in runs
            ]
            label_to_path = {labels[i]: runs[i] for i in range(len(runs))}
            pick = st.selectbox("folder", options=labels, index=0, key="results_pick_run")
            rp = label_to_path[pick]
            p = rp / "results.csv"
            if p.exists():
                csv_bytes = p.read_bytes()
                csv_label = str(p)
            else:
                st.error(f"Missing results.csv in: {rp}")

    else:
        up = st.file_uploader("upload results.csv", type=["csv"], accept_multiple_files=False, key="results_uploader")
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
    c1.metric("rows", k["rows"])
    c2.metric("valid", k["pct_ok"])
    c3.metric("mean", k["mean_norm_ok"])
    c4.metric("cost", k["total_cost"])

    st.write("")
    st.caption("filters")
    df = res_df.copy()

    if "ok" in df.columns:
        ok_only = st.checkbox("ok rows only", value=True, key="results_ok_only")
        if ok_only:
            df = df[df["ok"] == True]  # noqa: E712

    left, right = st.columns(2)
    with left:
        df = _apply_multiselect(df, "model_name" if "model_name" in df.columns else "model_id", "model", default_all=True)
        df = _apply_multiselect(
            df,
            "condition_name" if "condition_name" in df.columns else "condition_id",
            "system instruction",
            default_all=True,
        )
    with right:
        df = _apply_multiselect(df, "variant_label" if "variant_label" in df.columns else "variant_id", "variant", default_all=True)
        df = _apply_multiselect(df, "component_title", "component", default_all=True)

    st.caption(f"{len(df)} rows after filters")
    if ("model_id" in df.columns) and (df["model_id"].notna().any()):
        st.caption(f"{int(df['model_id'].nunique(dropna=True))} model(s)")

    if df.empty:
        st.warning("Nothing left after filters.")
        st.stop()

    st.write("")
    ctrl_l, ctrl_r = st.columns([.48,.52], vertical_alignment="center")
    with ctrl_l:
        scheme = st.selectbox(
        "color",
        options=COLOR_SCHEMES,
        index=COLOR_SCHEMES.index(DEFAULT_SCHEME),
        key="results_color_scheme",
    )
    with ctrl_r:
        error_mode = st.radio("error bars", ["ci95", "se", "sd", "none"], horizontal=True, index=0, key="results_error_mode")

    st.write("")
    df_ok = df
    if "ok" in df_ok.columns:
        df_ok = df_ok[df_ok["ok"] == True]  # noqa: E712

    # ----------------------------
    # Tabs: accessibility score vs axe score
    # ----------------------------
    tab_axe, tab_acc = st.tabs(["axe", "schema"])

    # ============================
    # Accessibility score (norm_score) — EXACT SAME CONTENT
    # ============================
   
    with tab_acc:
        st.caption("schema")
        st.write("")
        need_schema = {"norm_score", "raw_score", "max_score"}
        if not need_schema.issubset(set(df_ok.columns)):
            st.info(
                "No schema columns found in results.csv. "
            )
        else:
            st.caption("model")

            if ("model_name" in df_ok.columns) and ("norm_score" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                s_model = _safe_sort(_mean_stats(df_ok, ["model_name"]), "mean", ascending=False)
                _grouped_bar_with_error(
                    s_model,
                    x_col="model_name",
                    series_col=None,
                    title="schema by model",
                    error_mode=error_mode,
                    scheme=scheme,
                    x_title="model",
                    y_title="score",
                    label_angle=-35,
                    height=480,
                )
                st.dataframe(_pretty_table(s_model), use_container_width=True, height=240)
            else:
                st.caption("Need ≥2 models and columns: model_name (or model_id) and norm_score.")

            st.write("")
            st.caption("system instruction")

            rq1_split = st.checkbox("split by model", value=False, key="rq1_split_models")

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
                            title="schema by system instruction · model",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="System instruction",
                            series_title="model",
                            y_title="score",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "System instruction", "model_name": "model"})
                        st.dataframe(_pretty_table(_safe_sort(tbl, ["System instruction", "model"])), use_container_width=True, height=260)
                    else:
                        s = _mean_stats(df_ok, ["condition_name"])
                        _grouped_bar_with_error(
                            s,
                            x_col="condition_name",
                            series_col=None,
                            title="schema by system instruction",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="System instruction",
                            y_title="score",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "System instruction"})
                        st.dataframe(_pretty_table(_safe_sort(tbl, "mean", ascending=False)), use_container_width=True, height=260)
            else:
                st.caption("Need condition_name and norm_score.")

            st.write("")
            st.caption("variant")

            rq2_split = st.checkbox("split by model", value=False, key="rq2_split_models")

            need_rq2 = {"variant_label", "norm_score"}
            if need_rq2.issubset(set(df_ok.columns)):
                if rq2_split and ("model_name" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                    s = _mean_stats(df_ok, ["variant_label", "model_name"])
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col="model_name",
                        title="schema by variant · model",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="variant",
                        series_title="model",
                        height=480,
                    )
                    st.dataframe(_pretty_table(_safe_sort(s, ["variant_label", "model_name"])), use_container_width=True, height=300)
                else:
                    s = _mean_stats(df_ok, ["variant_label"])
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col=None,
                        title="schema by variant",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="variant",
                        height=480,
                    )
                    st.dataframe(_pretty_table(_safe_sort(s, "mean", ascending=False)), use_container_width=True, height=300)
            else:
                st.caption("Need variant_label and norm_score.")

            st.write("")
            st.subheader("system instruction × variant")

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
                            x=alt.X("variant_label:N", title="variant", axis=_axis_cat("variant", angle=-35)),
                            y=alt.Y("condition_name:N", title="System instruction", axis=_axis_cat("System instruction", angle=0)),
                            color=alt.Color(
                                "mean:Q",
                                title="mean score",
                                scale=alt.Scale(scheme="viridis", domain=[0.0, 1.0]),
                            ),
                            tooltip=[
                                alt.Tooltip("condition_name:N", title="System instruction"),
                                alt.Tooltip("variant_label:N", title="variant"),
                                alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                                alt.Tooltip("std:Q", title="SD", format=".3f"),
                                alt.Tooltip("se:Q", title="SE", format=".3f"),
                                alt.Tooltip("ci95:Q", title="95% CI half-width", format=".3f"),
                                alt.Tooltip("n:Q", title="n"),
                            ],
                        )
                        .properties(height=420)
                    )
                    heat = _apply_layout(heat, "System instruction × variant", pad=_chart_pad(top=46, bottom=70, left=12, right=12))
                    st.altair_chart(_base_config(heat), use_container_width=True)
                    st.dataframe(_pretty_table(s.sort_values(["condition_name", "variant_label"])), use_container_width=True, height=280)
            else:
                st.caption("Need condition_name, variant_label, norm_score.")

            st.write("")
            st.subheader("component")

            if "component_title" in df_ok.columns and "norm_score" in df_ok.columns:
                tmp = df_ok.copy()
                tmp["norm_score"] = pd.to_numeric(tmp["norm_score"], errors="coerce")
                tmp = tmp.dropna(subset=["norm_score"])
                if tmp.empty:
                    st.caption("No data.")
                else:
                    comp_means = tmp.groupby("component_title")["norm_score"].mean().sort_values(ascending=False)
                    component_options = comp_means.index.tolist()

                    show_all_components = st.checkbox(
                        "show all components",
                        value=False,
                        key="rq4_show_all_components",
                    )

                    if show_all_components:
                        selected_components = component_options
                        st.caption(f"{len(selected_components)} component(s)")
                    else:
                        default_components = component_options[: min(20, len(component_options))]
                        selected_components = st.multiselect(
                            "components to compare",
                            options=component_options,
                            default=default_components,
                            key="rq4_components_filter",
                        )
                        st.caption(f"{len(selected_components)} component(s)")

                    if not selected_components:
                        st.caption("Choose at least one component.")
                        st.stop()

                    tmp = tmp[tmp["component_title"].isin(selected_components)]
                    comp_means = tmp.groupby("component_title")["norm_score"].mean().sort_values(ascending=False)

                    rq4_split = st.checkbox("split by model", value=False, key="rq4_split_checkbox")

                    if rq4_split and ("model_name" in tmp.columns) and (tmp["model_name"].nunique() > 1):
                        s = _mean_stats(tmp, ["component_title", "model_name"])
                        order = comp_means.index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values(["component_title", "model_name"])
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col="model_name",
                            title="schema by component (grouped by model)",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="component",
                            series_title="model",
                            y_title="score",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(s), use_container_width=True, height=320)
                    else:
                        s = _mean_stats(tmp, ["component_title"])
                        order = comp_means.index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values("component_title")
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col=None,
                            title="schema by component",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="component",
                            y_title="score",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(_safe_sort(s, "mean", ascending=False)), use_container_width=True, height=320)
            else:
                st.caption("Need component_title and norm_score.")


            # st.subheader("component")
            # st.caption("component")

            # if "component_title" in df_ok.columns and "norm_score" in df_ok.columns:
            #     tmp = df_ok.copy()
            #     tmp["norm_score"] = pd.to_numeric(tmp["norm_score"], errors="coerce")
            #     tmp = tmp.dropna(subset=["norm_score"])
            #     if tmp.empty:
            #         st.caption("No data.")
            #     else:
            #         comp_means = tmp.groupby("component_title")["norm_score"].mean().sort_values(ascending=False)
            #         top_k = _safe_slider_topk(int(len(comp_means)), default_k=20, cap=60)
            #         keep = set(comp_means.head(top_k).index.tolist())
            #         tmp = tmp[tmp["component_title"].isin(keep)]
            #         rq4_split = st.checkbox("split by model", value=False, key="rq4_split_checkbox")

            #         if rq4_split and ("model_name" in tmp.columns) and (tmp["model_name"].nunique() > 1):
            #             s = _mean_stats(tmp, ["component_title", "model_name"])
            #             order = comp_means.head(top_k).index.tolist()
            #             s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
            #             s = s.sort_values(["component_title", "model_name"])
            #             _grouped_bar_with_error(
            #                 s,
            #                 x_col="component_title",
            #                 series_col="model_name",
            #                 title="schema by component (grouped by model)",
            #                 error_mode=error_mode,
            #                 scheme=scheme,
            #                 x_title="component",
            #                 series_title="model",
            #                 y_title="score",
            #                 label_angle=-35,
            #                 height=560,
            #             )
            #             st.dataframe(_pretty_table(s), use_container_width=True, height=320)
            #         else:
            #             s = _mean_stats(tmp, ["component_title"])
            #             order = comp_means.head(top_k).index.tolist()
            #             s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
            #             s = s.sort_values("component_title")
            #             _grouped_bar_with_error(
            #                 s,
            #                 x_col="component_title",
            #                 series_col=None,
            #                 title="schema by component",
            #                 error_mode=error_mode,
            #                 scheme=scheme,
            #                 x_title="component",
            #                 y_title="score",
            #                 label_angle=-35,
            #                 height=560,
            #             )
            #             st.dataframe(_pretty_table(_safe_sort(s, "mean", ascending=False)), use_container_width=True, height=320)
            # else:
            #     st.caption("Need component_title and norm_score.")

            
            # st.caption("preview")
            # st.dataframe(df.head(400), use_container_width=True, height=380)

            
    # ============================
    # Axe score (axe_score)
    # ============================
    with tab_axe:
        st.caption("axe")
        st.write("")

        if "axe_score" not in df_ok.columns:
            st.info("No axe_score column found in results.csv.")
        else:         

            st.caption("model")

            if ("model_name" in df_ok.columns) and ("axe_score" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                s_model = _safe_sort(_mean_stats_value(df_ok, ["model_name"], "axe_score"), "mean", ascending=False)
                _grouped_bar_with_error(
                    s_model,
                    x_col="model_name",
                    series_col=None,
                    title="axe by model",
                    error_mode=error_mode,
                    scheme=scheme,
                    x_title="model",
                    y_title="axe score",
                    label_angle=-35,
                    height=480,
                )
                st.dataframe(_pretty_table(s_model), use_container_width=True, height=240)
            else:
                st.caption("Need ≥2 models and columns: model_name (or model_id) and axe_score.")

            st.write("")
            st.caption("system instruction")

            axe_rq1_split = st.checkbox("split by model", value=False, key="axe_rq1_split_models")

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
                            title="axe by system instruction · model",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="System instruction",
                            series_title="model",
                            y_title="Axe score",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "System instruction", "model_name": "model"})
                        st.dataframe(_pretty_table(_safe_sort(tbl, ["System instruction", "model"])), use_container_width=True, height=260)
                    else:
                        s = _mean_stats_value(df_ok, ["condition_name"], "axe_score")
                        _grouped_bar_with_error(
                            s,
                            x_col="condition_name",
                            series_col=None,
                            title="axe by system instruction",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="System instruction",
                            y_title="axe score",
                            label_angle=-35,
                            height=460,
                        )
                        tbl = s.rename(columns={"condition_name": "System instruction"})
                        st.dataframe(_pretty_table(_safe_sort(tbl, "mean", ascending=False)), use_container_width=True, height=260)
            else:
                st.caption("Need condition_name and axe_score.")

            st.write("")
            st.caption("variant")

            axe_rq2_split = st.checkbox("split by model", value=False, key="axe_rq2_split_models")

            need_rq2_axe = {"variant_label", "axe_score"}
            if need_rq2_axe.issubset(set(df_ok.columns)):
                if axe_rq2_split and ("model_name" in df_ok.columns) and (df_ok["model_name"].nunique() > 1):
                    s = _mean_stats_value(df_ok, ["variant_label", "model_name"], "axe_score")
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col="model_name",
                        title="axe by variant · model",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="variant",
                        series_title="model",
                        y_title="axe score",
                        height=480,
                    )
                    st.dataframe(_pretty_table(_safe_sort(s, ["variant_label", "model_name"])), use_container_width=True, height=300)
                else:
                    s = _mean_stats_value(df_ok, ["variant_label"], "axe_score")
                    _grouped_bar_with_error(
                        s,
                        x_col="variant_label",
                        series_col=None,
                        title="axe by variant",
                        error_mode=error_mode,
                        scheme=scheme,
                        x_title="variant",
                        y_title="axe score",
                        height=480,
                    )
                    st.dataframe(_pretty_table(_safe_sort(s, "mean", ascending=False)), use_container_width=True, height=300)
            else:
                st.caption("Need variant_label and axe_score.")

            st.write("")
            st.subheader("system instruction × variant")

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
                            x=alt.X("variant_label:N", title="variant", axis=_axis_cat("variant", angle=-35)),
                            y=alt.Y("condition_name:N", title="System instruction", axis=_axis_cat("System instruction", angle=0)),
                            color=alt.Color(
                                "mean:Q",
                                title="mean axe",
                                scale=alt.Scale(scheme="viridis", domain=[0.0, 1.0]),
                            ),
                            tooltip=[
                                alt.Tooltip("condition_name:N", title="System instruction"),
                                alt.Tooltip("variant_label:N", title="variant"),
                                alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                                alt.Tooltip("std:Q", title="SD", format=".3f"),
                                alt.Tooltip("se:Q", title="SE", format=".3f"),
                                alt.Tooltip("ci95:Q", title="95% CI half-width", format=".3f"),
                                alt.Tooltip("n:Q", title="n"),
                            ],
                        )
                        .properties(height=420)
                    )
                    heat = _apply_layout(heat, "System instruction × variant", pad=_chart_pad(top=46, bottom=70, left=12, right=12))
                    st.altair_chart(_base_config(heat), use_container_width=True)
                    st.dataframe(_pretty_table(s.sort_values(["condition_name", "variant_label"])), use_container_width=True, height=280)
            else:
                st.caption("Need condition_name, variant_label, axe_score.")

            st.write("")
            # st.subheader("component")
            # st.caption("component")

            # if "component_title" in df_ok.columns and "axe_score" in df_ok.columns:
            #     tmp = df_ok.copy()
            #     tmp["axe_score"] = pd.to_numeric(tmp["axe_score"], errors="coerce")
            #     tmp = tmp.dropna(subset=["axe_score"])
            #     if tmp.empty:
            #         st.caption("No data.")
            #     else:
            #         comp_means = tmp.groupby("component_title")["axe_score"].mean().sort_values(ascending=False)
            #         top_k = _safe_slider_topk(int(len(comp_means)), default_k=20, cap=60)
            #         keep = set(comp_means.head(top_k).index.tolist())
            #         tmp = tmp[tmp["component_title"].isin(keep)]
            #         axe_rq4_split = st.checkbox("split by model", value=False, key="axe_rq4_split_checkbox")

            #         if axe_rq4_split and ("model_name" in tmp.columns) and (tmp["model_name"].nunique() > 1):
            #             s = _mean_stats_value(tmp, ["component_title", "model_name"], "axe_score")
            #             order = comp_means.head(top_k).index.tolist()
            #             s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
            #             s = s.sort_values(["component_title", "model_name"])
            #             _grouped_bar_with_error(
            #                 s,
            #                 x_col="component_title",
            #                 series_col="model_name",
            #                 title="axe by component (grouped by model)",
            #                 error_mode=error_mode,
            #                 scheme=scheme,
            #                 x_title="component",
            #                 series_title="model",
            #                 y_title="axe score",
            #                 label_angle=-35,
            #                 height=560,
            #             )
            #             st.dataframe(_pretty_table(s), use_container_width=True, height=320)
            #         else:
            #             s = _mean_stats_value(tmp, ["component_title"], "axe_score")
            #             order = comp_means.head(top_k).index.tolist()
            #             s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
            #             s = s.sort_values("component_title")
            #             _grouped_bar_with_error(
            #                 s,
            #                 x_col="component_title",
            #                 series_col=None,
            #                 title="axe by component",
            #                 error_mode=error_mode,
            #                 scheme=scheme,
            #                 x_title="component",
            #                 y_title="axe score",
            #                 label_angle=-35,
            #                 height=560,
            #             )
            #             st.dataframe(_pretty_table(_safe_sort(s, "mean", ascending=False)), use_container_width=True, height=320)
            # else:
            #     st.caption("Need component_title and axe_score.")

            st.subheader("component")

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
                    comp_means = tmp.groupby("component_title")["axe_score"].mean().sort_values(ascending=False)

                    axe_rq4_split = st.checkbox("split by model", value=False, key="axe_rq4_split_checkbox")

                    if axe_rq4_split and ("model_name" in tmp.columns) and (tmp["model_name"].nunique() > 1):
                        s = _mean_stats_value(tmp, ["component_title", "model_name"], "axe_score")
                        order = comp_means.index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values(["component_title", "model_name"])
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col="model_name",
                            title="axe by component (grouped by model)",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="component",
                            series_title="model",
                            y_title="axe score",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(s), use_container_width=True, height=320)
                    else:
                        s = _mean_stats_value(tmp, ["component_title"], "axe_score")
                        order = comp_means.index.tolist()
                        s["component_title"] = pd.Categorical(s["component_title"], categories=order, ordered=True)
                        s = s.sort_values("component_title")
                        _grouped_bar_with_error(
                            s,
                            x_col="component_title",
                            series_col=None,
                            title="axe by component",
                            error_mode=error_mode,
                            scheme=scheme,
                            x_title="component",
                            y_title="axe score",
                            label_angle=-35,
                            height=560,
                        )
                        st.dataframe(_pretty_table(_safe_sort(s, "mean", ascending=False)), use_container_width=True, height=320)
            else:
                st.caption("Need component_title and axe_score.")

                        
            # preview intentionally omitted to keep the results page compact


