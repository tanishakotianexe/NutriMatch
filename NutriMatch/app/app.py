"""NutriMatch - Nutrient-Targeted Food Recommendation Website (single file).

Pages (top navigation): Home | Report Reader | Graphs and Statistics | Predictor | My Daily Nutrition
Everything (theme, components, pages) lives in this one file, so there is no
theme.py / components.py to keep in sync.

Run:  streamlit run app/app.py
Outputs written by the app: data/diary.csv (the saved food diary).
data/model_results.csv + data/*.joblib are written by scripts/train_models.py.
Optional images: put hero_blur_bg.jpg (texture) or hero_photo.jpg (photo that
replaces the illustration) in the project's assets/ folder.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import os
import joblib
import re
from difflib import SequenceMatcher
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dishes_engineered.csv"
RESULTS = ROOT / "data" / "model_results.csv"
MODEL_DIR = ROOT / "data"  # saved .joblib models are currently stored in data/
RECIPE_BOOK = ROOT / "data" / "Cleaned_Indian_Food_Dataset.csv"   # Indian recipe dataset (put it in the data/ folder)
ASSETS = ROOT / "assets"

NUTRIENTS = ["Calories (kcal)", "Carbohydrates (g)", "Protein (g)", "Fats (g)",
             "Free Sugar (g)", "Fibre (g)", "Sodium (mg)", "Calcium (mg)",
             "Iron (mg)", "Vitamin C (mg)", "Folate (ug)"]
LABELS = {"Sodium (mg)": "sodium", "Iron (mg)": "iron", "Calcium (mg)": "calcium",
          "Protein (g)": "protein", "Fibre (g)": "fibre", "Vitamin C (mg)": "vitamin C",
          "Folate (ug)": "folate", "Calories (kcal)": "calories",
          "Carbohydrates (g)": "carbohydrates", "Fats (g)": "fat",
          "Free Sugar (g)": "free sugar"}
UNITS = {"Calories (kcal)": "kcal", "Carbohydrates (g)": "g", "Protein (g)": "g",
         "Fats (g)": "g", "Free Sugar (g)": "g", "Fibre (g)": "g",
         "Sodium (mg)": "mg", "Calcium (mg)": "mg", "Iron (mg)": "mg",
         "Vitamin C (mg)": "mg", "Folate (ug)": "ug"}

PAGES = ["Home", "Report Reader", "Graphs and Statistics", "Predictor", "My Daily Nutrition"]
PRED_VIEWS = ["Cards", "Recipe", "Table"]

LOGO_PATH = ASSETS / "logo.png"     # put your logo here: <project>/assets/logo.png


def _logo_b64() -> str:
    """Logo as base64 for the top bar ('' if the file is missing)."""
    try:
        return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except OSError:
        return ""


def _favicon():
    """Browser-tab icon from the same file. None (default icon) if missing or unreadable."""
    try:
        from PIL import Image
        return Image.open(LOGO_PATH)
    except Exception:
        return None


st.set_page_config(page_title="NutriMatch", page_icon=_favicon(), layout="wide",
                   initial_sidebar_state="collapsed")

# ==========================================================================
# THEME
# ==========================================================================
PALETTES = {
    "Light": {
        "bg": "#f3f1e7", "surface": "#ffffff", "glass": "rgba(255,255,255,.68)",
        "surface2": "#ebf0e4", "primary": "#4a7152", "primary_hover": "#3b5c43",
        "primary_soft": "#e1ebd9", "on_primary": "#ffffff", "accent": "#e9b44c", "accent_ink": "#2b2105",
        "text": "#1e2a21", "muted": "#637265", "border": "rgba(55,85,60,.16)",
        "grid": "#e2e9da", "success": "#3f8a57", "warning": "#b57d1c",
        "info": "#3b6f8f", "danger": "#b9503d",
        "shadow": "0 1px 2px rgba(30,50,35,.05), 0 8px 24px rgba(30,50,35,.06)",
        "shadow_lg": "0 18px 44px rgba(30,50,35,.16)",
        "hero1": "#d4e3c8", "hero2": "#f3efdf",
        "blob1": "rgba(138,172,125,.42)", "blob2": "rgba(233,180,76,.24)",
        "blob3": "rgba(120,160,130,.26)", "dot": "rgba(60,90,65,.11)",
    },
    "Dark": {
        "bg": "#0f1511", "surface": "#17201a", "glass": "rgba(26,36,29,.70)",
        "surface2": "#1d2921", "primary": "#9cc9a4", "primary_hover": "#b4d8ba",
        "primary_soft": "#223428", "on_primary": "#0f1511", "accent": "#efc062", "accent_ink": "#2b2105",
        "text": "#e6eee8", "muted": "#8fa493", "border": "rgba(160,200,170,.17)",
        "grid": "#1f2c24", "success": "#7fd1a0", "warning": "#e0b25a",
        "info": "#7fb0cf", "danger": "#e09484",
        "shadow": "0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25)",
        "shadow_lg": "0 18px 44px rgba(0,0,0,.5)",
        "hero1": "#1d3226", "hero2": "#131c16",
        "blob1": "rgba(80,130,95,.32)", "blob2": "rgba(239,192,98,.10)",
        "blob3": "rgba(70,120,100,.24)", "dot": "rgba(160,200,170,.08)",
    },
}
COLORWAY = {
    "Light": ["#4a7152", "#e9b44c", "#3b6f8f", "#b9503d", "#8fae86", "#8a6d3b"],
    "Dark": ["#9cc9a4", "#efc062", "#7fb0cf", "#e09484", "#6a9a7a", "#c2a266"],
}
HEAT_SCALE = {
    "Light": ["#f3f1e7", "#bcd6b0", "#4a7152"],
    "Dark": ["#17201a", "#2e5a3e", "#9cc9a4"],
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: __BG__; --surface: __SURFACE__; --glass: __GLASS__; --surface2: __SURFACE2__;
  --primary: __PRIMARY__; --primary-hover: __PRIMARY_HOVER__; --primary-soft: __PRIMARY_SOFT__; --on-primary: __ON_PRIMARY__;
  --accent: __ACCENT__; --accent-ink: __ACCENT_INK__;
  --text: __TEXT__; --muted: __MUTED__; --border: __BORDER__; --grid: __GRID__;
  --success: __SUCCESS__; --warning: __WARNING__; --info: __INFO__; --danger: __DANGER__;
  --shadow: __SHADOW__; --shadow-lg: __SHADOW_LG__;
  --hero1: __HERO1__; --hero2: __HERO2__;
  --blob1: __BLOB1__; --blob2: __BLOB2__; --blob3: __BLOB3__; --dot: __DOT__;
  --display: 'Fraunces', Georgia, serif;
}
* { box-sizing: border-box; }

.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
.stApp button, .stApp select, .stApp td, .stApp th {
  font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

/* ---------- atmospheric background: soft sage glows + dot grid ---------- */
[data-testid="stAppViewContainer"] {
  color: var(--text) !important;
  background:
    radial-gradient(var(--dot) 1px, transparent 1.3px) 0 0 / 24px 24px,
    radial-gradient(900px 520px at 4% -6%, var(--blob1), transparent 62%),
    radial-gradient(760px 480px at 104% 6%, var(--blob2), transparent 60%),
    radial-gradient(1000px 640px at 50% 110%, var(--blob3), transparent 62%),
    var(--bg) !important;
  background-attachment: fixed !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }

.block-container, [data-testid="stMain"] .block-container {
  max-width: 1200px !important; margin: 0 auto !important; padding: 1.4rem 3.2rem 5rem !important;
}

/* ---------- typography ---------- */
h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
  font-family: var(--display) !important; color: var(--text) !important;
  letter-spacing: -0.02em; font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] h3 { font-size: 1.4rem !important; }
[data-testid="stMarkdownContainer"] h4 { font-size: 1.1rem !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text); line-height: 1.65; }
[data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }
[data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 600; font-size: .88rem; }
hr, [data-testid="stDivider"] hr { border-color: var(--border) !important; }

/* ---------- top bar ---------- */
[data-testid="stHorizontalBlock"]:has(.brand) {
  align-items: center; padding: .55rem .9rem; border-radius: 999px;
  background: var(--glass); border: 1px solid var(--border);
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  box-shadow: var(--shadow); margin-bottom: 1.6rem;
}
.brand { display: flex; align-items: center; gap: .7rem; padding-left: .3rem; line-height: 1; }
.brand-mark { width: 40px; height: 40px; border-radius: 13px; background: var(--primary); color: var(--on-primary); display: flex; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 4px 12px color-mix(in srgb, var(--primary) 30%, transparent); }
.brand-mark .ico { width: 24px; height: 24px; vertical-align: 0; }
.brand-name { font-family: var(--display); font-weight: 700; font-size: 1.35rem; letter-spacing: -0.03em; color: var(--text); }
.brand-logo { height: 44px; width: 44px; object-fit: contain; display: block; }
/* centre every part of the top bar on one line */
[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"] { display: flex; flex-direction: column; justify-content: center; }
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stElementContainer"] { margin: 0; }

/* Dark mode toggle container */
[data-testid="stToggle"] {
    width: 100%;
    min-width: 105px;
    padding: 0;
    margin: 0;
}

[data-testid="stToggle"] label {
    color: var(--text) !important;
    font-size: 0.88rem !important;
    white-space: nowrap;
}

/* Toggle switch */
[data-testid="stToggle"] [role="switch"] {
    min-width: 40px;
    height: 22px;
    border-radius: 999px;
    transition: background-color 0.2s ease;
}

[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background-color: var(--primary) !important;
}

[data-testid="stToggle"] [role="switch"][aria-checked="false"] {
    background-color: var(--muted) !important;
}

/* ---------- inputs ---------- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background-color: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; color: var(--text) !important; min-height: 46px;
  transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover, [data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover { border-color: var(--primary) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within, [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
  border-color: var(--primary) !important; box-shadow: 0 0 0 4px color-mix(in srgb, var(--primary) 18%, transparent) !important;
}
[data-baseweb="select"] input, [data-testid="stSelectbox"] span, [data-testid="stMultiSelect"] span { color: var(--text) !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] { background-color: var(--primary-soft) !important; border-radius: 999px; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: var(--primary) !important; font-weight: 600; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px; box-shadow: var(--shadow-lg);
}
[data-baseweb="menu"] li, [role="option"] { color: var(--text) !important; }
[role="option"][aria-selected="true"], [role="option"]:hover { background: var(--primary-soft) !important; }

/* ---------- buttons ---------- */
.stApp button[kind], .stApp [data-testid^="stBaseButton"] {
  border-radius: 999px !important; font-weight: 600 !important; font-size: .94rem;
  min-height: 46px; padding: .5rem 1.5rem; box-shadow: none;
  transition: background .15s ease, border-color .15s ease, color .15s ease, transform .15s ease, box-shadow .15s ease;
}
.stApp button[kind="primary"], .stApp [data-testid="stBaseButton-primary"] {
  background: var(--primary) !important; color: var(--on-primary) !important; border: 1px solid var(--primary) !important;
  box-shadow: 0 6px 16px color-mix(in srgb, var(--primary) 28%, transparent);
}
.stApp button[kind="primary"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover {
  background: var(--primary-hover) !important; border-color: var(--primary-hover) !important; transform: translateY(-1px);
}
.stApp button[kind="primary"] p, .stApp [data-testid="stBaseButton-primary"] p { color: var(--on-primary) !important; }
.stApp button[kind="secondary"], .stApp [data-testid="stBaseButton-secondary"] {
  background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
}
.stApp button[kind="secondary"]:hover, .stApp [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--primary) !important; color: var(--primary) !important; background: var(--primary-soft) !important; transform: translateY(-2px);
}
.stApp button[kind="secondary"] p, .stApp [data-testid="stBaseButton-secondary"] p { color: inherit !important; white-space: nowrap; }
.stApp button:active { transform: none !important; }

/* form submit button: themed (works in light and dark) */
.stFormSubmitButton button { background: var(--primary) !important; color: var(--on-primary) !important; border: 1px solid var(--primary) !important; border-radius: 999px !important; }
.stFormSubmitButton button:hover { background: var(--primary-hover) !important; }
.stFormSubmitButton button p { color: var(--on-primary) !important; }

/* ---------- nav / view switch: segmented control ---------- */
[data-testid="stRadio"] [role="radiogroup"] { display: inline-flex; flex-wrap: nowrap; gap: 2px; background: var(--primary-soft); padding: 4px; border-radius: 999px; max-width: 100%; overflow-x: auto; scrollbar-width: none; }
[data-testid="stRadio"] [role="radiogroup"]::-webkit-scrollbar { display: none; }
[data-testid="stRadio"] label { display: inline-flex; align-items: center; border-radius: 999px !important; padding: .4rem .8rem !important; margin: 0 !important; cursor: pointer; background: transparent !important; transition: background .15s ease; flex-shrink: 0; }[data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] label p { color: var(--muted) !important; font-weight: 600; font-size: .86rem; white-space: nowrap; }
[data-testid="stRadio"] label:hover p { color: var(--text) !important; }
[data-testid="stRadio"] label:has(input:checked) { background: var(--surface) !important; box-shadow: var(--shadow); }
[data-testid="stRadio"] label:has(input:checked) p { color: var(--primary) !important; }
/* the hidden radio title was being re-shown by the pill rule above */
[data-testid="stRadio"] [data-testid="stWidgetLabel"] { display: none !important; }

/* ---------- toggle / slider / checkbox ---------- */
[data-testid="stToggle"] label p, [data-testid="stCheckbox"] label p { color: var(--text) !important; font-size: .9rem; font-weight: 500; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--primary) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--primary) !important; box-shadow: 0 0 0 5px color-mix(in srgb, var(--primary) 20%, transparent) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: var(--primary); font-weight: 700; }
[data-testid="stCheckbox"] span[data-checked="true"] { background: var(--primary) !important; border-color: var(--primary) !important; }
[data-testid="stCheckbox"] span[data-checked="true"] svg { stroke: var(--on-primary) !important; }
[data-testid="stToggle"] [data-checked="true"] { background-color: var(--primary) !important; }

/* ---------- tabs ---------- */
[data-testid="stTabs"] button { color: var(--muted) !important; padding: .6rem .9rem; }
[data-testid="stTabs"] button p { font-weight: 600; }
[data-testid="stTabs"] button[aria-selected="true"], [data-testid="stTabs"] button[aria-selected="true"] p { color: var(--primary) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--primary) !important; height: 3px !important; border-radius: 3px; }
[data-baseweb="tab-border"] { background-color: var(--border) !important; }

/* ---------- cards & hero ----------
   A container becomes a card/hero when its first element is an empty marker div.
   The marker's own element is hidden, and only the container that DIRECTLY holds
   it is styled (so nested Streamlit wrappers are never touched). */
[data-testid="stElementContainer"]:has(.hero-marker, .card-marker) { display: none !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }

[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-marker) { --fx: 1.5rem; --fy: 1.4rem; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hero-marker) { --fx: 2.6rem; --fy: 2.4rem; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-marker),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hero-marker) {
  position: relative; isolation: isolate; margin: var(--fy) 0;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-marker)::before,
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hero-marker)::before {
  content: ""; position: absolute; z-index: -1; pointer-events: none;
  inset: calc(var(--fy) * -1) calc(var(--fx) * -1);
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-marker)::before {
  background: var(--glass); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hero-marker)::before {
  border: 1px solid var(--border); border-radius: 40px; box-shadow: var(--shadow-lg);
  background: radial-gradient(540px 420px at 100% 0%, var(--blob2), transparent 70%), linear-gradient(135deg, var(--hero1), var(--hero2));
}
[data-testid="stHorizontalBlock"]:has(.hero-title) { align-items: center; }

[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 18px !important; background: var(--glass) !important; }

/* ---------- metrics / tables / alerts ---------- */
[data-testid="stMetric"] { background: var(--surface2) !important; border-radius: 16px !important; padding: 14px 16px !important; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--display) !important; font-weight: 600 !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--border) !important; border-radius: 16px !important; overflow: hidden; background: var(--surface) !important; }
[data-testid="stAlert"] { border-radius: 16px !important; }
[data-testid="stToast"] { border-radius: 14px; }

/* ---------- page header / section heads ---------- */
.page-head { margin: .2rem 0 1.5rem; }
.page-title { font-family: var(--display); font-weight: 600; font-size: 2.5rem; letter-spacing: -0.03em; color: var(--text); margin: 0 0 .4rem; line-height: 1.1; }
.page-sub { color: var(--muted); font-size: 1.04rem; max-width: 44em; margin: 0; line-height: 1.6; }
.section-head { margin: 3rem 0 1.1rem; }
.section-title { font-family: var(--display); font-weight: 600; font-size: 1.75rem; letter-spacing: -0.025em; color: var(--text); margin: 0 0 .25rem; }
.section-sub { color: var(--muted); font-size: .98rem; margin: 0; }

/* ---------- hero content ---------- */
.hero-pill { display: inline-flex; align-items: center; gap: .55rem; background: var(--surface); border: 1px solid var(--border); color: var(--primary); font-weight: 600; font-size: .82rem; padding: .4rem 1rem; border-radius: 999px; }
.hero-pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
.hero-title { font-family: var(--display) !important; font-weight: 500 !important; font-size: 3.35rem !important; line-height: 1.05 !important; letter-spacing: -0.035em !important; color: var(--text) !important; margin: 1.2rem 0 1.1rem !important; padding: 0 !important; }
.hero-tag { color: var(--muted) !important; font-size: 1.08rem !important; line-height: 1.7 !important; margin: 0 0 .6rem !important; max-width: 29em; }

.hero-art { position: relative; width: 100%; padding: .4rem 0 1.4rem; }
.hero-art img { width: 100%; display: block; }
.hero-art img.photo { aspect-ratio: 1/1; object-fit: cover; border-radius: 30px; box-shadow: var(--shadow-lg); }
.float-chip {
  position: absolute; display: flex; align-items: center; gap: .5rem;
  background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
  padding: .4rem 1rem .4rem .45rem; font-size: .85rem; font-weight: 600; color: var(--text);
  box-shadow: var(--shadow-lg); animation: bob 6s ease-in-out infinite;
}
.float-chip .ic { width: 28px; height: 28px; border-radius: 50%; background: var(--primary-soft); display: flex; align-items: center; justify-content: center; font-size: 1rem; }
.fc1 { top: 4%; left: 2%; } .fc2 { top: 30%; right: -1%; animation-delay: -2s; } .fc3 { bottom: 30%; left: -2%; animation-delay: -4s; }
.hero-pick {
  position: absolute; bottom: 0; right: -1%; background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; padding: .8rem 1.1rem; box-shadow: var(--shadow-lg); max-width: 215px;
}
.hp-k { font-size: .7rem; color: var(--muted); font-weight: 600; margin-bottom: .1rem; }
.hp-n { font-weight: 700; font-size: .9rem; color: var(--text); line-height: 1.25; }
.hp-v { font-family: var(--display); font-weight: 600; font-size: 1.4rem; color: var(--primary); margin-top: .2rem; line-height: 1.1; }
.hp-v span { font-size: .75rem; color: var(--muted); font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 500; }
@keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }

/* ---------- spotlight cards ---------- */
.spot { background: var(--glass); border: 1px solid var(--border); border-radius: 24px; padding: 1.3rem 1.3rem 1.2rem; height: 100%; box-shadow: var(--shadow); transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
.spot:hover { transform: translateY(-4px); border-color: var(--primary); box-shadow: var(--shadow-lg); }
.spot-top { display: flex; align-items: center; gap: .65rem; margin-bottom: 1rem; }
.spot-ic { width: 40px; height: 40px; border-radius: 13px; background: var(--primary-soft); display: flex; align-items: center; justify-content: center; font-size: 1.35rem; }
.spot-k { font-size: .8rem; font-weight: 600; color: var(--muted); }
.spot-name { font-weight: 700; font-size: .98rem; line-height: 1.3; color: var(--text); min-height: 2.6em; margin-bottom: .75rem; }
.spot-val { font-family: var(--display); font-weight: 600; font-size: 2rem; letter-spacing: -0.02em; color: var(--primary); line-height: 1; }
.spot-val span { font-size: .8rem; color: var(--muted); font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 500; }
.spot-type { margin-top: .8rem; }

/* ---------- stat strip ---------- */
.stat-strip { display: grid; grid-template-columns: repeat(4, 1fr); background: var(--glass); border: 1px solid var(--border); border-radius: 26px; overflow: hidden; backdrop-filter: blur(12px); box-shadow: var(--shadow); }
.stat-strip.strip-5 { grid-template-columns: repeat(5, 1fr); }
.stat-item { padding: 1.5rem 1.6rem; border-left: 1px solid var(--border); }
.stat-item:first-child { border-left: none; }
.stat-value { font-family: var(--display); font-weight: 600; font-size: 2.3rem; letter-spacing: -0.03em; color: var(--primary); line-height: 1.1; }
.stat-label { color: var(--text); font-size: .94rem; font-weight: 600; margin-top: .35rem; }
.stat-sub { color: var(--muted); font-size: .8rem; margin-top: .1rem; }

/* ---------- callouts ---------- */
.callout { border-radius: 16px; padding: 14px 18px; margin: 10px 0; font-size: .93rem; line-height: 1.6; color: var(--text); background: var(--glass); border: 1px solid var(--border); }
.callout-success { background: color-mix(in srgb, var(--success) 9%, var(--surface)); border-color: color-mix(in srgb, var(--success) 30%, var(--border)); }
.callout-info    { background: color-mix(in srgb, var(--info) 8%, var(--surface));    border-color: color-mix(in srgb, var(--info) 28%, var(--border)); }
.callout-warning { background: color-mix(in srgb, var(--warning) 11%, var(--surface)); border-color: color-mix(in srgb, var(--warning) 35%, var(--border)); }

/* ---------- predictor ---------- */
.panel-title { font-family: var(--display); font-weight: 600; font-size: 1.15rem; color: var(--text); margin: .1rem 0 .6rem; letter-spacing: -0.01em; }
.insight { display: flex; align-items: center; gap: 1rem; background: linear-gradient(120deg, var(--primary), color-mix(in srgb, var(--primary) 72%, #000)); color: var(--on-primary); border-radius: 22px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem; box-shadow: var(--shadow-lg); }
.insight-ic { font-size: 2.2rem; line-height: 1; }
.insight-k { font-size: .76rem; opacity: .8; font-weight: 600; }
.insight-t { font-family: var(--display); font-size: 1.35rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1.2; color: var(--on-primary); }
.insight-s { font-size: .86rem; opacity: .85; margin-top: .15rem; }
.insight .insight-t, .insight div { color: var(--on-primary); }

.dcard { position: relative; display: flex; gap: .95rem; align-items: flex-start; background: var(--glass); border: 1px solid var(--border); border-radius: 22px; padding: 1.1rem 1.1rem; height: 100%; backdrop-filter: blur(12px); box-shadow: var(--shadow); transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
.dcard:hover { transform: translateY(-4px); border-color: var(--primary); box-shadow: var(--shadow-lg); }
.dcard-top { border-color: var(--accent); }
.dthumb { flex-shrink: 0; width: 58px; height: 58px; border-radius: 18px; background: var(--primary-soft); display: flex; align-items: center; justify-content: center; font-size: 1.9rem; position: relative; }
.dthumb .rk { position: absolute; top: -7px; left: -7px; min-width: 24px; height: 24px; padding: 0 6px; border-radius: 999px; background: var(--primary); color: var(--on-primary); font-size: .72rem; font-weight: 700; display: flex; align-items: center; justify-content: center; border: 2px solid var(--bg); }
.dcard-top .dthumb .rk { background: var(--accent); color: var(--accent-ink); }
.dbody { flex: 1; min-width: 0; }
.dname { font-weight: 700; color: var(--text); font-size: .97rem; line-height: 1.3; margin-bottom: .45rem; overflow-wrap: anywhere; padding-right: 3.2rem; }
.dtags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: .75rem; }
.tag { font-size: .7rem; padding: 3px 10px; border-radius: 999px; background: var(--surface2); color: var(--muted); font-weight: 600; }
.tag-veg { color: var(--success); background: color-mix(in srgb, var(--success) 13%, var(--surface)); }
.tag-nonveg { color: var(--danger); background: color-mix(in srgb, var(--danger) 12%, var(--surface)); }
.dvalue { font-family: var(--display); font-weight: 600; font-size: 1.75rem; letter-spacing: -0.02em; color: var(--text); line-height: 1.1; }
.dunit { font-size: .82rem; font-weight: 500; color: var(--muted); font-family: 'Plus Jakarta Sans', sans-serif; }
.dbar { height: 7px; border-radius: 999px; background: var(--grid); margin: .7rem 0 .45rem; overflow: hidden; }
.dbarfill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, color-mix(in srgb, var(--primary) 60%, var(--accent)), var(--primary)); }
.dsub { color: var(--muted); font-size: .76rem; }
.dring { position: absolute; top: 1rem; right: 1rem; width: 46px; height: 46px; border-radius: 50%; background: conic-gradient(var(--primary) calc(var(--p) * 1%), var(--grid) 0); display: flex; align-items: center; justify-content: center; }
.dring::before { content: ""; position: absolute; inset: 5px; border-radius: 50%; background: var(--surface); }
.dring span { position: relative; font-size: .72rem; font-weight: 700; color: var(--primary); }

.empty-box { padding: 2.8rem 1.8rem; text-align: center; border: 1.5px dashed var(--border); border-radius: 24px; background: var(--glass); }
.empty-ic { font-size: 2.6rem; margin-bottom: .6rem; }
.empty-title { font-family: var(--display); font-weight: 600; font-size: 1.4rem; color: var(--text); margin-bottom: .4rem; }
.empty-note { color: var(--muted); font-size: .92rem; }

/* hero button row: one flex row, ONE gap value => every gap is identical */
.st-key-cta_row { display: flex !important; flex-direction: row !important; flex-wrap: wrap; align-items: center; gap: .6rem !important; }
.st-key-cta_row > [data-testid="stElementContainer"] { width: auto !important; flex: 0 0 auto !important; min-width: 0 !important; }
.st-key-cta_row [data-testid="stButton"] { width: auto !important; }
.stApp button p { white-space: nowrap; }
[data-testid="stIconMaterial"] { font-size: 1.15rem !important; }

/* ---------- line icons (drawn, theme-coloured) ---------- */
.ico { display: inline-block; width: 1.2em; height: 1.2em; background-color: currentColor; flex-shrink: 0; vertical-align: -0.2em;
  -webkit-mask-size: contain; mask-size: contain; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
  -webkit-mask-position: center; mask-position: center; }
.float-chip .ic { color: var(--primary); } .float-chip .ico { width: 16px; height: 16px; }
.spot-ic { color: var(--primary); } .spot-ic .ico { width: 22px; height: 22px; }
.dthumb { color: var(--primary); } .dthumb .ico { width: 28px; height: 28px; }
.dcard-top .dthumb { color: var(--accent-ink); }
.insight-ic .ico { width: 34px; height: 34px; }
.empty-ic { color: var(--primary); } .empty-ic .ico { width: 44px; height: 44px; }

/* dropdown menus: fully themed (Streamlit paints these light by default) */
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"], [data-baseweb="menu"] ul { background: var(--surface) !important; color: var(--text) !important; }
[data-baseweb="popover"] > div { border-radius: 14px !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"], [data-baseweb="menu"] li { background: transparent !important; color: var(--text) !important; }
[data-baseweb="popover"] li *, [data-baseweb="popover"] [role="option"] * { color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] li[aria-selected="true"], [data-baseweb="popover"] [role="option"][aria-selected="true"] { background: var(--primary-soft) !important; }

/* slider: no coloured blocks behind the min/max labels, same font as the app */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid^="stSliderTickBar"], [data-testid="stSlider"] [data-testid^="stSliderTickBar"] {
  background: transparent !important; background-color: transparent !important; color: var(--muted) !important;
}
[data-testid="stSlider"] [data-testid^="stSliderTickBar"], [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: .8rem;
}

/* file uploader + text area */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface2) !important; border: 1.5px dashed color-mix(in srgb, var(--primary) 45%, transparent) !important;
  border-radius: 22px !important; padding: 2rem 1.4rem !important; transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--primary) !important; background: var(--primary-soft) !important; }
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] div { color: var(--muted) !important; }
[data-testid="stFileUploaderFile"] { background: var(--surface2); border-radius: 14px; padding: .4rem .7rem; }
[data-testid="stFileUploaderFileName"] { color: var(--text) !important; }
[data-testid="stTextArea"] [data-baseweb="textarea"], [data-testid="stTextAreaRootElement"] { background: var(--surface) !important; border-radius: 14px !important; border-color: var(--border) !important; }
[data-testid="stTextArea"] textarea { background: var(--surface) !important; color: var(--text) !important; font-family: ui-monospace, Consolas, monospace !important; font-size: .84rem; }

/* dish grid (equal-height rows) */
.dgrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.1rem; padding-bottom: 2rem; }
[data-testid="stExpander"] { margin-bottom: 1.4rem; }
[data-testid="stExpander"] + [data-testid="stElementContainer"], [data-testid="stCaptionContainer"] { margin-top: .3rem; }
@media (max-width: 760px) { .dgrid { grid-template-columns: 1fr; } }

/* report table */
.rtable { background: var(--glass); border: 1px solid var(--border); border-radius: 22px; overflow: hidden; box-shadow: var(--shadow); }
.rrow { display: grid; grid-template-columns: 1.7fr 1fr 1.1fr .8fr 1.2fr; gap: .8rem; padding: .85rem 1.2rem; border-top: 1px solid var(--border); align-items: center; font-size: .9rem; color: var(--text); }
.rrow:first-child { border-top: none; }
.rhead { background: var(--surface2); font-size: .76rem; font-weight: 700; color: var(--muted); }
.rname { font-weight: 700; }
.rmuted { color: var(--muted); font-size: .84rem; }
.chip { font-weight: 700; font-size: .74rem; padding: 3px 11px; border-radius: 999px; display: inline-block; }
.chip-low { color: var(--danger); background: color-mix(in srgb, var(--danger) 15%, var(--surface)); }
.chip-high { color: var(--warning); background: color-mix(in srgb, var(--warning) 16%, var(--surface)); }
.chip-ok { color: var(--success); background: color-mix(in srgb, var(--success) 14%, var(--surface)); }
.chip-na { color: var(--muted); background: var(--surface2); }
@media (max-width: 700px) { .rrow { grid-template-columns: 1.4fr 1fr 1fr; } .rrow span:nth-child(3), .rrow span:nth-child(5) { display: none; } }

/* nutrient pills (multi-select on Home) */
[data-testid="stPills"] { gap: .55rem; }
[data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"] {
  border-radius: 999px !important; min-height: 44px; padding: .45rem 1.15rem !important; font-weight: 600 !important;
  transition: background .15s ease, border-color .15s ease, color .15s ease, transform .15s ease;
}
[data-testid="stBaseButton-pills"] { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
[data-testid="stBaseButton-pills"]:hover { border-color: var(--primary) !important; color: var(--primary) !important; background: var(--primary-soft) !important; transform: translateY(-1px); }
[data-testid="stBaseButton-pillsActive"] { background: var(--primary) !important; color: var(--on-primary) !important; border: 1px solid var(--primary) !important; box-shadow: 0 6px 16px color-mix(in srgb, var(--primary) 26%, transparent); }
[data-testid="stBaseButton-pills"] p { color: inherit !important; }
[data-testid="stBaseButton-pillsActive"] p { color: var(--on-primary) !important; }

/* combined-match dish cards */
.dmulti { display: grid; grid-template-columns: repeat(auto-fit, minmax(84px, 1fr)); gap: .6rem .9rem; }
.dm-v { font-family: var(--display); font-weight: 600; font-size: 1.3rem; color: var(--text); line-height: 1.1; letter-spacing: -0.02em; }
.dm-v i { font-style: normal; font-size: .72rem; color: var(--muted); font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 500; }
.dm-k { font-size: .72rem; color: var(--muted); font-weight: 600; margin-top: .1rem; }

/* recipes */
.recipe { background: var(--glass); border: 1px solid var(--border); border-radius: 26px; padding: 1.6rem; box-shadow: var(--shadow); margin-top: 1rem; max-width: 100%; overflow: hidden; }
.recipe-head { display: flex; gap: 1.2rem; align-items: center; margin-bottom: 1.2rem; }
.recipe-head > div { min-width: 0; }
.recipe-img { width: 120px; height: 120px; border-radius: 20px; object-fit: cover; flex-shrink: 0; }
.recipe-title { font-family: var(--display); font-weight: 600; font-size: 1.6rem; letter-spacing: -0.02em; color: var(--text); line-height: 1.15; margin-bottom: .55rem; overflow-wrap: anywhere; }
.recipe-meta { display: flex; gap: .45rem; flex-wrap: wrap; }
.recipe-cols { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.6fr); gap: 1.8rem; }
.recipe-cols h4 { font-family: var(--display); font-size: 1.05rem; margin: 0 0 .6rem; color: var(--text); }
.recipe-cols ul, .recipe-cols ol { margin: 0; padding-left: 1.2rem; color: var(--text); font-size: .92rem; line-height: 1.6; }
.recipe-cols li { margin-bottom: .35rem; overflow-wrap: anywhere; }
.recipe-src { margin-top: 1.1rem; font-size: .84rem; color: var(--muted); }
.recipe-src a { color: var(--primary); font-weight: 600; }
@media (max-width: 760px) { .recipe-cols { grid-template-columns: 1fr; } .recipe-head { flex-direction: column; align-items: flex-start; } }

/* placeholder + checkbox legibility (esp. dark mode) */
[data-baseweb="select"] input::placeholder { color: var(--muted) !important; opacity: 1 !important; }
[data-baseweb="checkbox"] > span:first-of-type {
  background: var(--surface2) !important; border: 1px solid color-mix(in srgb, var(--text) 32%, transparent) !important; border-radius: 7px !important;
}
[data-baseweb="checkbox"]:has(input:checked) > span:first-of-type { background: var(--primary) !important; border-color: var(--primary) !important; }
[data-baseweb="checkbox"]:has(input:checked) > span:first-of-type svg, [data-baseweb="checkbox"]:has(input:checked) > span:first-of-type svg path { color: var(--on-primary) !important; fill: var(--on-primary) !important; }
[data-testid="stDownloadButton"] { display: flex; justify-content: flex-end; }
[data-testid="stDownloadButton"] button { min-height: 42px; font-size: .88rem; padding: .4rem 1.2rem; }

.foot { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 3.5rem; }

/* select inputs: no focus box around the caret, thin themed cursor instead.
   The focus ring already lives on the whole control via :focus-within */
[data-baseweb="select"] input,
[data-baseweb="select"] input:focus,
[data-baseweb="select"] input:focus-visible {
  outline: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  caret-color: var(--primary) !important;
}

/* ---- multiselect box + tags: always themed ---- */
.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="select"] > div > div { background-color: var(--surface) !important; }
.stApp [data-baseweb="select"] div { color: var(--text) !important; }

.stApp [data-baseweb="tag"],
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background-color: var(--primary-soft) !important;
  border: 1px solid color-mix(in srgb, var(--primary) 30%, transparent) !important;
  border-radius: 999px !important;
}
.stApp [data-baseweb="tag"] span,
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
.stApp [data-baseweb="tag"] svg {
  color: var(--primary) !important; fill: var(--primary) !important; font-weight: 600;
}

/* ---- dark-mode coverage for inputs, forms, alerts, expanders ---- */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"] {
  background-color: var(--surface) !important; border-color: var(--border) !important; border-radius: 14px !important;
}
.stApp [data-baseweb="input"] input, .stApp [data-testid="stNumberInput"] input {
  background: transparent !important; color: var(--text) !important;
}
.stApp [data-testid="stNumberInput"] button { background: var(--surface2) !important; color: var(--text) !important; }
[data-testid="stForm"] { border: 1px solid var(--border) !important; border-radius: 20px !important; background: var(--glass) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color: var(--text) !important; }
[data-testid="stAlert"] { background-color: color-mix(in srgb, var(--info) 10%, var(--surface)) !important; border: 1px solid var(--border) !important; }
[data-testid="stAlert"] p { color: var(--text) !important; }

:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; border-radius: 6px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
@media (max-width: 900px) {
  .stat-strip, .stat-strip.strip-5 { grid-template-columns: repeat(2, 1fr); }
  .stat-item:nth-child(3) { border-left: none; }
  .stat-strip.strip-5 .stat-item:nth-child(odd) { border-left: none; }
  .stat-item:nth-child(n+3) { border-top: 1px solid var(--border); }
  .hero-title { font-size: 2.4rem !important; }
  .float-chip { display: none; }
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hero-marker) { --fx: 1rem; --fy: 1.4rem; }
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-marker) { --fx: .8rem; --fy: 1rem; }
}
@media (max-width: 768px) {
  .block-container, [data-testid="stMain"] .block-container { padding: 1rem 1rem 3rem !important; }
  .page-title { font-size: 1.9rem; }
}
/* ===== top bar fix ===== */
/* columns size to their content instead of fixed ratios; nav takes the leftover space */
[data-testid="stHorizontalBlock"]:has(.brand) { flex-wrap: nowrap !important; gap: 1rem !important; }
[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"] {
  width: auto !important; flex: 0 0 auto !important; min-width: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"]:has([data-testid="stRadio"]) {
  flex: 1 1 0 !important; overflow: hidden;
}
.brand { white-space: nowrap; }

/* nav pill: start-aligned, scrolls instead of overflowing */
[data-testid="stRadio"] [role="radiogroup"] { max-width: 100%; overflow-x: auto; }
[data-testid="stRadio"] label { padding: .4rem .85rem !important; }

/* hide the radio circle robustly (old + new Streamlit markup) */
[data-testid="stRadio"] label > div:first-child:not(:has(p)),
[data-testid="stRadio"] label > span:first-child,
[data-testid="stRadio"] label [data-baseweb="radio"] > div:first-child:not(:has(p)),
[data-testid="stRadio"] label input[type="radio"] + div:not(:has(p)) {
  display: none !important;
}
[data-testid="stRadio"] label input { position: absolute; opacity: 0; width: 0; height: 0; }

/* dark-mode toggle stays on one line */
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stToggle"] { width: auto; min-width: 0; white-space: nowrap; }

/* ===== columns start at the top (topbar + hero stay centred) ===== */
[data-testid="stHorizontalBlock"] { align-items: flex-start; }
[data-testid="stHorizontalBlock"]:has(.brand),
[data-testid="stHorizontalBlock"]:has(.hero-title) { align-items: center; }

/* ===== lighter multiselect tags (e.g. "Iron (mg)") ===== */
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background: color-mix(in srgb, var(--primary) 16%, var(--surface)) !important;
  border: 1px solid color-mix(in srgb, var(--primary) 28%, transparent) !important;
  border-radius: 999px !important;
}
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] *,
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] span {
  background: transparent !important;
  color: #2b2105 !important;
  fill: #2b2105 !important;

  font-weight: 600;
}
</style>
"""

def init_theme() -> str:
    if "theme" not in st.session_state:
        base = "light"
        try:
            base = str(st.get_option("theme.base") or "light").lower()
        except Exception:
            pass
        st.session_state["theme"] = "Dark" if base == "dark" else "Light"
    return st.session_state["theme"]


def on_theme_toggle() -> None:
    st.session_state["theme"] = (
        "Dark" if st.session_state["theme_toggle"] else "Light"
    )


def inject_theme(theme: str) -> None:
    css = _CSS
    for key, value in PALETTES[theme].items():
        css = css.replace("__" + key.upper() + "__", value)
    scheme = "dark" if theme == "Dark" else "light"
    st.markdown(css + _icon_css() + f"<style>:root{{color-scheme:{scheme};}}</style>",
                unsafe_allow_html=True)


PLOT_CONFIG = {"displayModeBar": False}


def apply_plotly_theme(fig, theme: str, height: int | None = None,
                       xlabel: str | None = None, ylabel: str | None = None,
                       title: str | None = None):
    p = PALETTES[theme]
    layout = dict(
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, 'Segoe UI', sans-serif", color=p["text"], size=13),
        title_font=dict(family="Fraunces, Georgia, serif", color=p["text"], size=18),
        colorway=COLORWAY[theme],
        margin=dict(t=56 if title else 48, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(bgcolor=p["surface"], font_color=p["text"], bordercolor=p["primary"]),
        bargap=0.18,
    )
    if height:
        layout["height"] = height
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    if fig.layout.title.text:
        fig.update_layout(title_x=0)
    else:  # an empty title object makes Plotly print the word 'undefined'
        fig.update_layout(title_text="", margin_t=16)
    axis = dict(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["grid"],
                tickfont=dict(color=p["muted"]), title_font=dict(color=p["muted"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if xlabel is not None:
        fig.update_xaxes(title_text=xlabel)
    if ylabel is not None:
        fig.update_yaxes(title_text=ylabel)
    return fig


# ==========================================================================
# UI COMPONENTS (HTML helpers)
# ==========================================================================
ICONS = {
    "flame": '<path d="M12 3c1 3.5 5 5.5 5 10.5a5 5 0 0 1-10 0c0-2.2 1-3.6 2.4-4.8.2 1.8 1 2.8 2.1 3.3C11.3 9 11 6 12 3z"/>',
    "grain": '<path d="M12 21V4"/><path d="M12 9c-3 0-4.5-1.8-4.5-4.5C10.5 4.5 12 6 12 9z"/><path d="M12 9c3 0 4.5-1.8 4.5-4.5C13.5 4.5 12 6 12 9z"/><path d="M12 15c-3 0-4.5-1.8-4.5-4.5C10.5 10.5 12 12 12 15z"/><path d="M12 15c3 0 4.5-1.8 4.5-4.5C13.5 10.5 12 12 12 15z"/>',
    "dumbbell": '<path d="M6.5 7v10M17.5 7v10M3.5 9.5v5M20.5 9.5v5M6.5 12h11"/>',
    "droplet": '<path d="M12 3c3.2 4 6 7 6 11a6 6 0 0 1-12 0c0-4 2.8-7 6-11z"/>',
    "candy": '<rect x="6" y="8" width="12" height="8" rx="3"/><path d="M6 12l-3.5-3v6L6 12zM18 12l3.5-3v6L18 12z"/>',
    "leaf": '<path d="M5 19C5 10 10 5 20 5c0 10-5 15-13 15"/><path d="M5 19l9-9"/>',
    "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M4 7.5L12 12l8-4.5M12 12v9"/>',
    "glass": '<path d="M7 3h10l-1.2 17.5a1 1 0 0 1-1 .9H9.2a1 1 0 0 1-1-.9L7 3z"/><path d="M7.4 9h9.2"/>',
    "shield": '<path d="M12 3l7.5 2.8v6c0 4.6-3.2 7.8-7.5 9.2-4.3-1.4-7.5-4.6-7.5-9.2v-6L12 3z"/><path d="M9 12l2.2 2.2L15.5 10"/>',
    "citrus": '<circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6L5.6 18.4"/>',
    "sprout": '<path d="M12 21v-9"/><path d="M12 12c0-4-3-6.5-7.5-6.5 0 4.2 3 6.5 7.5 6.5z"/><path d="M12 14.5c0-3.2 2.3-5.5 6.5-5.5 0 3.5-2.4 5.5-6.5 5.5z"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
    "sliders": '<path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/>',
    "bars": '<path d="M5 20v-7M12 20V5M19 20v-11"/>',
    "bowl": '<path d="M3 11h18a9 9 0 0 1-18 0z"/><path d="M8.5 7c0-1.6 1.5-1.6 1.5-3.2M13.5 7c0-1.6 1.5-1.6 1.5-3.2"/>',
    "cup": '<path d="M6 8h11l-1.2 11a2 2 0 0 1-2 1.8h-5.6a2 2 0 0 1-2-1.8L6 8z"/><path d="M17 10h1.5a2.5 2.5 0 0 1 0 5H16.6M9 3.5v2M13 3.5v2"/>',
    "pot": '<path d="M4 10h16v7a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3v-7z"/><path d="M2 10h20M9 6.5h6"/>',
    "logo": '<path d="M4 13h16a8 8 0 0 1-16 0z"/><path d="M12 10.5c0-3.4 2.4-5.8 6.4-6 0 3.7-2.5 6-6.4 6z"/><path d="M12 10.5c0-2.2-1.7-3.8-4.6-4 0 2.5 1.8 4 4.6 4z"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.8-4.8"/>',
}

# nutrient -> drawn icon (HTML cards) and -> Material icon (Streamlit buttons)
NUT_ICON = {"Calories (kcal)": "flame", "Carbohydrates (g)": "grain", "Protein (g)": "dumbbell",
            "Fats (g)": "droplet", "Free Sugar (g)": "candy", "Fibre (g)": "leaf",
            "Sodium (mg)": "cube", "Calcium (mg)": "glass", "Iron (mg)": "shield",
            "Vitamin C (mg)": "citrus", "Folate (ug)": "sprout"}
MAT_ICON = {"Calories (kcal)": ":material/local_fire_department:", "Carbohydrates (g)": ":material/grain:",
            "Protein (g)": ":material/fitness_center:", "Fats (g)": ":material/opacity:",
            "Free Sugar (g)": ":material/cookie:", "Fibre (g)": ":material/grass:",
            "Sodium (mg)": ":material/bubble_chart:", "Calcium (mg)": ":material/water_drop:",
            "Iron (mg)": ":material/bolt:", "Vitamin C (mg)": ":material/wb_sunny:",
            "Folate (ug)": ":material/eco:"}


def _icon_css() -> str:
    rules = []
    for name, body in ICONS.items():
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" '
               'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + body + '</svg>')
        uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
        rules.append(f'.ico-{name}{{-webkit-mask-image:url("{uri}");mask-image:url("{uri}");}}')
    return "<style>" + "".join(rules) + "</style>"


def ico(name: str) -> str:
    return f'<span class="ico ico-{name if name in ICONS else "leaf"}"></span>'


def _h(markup: str) -> str:
    """Collapse to one line so Streamlit's markdown never treats HTML as code."""
    return "".join(line.strip() for line in markup.splitlines())


HERO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 440">
<circle cx="240" cy="222" r="198" fill="#cfe0c4" opacity=".55"/>
<circle cx="240" cy="222" r="150" fill="#ffffff" opacity=".42"/>
<circle cx="408" cy="78" r="26" fill="#f1c75b"/>
<circle cx="74" cy="92" r="9" fill="#e9b44c" opacity=".85"/>
<circle cx="440" cy="300" r="7" fill="#6f9a74" opacity=".85"/>
<circle cx="48" cy="286" r="11" fill="#6f9a74" opacity=".6"/>
<circle cx="120" cy="40" r="5" fill="#6f9a74" opacity=".7"/>
<g transform="translate(116 236) scale(-1 1) rotate(-14)"><path d="M0 0 C-6 -70 54 -114 110 -88 C108 -30 64 14 0 0Z" fill="#7aa17a"/><path d="M0 0 C36 -24 66 -52 100 -84" stroke="#4a7152" stroke-width="3.5" fill="none" stroke-linecap="round"/></g>
<g transform="translate(364 236) rotate(-14)"><path d="M0 0 C-6 -70 54 -114 110 -88 C108 -30 64 14 0 0Z" fill="#5d8a63"/><path d="M0 0 C36 -24 66 -52 100 -84" stroke="#3f6a47" stroke-width="3.5" fill="none" stroke-linecap="round"/></g>
<ellipse cx="240" cy="398" rx="142" ry="13" fill="#2b3a2e" opacity=".14"/>
<path d="M90 240 C90 332 158 390 240 390 C322 390 390 332 390 240Z" fill="#fbf7ea" stroke="#4a7152" stroke-width="6" stroke-linejoin="round"/>
<path d="M120 286 C134 330 172 360 216 370" stroke="#ffffff" stroke-width="8" fill="none" stroke-linecap="round" opacity=".75"/>
<ellipse cx="240" cy="240" rx="150" ry="22" fill="#efe6cf" stroke="#4a7152" stroke-width="6"/>
<path d="M112 238 C112 170 168 128 240 128 C312 128 368 170 368 238 C340 264 140 264 112 238Z" fill="#f0b94d"/>
<circle cx="176" cy="178" r="22" fill="#d9573f"/><circle cx="170" cy="171" r="6" fill="#ec7f69" opacity=".75"/>
<circle cx="304" cy="172" r="19" fill="#d9573f"/><circle cx="298" cy="166" r="5" fill="#ec7f69" opacity=".75"/>
<path d="M208 166 C198 128 234 106 260 130 C258 154 236 170 208 166Z" fill="#69a06c"/>
<path d="M222 164 C234 152 246 140 254 130" stroke="#3f6a47" stroke-width="2.5" fill="none" stroke-linecap="round"/>
<circle cx="140" cy="216" r="9" fill="#f7e2a6"/><circle cx="158" cy="228" r="8" fill="#f7e2a6"/><circle cx="322" cy="220" r="9" fill="#f7e2a6"/><circle cx="340" cy="208" r="8" fill="#f7e2a6"/><circle cx="240" cy="216" r="9" fill="#f7e2a6"/><circle cx="214" cy="230" r="8" fill="#f7e2a6"/><circle cx="270" cy="232" r="8" fill="#f7e2a6"/>
<path d="M176 208 q8 -12 16 0 q-8 12 -16 0Z" fill="#7aa17a"/><path d="M286 206 q8 -12 16 0 q-8 12 -16 0Z" fill="#7aa17a"/>
<path d="M190 106 C180 88 202 80 192 60" stroke="#ffffff" stroke-width="5" fill="none" stroke-linecap="round" opacity=".9"/>
<path d="M240 100 C230 82 252 74 242 54" stroke="#ffffff" stroke-width="5" fill="none" stroke-linecap="round" opacity=".9"/>
<path d="M290 106 C280 88 302 80 292 60" stroke="#ffffff" stroke-width="5" fill="none" stroke-linecap="round" opacity=".9"/>
</svg>"""


def hero_art_html(pick) -> str:
    photo = ASSETS / "hero_photo.jpg"
    if photo.exists():
        src = "data:image/jpeg;base64," + base64.b64encode(photo.read_bytes()).decode("ascii")
        img = f'<img class="photo" src="{src}" alt="">'
    else:
        src = "data:image/svg+xml;base64," + base64.b64encode(HERO_SVG.encode("utf-8")).decode("ascii")
        img = f'<img src="{src}" alt="A bowl of colourful food">'
    pick_html = ""
    if pick:
        n, v, u = pick
        pick_html = (f'<div class="hero-pick"><div class="hp-k">Top iron-rich dish</div>'
                     f'<div class="hp-n">{escape(n)}</div><div class="hp-v">{v} <span>{u}</span></div></div>')
    return _h(f"""
    <div class="hero-art">{img}
      <div class="float-chip fc1"><span class="ic">{ico("shield")}</span>Iron</div>
      <div class="float-chip fc2"><span class="ic">{ico("leaf")}</span>Fibre</div>
      <div class="float-chip fc3"><span class="ic">{ico("dumbbell")}</span>Protein</div>
      {pick_html}
    </div>""")


def brand_markup() -> str:
    b64 = _logo_b64()
    mark = (f'<img class="brand-logo" src="data:image/png;base64,{b64}" alt="NutriMatch logo">' if b64
            else f'<span class="brand-mark">{ico("logo")}</span>')
    return f'<div class="brand">{mark}<span class="brand-name">NutriMatch</span></div>'


def page_header(title: str, subtitle: str = "") -> str:
    sub = f'<p class="page-sub">{subtitle}</p>' if subtitle else ""
    return f'<div class="page-head"><h1 class="page-title">{title}</h1>{sub}</div>'


def section_head(title: str, sub: str = "") -> str:
    s = f'<p class="section-sub">{sub}</p>' if sub else ""
    return f'<div class="section-head"><h2 class="section-title">{title}</h2>{s}</div>'


def stat_strip(items, cols: int = 4) -> str:
    cells = "".join(
        f'<div class="stat-item"><div class="stat-value">{v}</div><div class="stat-label">{l}</div>'
        + (f'<div class="stat-sub">{s}</div>' if s else "") + "</div>"
        for l, v, s in items)
    cls = "stat-strip" + (f" strip-{cols}" if cols != 4 else "")
    return f'<div class="{cls}">{cells}</div>'


def callout(kind: str, text: str) -> str:
    return f'<div class="callout callout-{kind}">{text}</div>'


HERO_MARK = '<div class="hero-marker"></div>'
CARD = '<div class="card-marker"></div>'


def spotlight_card(icon: str, kicker: str, name: str, value: str, unit: str, dish_type: str) -> str:
    return _h(f"""<div class="spot"><div class="spot-top"><div class="spot-ic">{ico(icon)}</div>
    <div class="spot-k">{kicker}</div></div><div class="spot-name">{escape(name)}</div>
    <div class="spot-val">{value} <span>{unit}</span></div>
    <div class="spot-type"><span class="tag">{escape(dish_type)}</span></div></div>""")


def dish_icon(dish_type: str) -> str:
    t = str(dish_type).lower()
    for keys, name in (
        (("dessert", "sweet", "mithai"), "candy"), (("beverage", "drink", "juice"), "cup"),
        (("soup",), "bowl"), (("salad",), "leaf"),
        (("bread", "roti", "paratha", "rice", "biryani", "pulao", "breakfast"), "grain"),
        (("snack", "starter", "chaat"), "flame"), (("chutney", "condiment", "sauce", "pickle"), "droplet"),
        (("curry", "gravy", "dal", "main", "side"), "pot"),
    ):
        if any(k in t for k in keys):
            return name
    return "bowl"


def daily_need_txt(nutrient: str, value) -> str:
    """'18% of daily need' / '25% of daily limit' for one per-serving value."""
    ref = DAILY_REFERENCES.get(nutrient)
    if not ref or value is None or pd.isna(value):
        return ""
    amount, kind = ref
    word = {"target": "need", "limit": "limit"}.get(kind, kind)
    try:
        return f"{float(value) / amount * 100:.0f}% of daily {word}"
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


def dish_card(rank, name, dish_type, value, unit, pct_of_max, percentile, veg,
              extras=None, daily_txt: str = "") -> str:
    veg_cls = "tag-veg" if veg == 0 else "tag-nonveg"
    veg_lbl = "Vegetarian" if veg == 0 else "Non-veg"
    width = min(100.0, max(0.0, pct_of_max))
    top = " dcard-top" if rank == 1 else ""
    safe = escape(str(name))
    if extras:   # several nutrients: show each value, bar + ring = combined match
        items = ""
        for extra in extras:
            k, v, u = extra[0], extra[1], extra[2]
            pct = escape(extra[3]) if len(extra) > 3 and extra[3] else ""
            lbl = escape(k) + (f" · {pct}" if pct else "")
            items += f'<div><div class="dm-v">{v} <i>{u}</i></div><div class="dm-k">{lbl}</div></div>'
        value_html = f'<div class="dmulti">{items}</div>'
        sub_text = "Combined match across your nutrients"
    else:
        value_html = f'<div class="dvalue">{value} <span class="dunit">{unit}</span></div>'
        sub_text = f"Top {max(1, 100 - percentile)}% of all dishes"
    if daily_txt:
        sub_text = f"{daily_txt} · {sub_text}"
    return _h(f"""
    <div class="dcard{top}">
      <div class="dthumb">{ico(dish_icon(dish_type))}<span class="rk">{rank}</span></div>
      <div class="dbody">
        <div class="dname" title="{safe}">{safe}</div>
        <div class="dtags"><span class="tag">{escape(str(dish_type))}</span><span class="tag {veg_cls}">{veg_lbl}</span></div>
        {value_html}
        <div class="dbar"><div class="dbarfill" style="width:{width:.1f}%"></div></div>
        <div class="dsub">{sub_text}</div>
      </div>
      <div class="dring" style="background:conic-gradient(var(--primary) {percentile}%, var(--grid) 0)"><span>{percentile}</span></div>
    </div>""")


def insight_banner(icon: str, kicker: str, title: str, sub: str) -> str:
    return _h(f"""<div class="insight"><div class="insight-ic">{ico(icon)}</div><div>
    <div class="insight-k">{kicker}</div><div class="insight-t">{title}</div>
    <div class="insight-s">{sub}</div></div></div>""")


def empty_state(icon: str, title: str, note: str) -> str:
    return _h(f"""<div class="empty-box"><div class="empty-ic">{ico(icon)}</div>
    <div class="empty-title">{title}</div><div class="empty-note">{note}</div></div>""")


# ==========================================================================
# DATA + LOGIC (unchanged)
# ==========================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
    res = pd.read_csv(RESULTS) if RESULTS.exists() else None
    return df, res


def nutrient_label(nutrient: str) -> str:
    return f"{LABELS.get(nutrient, nutrient).capitalize()} ({UNITS.get(nutrient, '')})".strip()


def recommend(df, nutrient, prefer_types, veg_only, avoid_sweet, avoid_fried,
              meals_only, top_n, w_nut=0.7, w_pref=0.3):
    """Rank dishes for a nutrient target (unchanged logic; output column names fixed)."""
    out = df.dropna(subset=[nutrient]).copy()
    if meals_only:
        out = out[(out["is_condiment"] == 0) & (out["dish_type"] != "Beverage")]
    if veg_only:
        out = out[out["is_nonveg"] == 0]
    if avoid_sweet:
        out = out[out["is_sweet"] == 0]
    if avoid_fried:
        out = out[out["is_fried"] == 0]
    if prefer_types:
        out = out[out["dish_type"].isin(prefer_types)]
    if out.empty:
        return out

    out["nutrient_score"] = out[nutrient].rank(pct=True)
    pref = pd.Series(0.0, index=out.index)
    if prefer_types:
        pref += out["dish_type"].isin(prefer_types).astype(float)
    if veg_only:
        pref += out["is_nonveg"].eq(0).astype(float)
    if avoid_sweet:
        pref += out["is_sweet"].eq(0).astype(float)
    if avoid_fried:
        pref += out["is_fried"].eq(0).astype(float)
    out["pref_score"] = pref / (pref.max() if pref.max() > 0 else 1)
    out["final_score"] = w_nut * out["nutrient_score"] + w_pref * out["pref_score"]
    out = out.sort_values("final_score", ascending=False).head(top_n)

    result = out[["Dish Name", nutrient, "dish_type", "nutrient_score"]].reset_index(drop=True)
    result["match"] = (result["nutrient_score"] * 100).round(0).astype(int)
    return result


def recommend_multi(df, nutrients, prefer_types, veg_only, avoid_sweet, avoid_fried,
                    meals_only, top_n, w_nut=0.7, w_pref=0.3):
    """Dishes rich in ALL chosen nutrients: each nutrient is ranked as a percentile and the
    percentiles are averaged. Filters and the 70/30 preference blend match recommend()."""
    out = df.dropna(subset=list(nutrients)).copy()
    if meals_only:
        out = out[(out["is_condiment"] == 0) & (out["dish_type"] != "Beverage")]
    if veg_only:
        out = out[out["is_nonveg"] == 0]
    if avoid_sweet:
        out = out[out["is_sweet"] == 0]
    if avoid_fried:
        out = out[out["is_fried"] == 0]
    if prefer_types:
        out = out[out["dish_type"].isin(prefer_types)]
    if out.empty:
        return out

    out["nutrient_score"] = pd.concat([out[n].rank(pct=True) for n in nutrients], axis=1).mean(axis=1)
    pref = pd.Series(0.0, index=out.index)
    if prefer_types:
        pref += out["dish_type"].isin(prefer_types).astype(float)
    if veg_only:
        pref += out["is_nonveg"].eq(0).astype(float)
    if avoid_sweet:
        pref += out["is_sweet"].eq(0).astype(float)
    if avoid_fried:
        pref += out["is_fried"].eq(0).astype(float)
    out["pref_score"] = pref / (pref.max() if pref.max() > 0 else 1)
    out["final_score"] = w_nut * out["nutrient_score"] + w_pref * out["pref_score"]
    out = out.sort_values("final_score", ascending=False).head(top_n)

    result = out[["Dish Name", *nutrients, "dish_type", "nutrient_score"]].reset_index(drop=True)
    result["match"] = (result["nutrient_score"] * 100).round(0).astype(int)
    return result


def dish_grid_multi_html(df, rec, nutrients) -> str:
    veg_lookup = df.drop_duplicates("Dish Name").set_index("Dish Name")["is_nonveg"].astype(int).to_dict()
    cards = []
    for i, row in enumerate(rec.to_dict("records"), start=1):
        extras = [(LABELS.get(n, n).capitalize(), f"{row[n]:,.1f}", UNITS.get(n, ""),
                   daily_need_txt(n, row[n])) for n in nutrients]
        pct = int(row["match"])
        cards.append(dish_card(i, row["Dish Name"], row["dish_type"], "", "", pct, pct,
                               veg_lookup.get(row["Dish Name"], 1), extras=extras))
    return '<div class="dgrid">' + "".join(cards) + "</div>"


def dish_grid_html(df, rec, nutrient) -> str:
    """Cards in one CSS grid so every row has equal-height cards."""
    unit = UNITS.get(nutrient, "")
    veg_lookup = df.drop_duplicates("Dish Name").set_index("Dish Name")["is_nonveg"].astype(int).to_dict()
    max_val = float(rec[nutrient].max())
    cards = []
    for i, (name, value, dtype, _score, match) in enumerate(rec.itertuples(index=False), start=1):
        pct = (float(value) / max_val * 100) if max_val else 0.0
        cards.append(dish_card(i, name, dtype, f"{value:,.1f}", unit, pct, int(match),
                               veg_lookup.get(name, 1),
                               daily_txt=daily_need_txt(nutrient, value)))
    return '<div class="dgrid">' + "".join(cards) + "</div>"


def hero_pick(df, nutrient="Iron (mg)"):
    if nutrient not in df.columns:
        return None
    m = df.dropna(subset=[nutrient])
    if "is_condiment" in m.columns:
        m = m[m["is_condiment"] == 0]
    if "dish_type" in m.columns:
        m = m[m["dish_type"] != "Beverage"]
    if m.empty:
        return None
    r = m.loc[m[nutrient].idxmax()]
    return str(r["Dish Name"]), f"{r[nutrient]:,.1f}", UNITS.get(nutrient, "")


def top_dish(df, nutrient):
    """Richest meal for a nutrient: (name, value, unit, dish_type)."""
    if nutrient not in df.columns:
        return None
    m = df.dropna(subset=[nutrient])
    if "is_condiment" in m.columns:
        m = m[m["is_condiment"] == 0]
    if "dish_type" in m.columns:
        m = m[m["dish_type"] != "Beverage"]
    if m.empty:
        return None
    r = m.loc[m[nutrient].idxmax()]
    return str(r["Dish Name"]), f"{r[nutrient]:,.1f}", UNITS.get(nutrient, ""), str(r.get("dish_type", ""))


def hero_texture_css() -> str:
    """Optional faint photo texture inside the hero only (assets/hero_blur_bg.jpg)."""
    img = ASSETS / "hero_blur_bg.jpg"
    if not img.exists():
        return ""
    b64 = base64.b64encode(img.read_bytes()).decode("ascii")
    return ('<style>[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hero-marker)::before{'
            f'background-image:url("data:image/jpeg;base64,{b64}");}}</style>')


# --- navigation callbacks (widget state may only change before widgets render)
def go_predictor() -> None:
    st.session_state["nav"] = "Predictor"


def go_graphs() -> None:
    st.session_state["nav"] = "Graphs and Statistics"


def go_report() -> None:
    st.session_state["nav"] = "Report Reader"


def pick_nutrient(nutrient: str) -> None:
    st.session_state["p_nuts"] = [nutrient]
    st.session_state["nav"] = "Predictor"


def open_in_predictor(nutrients) -> None:
    st.session_state["p_nuts"] = list(nutrients)
    st.session_state["nav"] = "Predictor"


def go_with_selection() -> None:
    chosen = list(st.session_state.get("home_nuts") or [])
    if chosen:
        open_in_predictor(chosen)


# ==========================================================================
# RECIPES - an LLM writes the recipe: Google Gemini by default, OpenAI as an
# alternative. TheMealDB (free, no key) stays as a fallback.
# Set GEMINI_API_KEY (or GOOGLE_API_KEY) / OPENAI_API_KEY as an environment
# variable or in .streamlit/secrets.toml. Optional overrides: LLM_PROVIDER
# (gemini|openai), GEMINI_MODEL, OPENAI_MODEL.
# ==========================================================================
class RecipeError(Exception):
    pass


# Key names looked up for each provider, in order of preference.
LLM_KEY_NAMES = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"),
                 "openai": ("OPENAI_API_KEY",)}
LLM_DEFAULT_MODEL = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}
LLM_MODEL_ENV = {"gemini": "GEMINI_MODEL", "openai": "OPENAI_MODEL"}
LLM_LABEL = {"gemini": "Gemini", "openai": "OpenAI"}

RECIPE_PROMPT = (
    "You are an Indian home-cooking assistant. Write one concise, reliable recipe for the dish "
    "'{dish}' as it is usually cooked in India. Use common ingredients with practical quantities "
    "for 2 servings, and keep every method step to one or two sentences.\n"
    "Answer with JSON only, in exactly this shape: "
    '{{"title": "...", "servings": 2, "time_minutes": 30, '
    '"ingredients": ["1 cup ...", "..."], "steps": ["...", "..."]}}'
)


def _secrets_file_exists() -> bool:
    """Touching st.secrets with no secrets.toml makes Streamlit print a red 'No secrets found' box,
    so only read it when a file really exists."""
    return any(p.is_file() for p in (Path.home() / ".streamlit" / "secrets.toml",
                                     Path.cwd() / ".streamlit" / "secrets.toml"))


def _secret(name: str) -> str:
    if not _secrets_file_exists():
        return ""
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def llm_config() -> tuple:
    """(provider, key, model) for the recipe writer; ('', '', '') when no key is configured.

    Gemini comes first because its free tier is enough for a classroom app; OpenAI is
    used instead when that is the only key present (or when LLM_PROVIDER says so).
    """
    wanted = (os.environ.get("LLM_PROVIDER") or _secret("LLM_PROVIDER")).strip().lower()
    order = [wanted] if wanted in LLM_KEY_NAMES else ["gemini", "openai"]
    for provider in order:
        for name in LLM_KEY_NAMES[provider]:
            key = os.environ.get(name, "").strip() or _secret(name)
            if key:
                model = (os.environ.get(LLM_MODEL_ENV[provider])
                         or _secret(LLM_MODEL_ENV[provider])
                         or LLM_DEFAULT_MODEL[provider]).strip()
                return provider, key, model
    return "", "", ""


def _parse_recipe_json(text: str, provider: str):
    """Turn the model's JSON answer into the recipe dict the page renders (None if unusable)."""
    text = re.sub(r"```(?:json)?", "", str(text or "")).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    ingredients = [_clean(x) for x in data.get("ingredients") or [] if _clean(x)]
    steps = [_clean(x) for x in data.get("steps") or [] if _clean(x)]
    if not ingredients or not steps:
        return None
    meta = []
    for field, fmt in (("time_minutes", "{} min"), ("servings", "serves {}")):
        try:
            value = int(round(float(data.get(field))))
        except (TypeError, ValueError):
            continue
        if value > 0:
            meta.append(fmt.format(value))
    meta.append("AI-written recipe - check it before cooking")
    return {"provider": LLM_LABEL[provider], "title": _clean(data.get("title")) or "Recipe",
            "image": "", "meta": meta, "ingredients": ingredients, "steps": steps, "source": ""}


def _name_variants(dish: str) -> list[str]:
    """'Sesame ladoo (Til ke ladoo)' -> ['sesame ladoo', 'til ke ladoo', 'ladoo', 'sesame']"""
    base = re.sub(r"\([^)]*\)", " ", dish).strip()
    inner = [t.strip() for t in re.findall(r"\(([^)]*)\)", dish)]
    stop = {"and", "the", "with", "aur", "for", "ki", "ke", "ka", "ko", "se", "in", "of"}
    words = [w for w in re.findall(r"[A-Za-z]+", base) if len(w) >= 3 and w.lower() not in stop]
    out, seen = [], set()
    for q in [base, *inner, *words[::-1][:2]]:
        q = " ".join(q.split()).lower()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def _themealdb(query: str):
    import requests
    r = requests.get("https://www.themealdb.com/api/json/v1/1/search.php", params={"s": query},
                     timeout=8, headers={"User-Agent": "NutriMatch/1.0"})
    r.raise_for_status()
    meals = (r.json() or {}).get("meals") or []
    if not meals:
        return None
    meals.sort(key=lambda m: m.get("strArea") != "Indian")      # prefer Indian dishes
    m = meals[0]
    ingredients = []
    for i in range(1, 21):
        ing = _clean(m.get(f"strIngredient{i}"))
        if ing:
            ingredients.append(f'{_clean(m.get(f"strMeasure{i}"))} {ing}'.strip())
    steps = [_clean(x) for x in re.split(r"\r?\n+", m.get("strInstructions") or "")]
    steps = [x for x in steps if x and not re.fullmatch(r"(step\s*)?\d+\.?", x, re.I)]
    return {"provider": "TheMealDB", "title": _clean(m.get("strMeal")), "image": m.get("strMealThumb") or "",
            "meta": [t for t in (m.get("strArea"), m.get("strCategory")) if t],
            "ingredients": ingredients, "steps": steps,
            "source": m.get("strSource") or m.get("strYoutube") or ""}


def _gemini_recipe(query: str, key: str, model: str):
    import requests
    payload = {
        "contents": [{"role": "user", "parts": [{"text": RECIPE_PROMPT.format(dish=query)}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "servings": {"type": "NUMBER"},
                    "time_minutes": {"type": "NUMBER"},
                    "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "steps": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["title", "ingredients", "steps"],
            },
        },
    }
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key}, json=payload, timeout=45,
        headers={"Content-Type": "application/json", "User-Agent": "NutriMatch/1.0"})
    r.raise_for_status()
    parts = ((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return _parse_recipe_json("".join(p.get("text", "") for p in parts), "gemini")


def _openai_recipe(query: str, key: str, model: str):
    import requests
    payload = {
        "model": model,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": "You answer with valid JSON only."},
                     {"role": "user", "content": RECIPE_PROMPT.format(dish=query)}],
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, timeout=45,
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json",
                               "User-Agent": "NutriMatch/1.0"})
    r.raise_for_status()
    text = (((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return _parse_recipe_json(text, "openai")


# ---- local Indian recipe dataset (most accurate source; optional) ----
_RB_STOP = {"recipe", "style", "with", "and", "in", "of", "the", "ke", "ki", "ka", "ko", "se", "aur", "a", "an", "for", "to", "indian"}
MATCH_MIN = 0.72      # below this a recipe is not shown from the dataset
MATCH_EXACT = 0.95    # at/above this the wording says "matched", below it says "similar dish"


def _rb_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in _RB_STOP and len(w) > 1}


def _rb_score(query_tokens: set, name_tokens: set, fragment: bool = False) -> float:
    if not query_tokens or not name_tokens:
        return 0.0
    overlap = len(query_tokens & name_tokens)
    cov, prec = overlap / len(query_tokens), overlap / len(name_tokens)
    seq = SequenceMatcher(None, " ".join(sorted(query_tokens)), " ".join(sorted(name_tokens))).ratio()
    score = 0.55 * cov + 0.25 * prec + 0.20 * seq
    if fragment and len(query_tokens) == 1 and len(name_tokens) > 1:   # a lone alias word ("tadka") is weak evidence
        score *= 0.85
    return score


def _rb_pick(df, *candidates):
    lower = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


@st.cache_resource(show_spinner=False)
def _recipe_book(mtime: float):
    """Load data/indian_recipes.csv once: rows + a token index for fast lookup."""
    df = pd.read_csv(RECIPE_BOOK, encoding="utf-8", encoding_errors="replace")
    cols = {
        "name": _rb_pick(df, "translatedrecipename", "recipename", "name", "title"),
        "ingredients": _rb_pick(df, "translatedingredients", "cleaningredients", "ingredients"),
        "steps": _rb_pick(df, "translatedinstructions", "instructions", "steps", "method"),
        "url": _rb_pick(df, "url", "link", "recipelink"),
        "image": _rb_pick(df, "imageurl", "image"),
        "time": _rb_pick(df, "totaltimeinmins", "totaltime", "total"),
        "servings": _rb_pick(df, "servings"),
        "course": _rb_pick(df, "course"),
        "cuisine": _rb_pick(df, "cuisine"),
        "diet": _rb_pick(df, "diet"),
    }
    if not (cols["name"] and cols["steps"] and cols["ingredients"]):
        raise ValueError("The recipe CSV needs recipe name, ingredients and instructions columns")
    df = df.dropna(subset=[cols["name"]]).reset_index(drop=True)
    tokens = [_rb_tokens(n) for n in df[cols["name"]]]
    heads = []            # last meaningful word of each name = what the dish IS ('Rava Idli' -> idli)
    for n in df[cols["name"]]:
        ordered = [w for w in re.findall(r"[a-z]+", re.split(r"\s+[-|(]", str(n).lower())[0])
                   if w not in _RB_STOP and len(w) > 1]
        heads.append(ordered[-1] if ordered else None)
    index: dict = {}
    for i, toks in enumerate(tokens):
        for t in toks:
            index.setdefault(t, []).append(i)
    return df, cols, tokens, index, heads


def _split_ingredients(text: str) -> list[str]:
    parts = re.split(r",\s*(?![^()]*\))", str(text or ""))        # commas inside (...) stay together
    return [_clean(x) for x in parts if _clean(x)]


def _split_steps(text: str) -> list[str]:
    text = str(text or "")
    chunks = [_clean(x) for x in re.split(r"\r?\n+", text) if _clean(x)]
    if len(chunks) <= 1:
        chunks = [_clean(x) for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if _clean(x)]
    return chunks


def _pretty_title(title: str) -> str:
    """'Palak Paneer Recipe - Palak Paneer Recipe' -> 'Palak Paneer'; drops the word Recipe."""
    out, seen = [], set()
    for part in re.split(r"\s+[-|\u2013]\s+", _clean(title)):
        part = re.sub(r"\(?\s*\brecipes?\b\s*\)?\s*$", "", part, flags=re.I).strip(" -|")
        part = re.sub(r"\s*\brecipe\b\s*", " ", part, flags=re.I).strip() if part.lower().endswith("recipe") else part
        key = " ".join(sorted(_rb_tokens(part)))
        if part and key not in seen:
            seen.add(key)
            out.append(part)
    return " \u2013 ".join(out) or _clean(title)


def local_recipe(dish: str):
    """Best fuzzy name match in the local Indian recipe dataset, or None."""
    if not RECIPE_BOOK.exists():
        return None
    try:
        df, cols, tokens, index, heads = _recipe_book(RECIPE_BOOK.stat().st_mtime)
    except Exception:
        return None
    base = re.sub(r"\([^)]*\)", " ", dish)
    variants = [(base, False)]
    for group in re.findall(r"\(([^)]*)\)", dish):
        variants.append((group, False))
        parts = [x for x in re.split(r"[/;]", group) if x.strip()]
        if len(parts) > 1:
            variants += [(x, True) for x in parts]
    best_i, best_s = None, 0.0
    for variant, fragment in variants:
        q = _rb_tokens(variant)
        cand = {i for t in q for i in index.get(t, ())}
        for i in cand:
            sc = _rb_score(q, tokens[i], fragment)
            if sc and heads[i] in q:
                sc = min(1.0, sc + 0.03)       # 'Rava Idli' beats 'Idli Upma' for the query 'idli'
            better = sc > best_s + 1e-9 or (abs(sc - best_s) <= 1e-9 and best_i is not None
                                            and len(tokens[i]) < len(tokens[best_i]))
            if better:
                best_i, best_s = i, sc
    if best_i is None or best_s < MATCH_MIN:
        return None
    row = df.iloc[best_i]
    meta = []
    try:
        if cols["time"] and float(row[cols["time"]]) > 0:
            meta.append(f"{int(float(row[cols['time']]))} min")
    except (TypeError, ValueError):
        pass
    for key, label in (("cuisine", "{}"), ("servings", "serves {}"), ("course", "{}"), ("diet", "{}")):
        val = _clean(row[cols[key]]) if cols[key] else ""
        val = re.sub(r"\s*recipes?$", "", val, flags=re.I)         # "South Indian Recipes" -> "South Indian"
        if val and val.lower() != "nan":
            meta.append(label.format(val))
    img = _clean(row[cols["image"]]) if cols["image"] else ""
    url = _clean(row[cols["url"]]) if cols["url"] else ""
    return {"provider": "Indian recipe dataset", "title": _pretty_title(row[cols["name"]]),
            "image": img if img.startswith("http") else "", "meta": meta,
            "ingredients": _split_ingredients(row[cols["ingredients"]]),
            "steps": _split_steps(row[cols["steps"]]), "source": url if url.startswith("http") else "",
            "score": round(best_s, 2)}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_recipe(dish: str, llm: tuple = ("", "", "")):
    """Closest recipe for a dish name, or None. Raises RecipeError if the services are unreachable."""
    local = local_recipe(dish)          # best source: a real Indian recipe dataset
    if local and local["steps"]:
        return local
    provider, key, model = llm
    tried = errors = 0
    for q in _name_variants(dish):
        calls = []
        if provider in ("gemini", "openai") and key:
            generate = _gemini_recipe if provider == "gemini" else _openai_recipe
            calls.append(lambda q=q: generate(q, key, model))
        calls.append(lambda q=q: _themealdb(q))
        for call in calls:
            tried += 1
            try:
                rec = call()
            except Exception:
                errors += 1
                continue
            if rec and rec["steps"]:
                rec["searched"] = q
                return rec
    if tried and errors == tried:
        raise RecipeError("unreachable")
    return None


def recipe_card_html(r: dict) -> str:
    img = f'<img class="recipe-img" src="{escape(r["image"], quote=True)}" alt="">' if r.get("image") else ""
    chips = "".join(f'<span class="tag">{escape(m)}</span>' for m in [f'via {r["provider"]}', *r.get("meta", [])])
    ing = "".join(f"<li>{escape(i)}</li>" for i in r["ingredients"])
    steps = "".join(f"<li>{escape(x)}</li>" for x in r["steps"])
    src = (f'<div class="recipe-src">Full recipe: <a href="{escape(r["source"], quote=True)}" target="_blank" '
           'rel="noopener">open original</a></div>') if r.get("source") else ""
    return _h(f"""<div class="recipe"><div class="recipe-head">{img}<div>
    <div class="recipe-title">{escape(r["title"])}</div><div class="recipe-meta">{chips}</div></div></div>
    <div class="recipe-cols"><div><h4>Ingredients</h4><ul>{ing}</ul></div>
    <div><h4>Method</h4><ol>{steps}</ol></div></div>{src}</div>""")


def recipe_section(dishes: list[str], prefix: str) -> None:
    """Recipe tab: pick a dish from the results; its recipe shows straight away."""
    from urllib.parse import quote_plus
    st.selectbox("Choose a dish to see its recipe", dishes, key=f"{prefix}_dish")
    active = st.session_state.get(f"{prefix}_dish") or dishes[0]
    if active not in dishes:
        active = dishes[0]

    link = f"https://www.google.com/search?q={quote_plus(active + ' recipe')}"
    try:
        with st.spinner("Looking for a recipe..."):
            rec = fetch_recipe(active, llm_config())
    except RecipeError:
        st.markdown(callout("warning", "Could not reach the recipe service. Check your internet connection and try again, "
                                       f'or <a href="{link}" target="_blank" rel="noopener">search the web for it</a>. '
                                       "If you added an AI key, make sure it is valid."),
                    unsafe_allow_html=True)
        return
    if not rec:
        st.markdown(callout("info", f"No matching recipe found for <b>{escape(active)}</b>. "
                                    f'<a href="{link}" target="_blank" rel="noopener">Search the web for it</a>. '
                                    "Adding a free Gemini (or OpenAI) API key lets the app write a recipe "
                                    "for any dish."),
                    unsafe_allow_html=True)
        return
    if rec.get("score"):
        kind = "Matched" if rec["score"] >= MATCH_EXACT else "Similar dish"
        st.caption(f"{kind}: your dish '{active}' was matched to '{rec['title']}' in the Indian recipe dataset "
                   f"(name match {int(rec['score'] * 100)}%).")
    elif rec.get("provider") in LLM_LABEL.values():
        st.caption(f"No exact recipe for '{active}' in the dataset, so {rec['provider']} wrote one from "
                   "typical Indian home cooking. Treat it as a guide and check the ingredients "
                   "against what you actually eat.")
    else:
        st.caption(f"No exact recipe found for '{active}'. This is the closest online match "
                   f"('{rec['title']}'), so it may be a different dish.")
        if not RECIPE_BOOK.exists():
            st.caption("Tip: add Cleaned_Indian_Food_Dataset.csv to the data/ folder for more accurate recipes.")
    st.markdown(recipe_card_html(rec), unsafe_allow_html=True)


# ==========================================================================
# PAGE 1: HOME
# ==========================================================================
def page_home(df, theme: str):
    with st.container():
        st.markdown(HERO_MARK, unsafe_allow_html=True)
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            st.markdown(
                '<span class="hero-pill">Nutrient-targeted food picks</span>'
                '<h1 class="hero-title">Find the dishes that fill your nutrient gap.</h1>'
                '<p class="hero-tag">Tell us the nutrient your doctor wants you to increase. NutriMatch ranks '
                'Indian dishes by how much of it they deliver, and how well they fit your diet.</p>',
                unsafe_allow_html=True)
            try:
                row = st.container(key="cta_row")     # keyed container (Streamlit 1.39+)
            except TypeError:
                row = st.container()
            with row:
                st.button("Find dishes", type="primary", key="cta_start", on_click=go_predictor)
                st.button("Read my report", key="cta_report", on_click=go_report)
                st.button("Explore the data", key="cta_explore", on_click=go_graphs)
        with right:
            st.markdown(hero_art_html(hero_pick(df)), unsafe_allow_html=True)

    st.markdown(section_head("What do you need more of?",
                             "Select one or more nutrients. Dishes are ranked on all of them together."),
                unsafe_allow_html=True)
    st.session_state.setdefault("home_nuts", [])
    if hasattr(st, "pills"):
        picked = st.pills("Nutrients", NUTRIENTS, selection_mode="multi", key="home_nuts",
                          format_func=lambda n: f"{MAT_ICON.get(n, '')} {LABELS[n].capitalize()}",
                          label_visibility="collapsed")
    else:   # older Streamlit without st.pills
        picked = st.multiselect("Nutrients", NUTRIENTS, key="home_nuts", placeholder="Choose one or more nutrients",
                                format_func=lambda n: LABELS[n].capitalize(), label_visibility="collapsed")
    count = len(picked or [])
    st.button(f"Show dishes for {count} nutrient{'s' if count != 1 else ''}" if count else "Show dishes",
              type="primary", disabled=count == 0, key="chips_go", on_click=go_with_selection)

    spots = [(n, top_dish(df, n)) for n in ("Iron (mg)", "Calcium (mg)", "Protein (g)", "Fibre (g)")]
    spots = [(n, t) for n, t in spots if t]
    if spots:
        st.markdown(section_head("Nutrient spotlights", "The richest meal in the dataset for each nutrient."),
                    unsafe_allow_html=True)
        cols = st.columns(len(spots), gap="medium")
        for col, (n, (name, val, unit, dtype)) in zip(cols, spots):
            col.markdown(spotlight_card(NUT_ICON.get(n, "leaf"), f"Richest in {LABELS[n]}",
                                        name, val, unit, dtype), unsafe_allow_html=True)

    st.markdown(section_head("The dataset behind it"), unsafe_allow_html=True)
    st.markdown(stat_strip([
        ("Indian dishes", f"{len(df):,}", "per-serving nutrition values"),
        ("Nutrients tracked", f"{len(NUTRIENTS)}", "energy, macros, minerals, vitamins"),
        ("Vegetarian", f"{int((df['is_nonveg'] == 0).sum()):,}",
         f"{int(df['is_nonveg'].sum()):,} non-vegetarian"),
        ("Typical energy", f"{df['Calories (kcal)'].median():,.0f} kcal", "median dish, one serving"),
    ]), unsafe_allow_html=True)
    st.markdown(f'<div class="foot">Values are per serving from an Indian food nutrition table of {len(df):,} dishes.</div>',
                unsafe_allow_html=True)


# ==========================================================================
# PAGE 2: GRAPHS
# ==========================================================================
def page_graphs(df, res, theme: str):
    st.markdown(page_header(
        "Graphs and statistics",
        "How each nutrient spreads across Indian dishes, how nutrients group together, "
        "and how well the prediction models perform."), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Nutrient distributions", "By dish type", "Correlations", "Model performance"])

    with tab1:
        with st.container():
            st.markdown(CARD, unsafe_allow_html=True)
            sel_c, opt_c = st.columns([2, 1.4], gap="large")
            with sel_c:
                nutrient = st.selectbox("Nutrient", NUTRIENTS, index=6, key="g_nut")
            with opt_c:
                log_scale = st.toggle("Log scale (skewed data)", key="g_log")
                trim = st.toggle("Trim top 1% outliers", key="g_trim")

            unit = UNITS.get(nutrient, "")
            sub = df.dropna(subset=[nutrient])
            raw = sub[nutrient]
            plot_sub = sub[raw <= raw.quantile(0.99)] if trim else sub
            xvals = np.log1p(plot_sub[nutrient]) if log_scale else plot_sub[nutrient]
            tmp = pd.DataFrame({"value": xvals})
            fig = px.histogram(tmp, x="value", nbins=50,
                               title=f"Distribution of {nutrient_label(nutrient)}",
                               color_discrete_sequence=[PALETTES[theme]["primary"]])
            prefix = "log1p(" if log_scale else ""
            suffix = ")" if log_scale else ""
            apply_plotly_theme(fig, theme, height=400,
                               xlabel=f"{prefix}{nutrient_label(nutrient)}{suffix}",
                               ylabel="number of dishes")
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            m = st.columns(5)
            m[0].metric("Dishes", f"{len(sub):,}")
            m[1].metric("Mean", f"{raw.mean():,.1f} {unit}")
            m[2].metric("Median", f"{raw.median():,.1f} {unit}")
            m[3].metric("Std dev", f"{raw.std():,.1f} {unit}")
            m[4].metric("'High' cut-off", f"{raw.quantile(0.75):,.1f} {unit}",
                        help="75th percentile = the 'high' threshold used by the predictor")

        st.markdown(callout("info",
                            f"<b>Interpretation:</b> the median dish provides <b>{raw.median():.1f} {unit}</b> of "
                            f"{LABELS.get(nutrient, nutrient)}. The 75th percentile "
                            f"(<b>{raw.quantile(0.75):,.1f} {unit}</b>) is the 'high' cut-off used by the predictor. "
                            f"A wide spread (std = {raw.std():,.1f} {unit}) means this nutrient varies strongly "
                            f"between dishes."), unsafe_allow_html=True)

    with tab2:
        metric = st.selectbox("Metric", NUTRIENTS, index=6, key="g_metric")
        sub = df.dropna(subset=[metric])
        fig = px.box(sub, x="dish_type", y=metric, color="dish_type",
                     color_discrete_sequence=COLORWAY[theme],
                     title=f"{LABELS.get(metric, metric).capitalize()} by dish type")
        apply_plotly_theme(fig, theme, height=440, xlabel="dish type", ylabel=nutrient_label(metric))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.markdown(callout("info",
                            "<b>Interpretation:</b> desserts and snacks tend to be dense in calories and sugar; "
                            "mains and sides drive protein, fibre and minerals. Each box shows the median and "
                            "inter-quartile range; dots are outlier dishes."), unsafe_allow_html=True)

    with tab3:
        cols = ["Calories (kcal)", "Protein (g)", "Fats (g)", "Fibre (g)",
                "Sodium (mg)", "Calcium (mg)", "Iron (mg)", "Vitamin C (mg)"]
        corr = df[cols].corr(method="spearman")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=HEAT_SCALE[theme],
                        title="Spearman correlations between nutrients")
        apply_plotly_theme(fig, theme, height=540)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.markdown(callout("info",
                            "<b>Interpretation:</b> nutrients tend to travel together - dense foods score high on "
                            "many nutrients at once - which is why simple rules beat random guessing."),
                    unsafe_allow_html=True)

    with tab4:
        if res is None:
            st.warning("model_results.csv not found - add it to the data/ folder.")
            return
        grey = "#9aa79d"
        st.subheader("Classification: can we predict 'high in nutrient'?")
        clf = res[res["task"] == "clf"].copy()
        fig = px.bar(clf, x="nutrient", y="f1", color="model", barmode="group",
                     title="F1 score by model and nutrient (Dummy = 0)",
                     color_discrete_map={"Dummy": grey, "LogReg": COLORWAY[theme][1],
                                         "RandomForest": PALETTES[theme]["primary"]})
        apply_plotly_theme(fig, theme, height=420, xlabel="nutrient", ylabel="F1 score")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        dummy_acc = float(clf[clf["model"] == "Dummy"]["accuracy"].mean())
        best_f1 = float(clf[clf["model"] == "RandomForest"]["f1"].max())
        st.markdown(callout(
            "info",
            f"<b>Why F1 instead of accuracy:</b> about {dummy_acc:.0%} of dishes fall below the "
            f"'high' cut-off, so a model that always answers 'not high' already scores "
            f"{dummy_acc:.0%} accuracy - accuracy flatters every model. F1 (0 = useless, 1 = perfect) "
            f"measures how well the 'high' dishes are actually found; the best RandomForest here "
            f"reaches {best_f1:.2f}. Accuracy, MAE and R squared for every model are in the table below."),
            unsafe_allow_html=True)

        st.subheader("Regression: can we predict the nutrient amount?")
        reg = res[res["task"] == "reg"].copy()
        fig = px.bar(reg, x="nutrient", y="r2", color="model", barmode="group",
                     title="R squared by model and nutrient",
                     color_discrete_map={"Dummy": grey, "Ridge": COLORWAY[theme][1],
                                         "RandomForest": PALETTES[theme]["primary"]})
        apply_plotly_theme(fig, theme, height=420, xlabel="nutrient", ylabel="R squared")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("#### Full results table")
        st.dataframe(res, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 3: PREDICTOR (live: results update as you change any control)
# ==========================================================================
def page_predictor(df, res, theme: str):
    st.markdown(page_header(
        "Find dishes for your nutrients",
        "Choose one or more nutrients the doctor asked the patient to increase and set the patient's "
        "food preferences. The ranked list updates as you go."), unsafe_allow_html=True)

    st.session_state.setdefault("p_nuts", ["Iron (mg)"])
    # a view saved by an older version of the app (e.g. "Chart") would crash the radio
    if st.session_state.get("p_view") not in PRED_VIEWS:
        st.session_state["p_view"] = PRED_VIEWS[0]
    left, right = st.columns([0.9, 1.5], gap="large")

    with left:
        with st.container():
            st.markdown(CARD, unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Nutrient need</div>', unsafe_allow_html=True)
            nutrients = st.multiselect("Which nutrients should the dish be rich in?", NUTRIENTS,
                                       key="p_nuts", format_func=nutrient_label,
                                       placeholder="Choose one or more")
            st.divider()
            st.markdown('<div class="panel-title">Food preferences</div>', unsafe_allow_html=True)
            prefer_types = st.multiselect("Preferred dish types",
                                          sorted(df["dish_type"].dropna().unique()),
                                          placeholder="Any type", key="p_types")
            st.divider()
            st.markdown('<div class="panel-title">Dietary filters</div>', unsafe_allow_html=True)
            veg_only = st.checkbox("Vegetarian only", key="p_veg")
            avoid_sweet = st.checkbox("Avoid sweets", key="p_sweet")
            avoid_fried = st.checkbox("Avoid fried food", key="p_fried")
            meals_only = st.checkbox("Meals only (skip condiments & beverages)", value=True, key="p_meals")
            top_n = st.slider("Number of suggestions", 5, 20, 10, key="p_n")

    with right:
        if not nutrients:
            st.markdown(empty_state("target", "Choose a nutrient to begin",
                                    "Pick one or more nutrients on the left and the ranked dishes appear here."),
                        unsafe_allow_html=True)
            return

        multi = len(nutrients) > 1
        if multi:
            rec = recommend_multi(df, nutrients, prefer_types, veg_only, avoid_sweet,
                                  avoid_fried, meals_only, top_n)
        else:
            rec = recommend(df, nutrients[0], prefer_types, veg_only, avoid_sweet,
                            avoid_fried, meals_only, top_n)

        if rec.empty:
            st.markdown(empty_state(
                "search", "No dish matches all the filters",
                "Try relaxing one: untick 'Vegetarian only', broaden the dish types, or untick 'Meals only'."),
                unsafe_allow_html=True)
            return

        nutrient = nutrients[0]
        names_l = " + ".join(LABELS.get(n, n) for n in nutrients)
        top = rec.iloc[0]
        if multi:
            detail = " · ".join(f"{LABELS.get(n, n).capitalize()} {float(top[n]):,.1f} {UNITS.get(n, '')}"
                                for n in nutrients)
            st.markdown(insight_banner("target", f"Best all-rounder for {names_l}",
                                       escape(str(top["Dish Name"])), detail), unsafe_allow_html=True)
        else:
            unit = UNITS.get(nutrient, "")
            med = float(df[nutrient].median())
            mult = f"{float(top[nutrient]) / med:.1f}x the median dish" if med else "top of the list"
            st.markdown(insight_banner(
                NUT_ICON.get(nutrient, "leaf"), f"Top pick for {names_l}", escape(str(top["Dish Name"])),
                f"{float(top[nutrient]):,.1f} {unit} per serving, {mult}"), unsafe_allow_html=True)

        view_c, dl_c = st.columns([2.4, 1])
        with view_c:
            view = st.radio("View", PRED_VIEWS, horizontal=True,
                            key="p_view", label_visibility="collapsed")
        with dl_c:
            st.download_button(
                "Download CSV", rec.to_csv(index=False).encode("utf-8"),
                file_name=f"nutrimatch_{names_l.replace(' + ', '_').replace(' ', '')}_{top_n}.csv",
                mime="text/csv", key="p_csv")

        if view == "Table":
            rename = {"dish_type": "Dish type", "match": "match %" if multi else "top %"}
            rename.update({n: nutrient_label(n) for n in nutrients})
            cols = ["Dish Name", "Dish type", *[nutrient_label(n) for n in nutrients], rename["match"]]
            st.dataframe(rec.rename(columns=rename)[cols], use_container_width=True, hide_index=True)
        elif view == "Recipe":
            recipe_section(list(rec["Dish Name"]), "pred")
        elif multi:
            st.markdown(dish_grid_multi_html(df, rec, nutrients), unsafe_allow_html=True)
        else:
            st.markdown(dish_grid_html(df, rec, nutrient), unsafe_allow_html=True)

        if not multi and res is not None and nutrient in res["nutrient"].values:
            with st.expander("How reliable is this?"):
                best = res[(res["nutrient"] == nutrient) & (res["model"] == "RandomForest")]
                dummy = res[(res["nutrient"] == nutrient) & (res["task"] == "clf")
                            & (res["model"] == "Dummy")]
                c1, c2 = st.columns(2)
                if len(best[best["task"] == "clf"]):
                    clf_row = best[best["task"] == "clf"].iloc[0]
                    c1.metric("Classifier F1", f"{clf_row['f1']:.2f}")
                    note = f"accuracy {clf_row['accuracy']:.0%}"
                    if len(dummy):
                        note += f" vs {float(dummy['accuracy'].values[0]):.0%} baseline"
                    c1.caption(f"({note})")
                if len(best[best["task"] == "reg"]):
                    reg_row = best[best["task"] == "reg"].iloc[0]
                    c2.metric("Regression R squared", f"{reg_row['r2']:.2f}")
                    c2.caption(f"(typical error ± {reg_row['mae']:,.1f} {UNITS.get(nutrient, '')})")
                st.caption("The classifier separates 'high' dishes from the rest - shown as F1, "
                           "not accuracy, because about three quarters of dishes are 'not high'. "
                           "The regressor predicts the amount; read R squared together with its "
                           "typical error (MAE).")

        if multi:
            st.caption("Combined match averages how each dish ranks on every nutrient you chose. "
                       "Dishes missing data for any of them are left out.")
        else:
            st.caption("Values are per serving; 'high' means the top quarter of dishes for that nutrient. "
                       "The ring on each card shows the dish's nutrient percentile.")


# ============================================================
# PAGE: MY DAILY NUTRITION
# ============================================================

DAILY_REFERENCES = {
    "Calories (kcal)": (2000, "target"),
    "Carbohydrates (g)": (275, "target"),
    "Protein (g)": (50, "target"),
    "Fats (g)": (70, "target"),
    "Free Sugar (g)": (50, "limit"),
    "Fibre (g)": (25, "target"),
    "Sodium (mg)": (2000, "limit"),
    "Calcium (mg)": (1000, "target"),
    "Iron (mg)": (18, "target"),
    "Vitamin C (mg)": (75, "target"),
    "Folate (ug)": (400, "target"),
}


MODEL_FILES = {
    "Sodium (mg)": "Sodium_mg_reg.joblib",
    "Iron (mg)": "Iron_mg_reg.joblib",
    "Calcium (mg)": "Calcium_mg_reg.joblib",
    "Protein (g)": "Protein_g_reg.joblib",
    "Fibre (g)": "Fibre_g_reg.joblib",
}

EST_INGREDIENTS = [
    # Grains, Flours & Bases
    ("Rice / Poha / Murmura", ["is_rice"]),
    ("Wheat / Roti / Flour", ["is_wheat"]),
    ("Millets / Ragi / Jowar / Oats", ["is_millet"]),
    ("Semolina / Suji / Rava", ["is_semolina"]),
    ("Besan / Gram Flour", ["is_besan"]),
    ("Pasta / Noodles / Vermicelli", ["is_pasta_noodle"]),
    # Proteins
    ("Lentils / Dal", ["is_legume"]),
    ("Chickpeas / Chole / Rajma / Beans", ["is_chickpea_bean", "is_legume"]),
    ("Soya / Nutrinuggets", ["is_soya", "is_legume"]),
    ("Paneer / Chhena / Tofu", ["is_paneer", "is_dairy"]),
    ("Cheese / Cream", ["is_cheese_cream", "is_dairy"]),
    ("Egg", ["is_egg"]),
    ("Chicken / Poultry", ["is_chicken", "is_poultry"]),
    ("Mutton / Lamb / Keema", ["is_meat", "is_red_meat"]),
    ("Fish / Prawn / Seafood", ["is_fish"]),
    # Vegetables & Fruits
    ("Leafy Greens (Palak, Methi, Saag)", ["is_leafy_green", "is_vegetable"]),
    ("Potato / Sweet Potato (Aloo)", ["is_potato", "is_vegetable"]),
    ("Gourds / Brinjal / Okra / Pumpkin", ["is_gourd_squash", "is_vegetable"]),
    ("Cauliflower / Cabbage / Broccoli", ["is_cruciferous", "is_vegetable"]),
    ("Peas / Corn / Green Beans", ["is_pea_corn", "is_vegetable"]),
    ("Tomato (Tamatar)", ["is_tomato", "is_vegetable"]),
    ("Onion / Garlic / Ginger", ["is_onion_garlic", "is_vegetable"]),
    ("Mixed Vegetables", ["is_vegetable"]),
    ("Fresh Fruit (Mango, Banana, Apple, etc.)", ["is_fruit"]),
    ("Citrus / Lemon / Tamarind / Amla", ["is_citrus_tangy"]),
    # Dairy, Fats, Nuts & Seeds
    ("Milk / Curd / Yogurt / Khoa", ["is_dairy"]),
    ("Ghee / Butter", ["is_ghee_butter"]),
    ("Coconut (Fresh / Milk / Desiccated)", ["is_coconut"]),
    ("Nuts (Almond, Cashew, Peanut, Walnut)", ["is_nuts_seed"]),
    ("Sesame / Til / Poppy / Chia Seeds", ["is_sesame", "is_nuts_seed"]),
    # Preparation & Style
    ("Deep Fried / Crispy", ["is_fried"]),
    ("Sweetened (Sugar / Jaggery / Honey)", ["is_sugar", "is_sweet"]),
    ("Spicy / Masala / Chilli", ["is_spicy"]),
    ("Extra Salted / Pickled / Papad", ["is_salted"]),
    ("Roasted / Tandoori / Baked", ["is_cured"]),
]

# Culinary domain rules: only allow ingredients that make sense for the selected Dish Type!
ALLOWED_INGREDIENTS_BY_TYPE = {
    "Main": [
        "Rice / Poha / Murmura", "Wheat / Roti / Flour", "Millets / Ragi / Jowar / Oats",
        "Besan / Gram Flour", "Pasta / Noodles / Vermicelli", "Lentils / Dal",
        "Chickpeas / Chole / Rajma / Beans", "Soya / Nutrinuggets", "Paneer / Chhena / Tofu",
        "Cheese / Cream", "Egg", "Chicken / Poultry", "Mutton / Lamb / Keema",
        "Fish / Prawn / Seafood", "Leafy Greens (Palak, Methi, Saag)", "Potato / Sweet Potato (Aloo)",
        "Gourds / Brinjal / Okra / Pumpkin", "Cauliflower / Cabbage / Broccoli",
        "Peas / Corn / Green Beans", "Tomato (Tamatar)", "Onion / Garlic / Ginger",
        "Mixed Vegetables", "Citrus / Lemon / Tamarind / Amla", "Milk / Curd / Yogurt / Khoa",
        "Ghee / Butter", "Coconut (Fresh / Milk / Desiccated)", "Nuts (Almond, Cashew, Peanut, Walnut)",
        "Sesame / Til / Poppy / Chia Seeds", "Deep Fried / Crispy", "Spicy / Masala / Chilli",
        "Roasted / Tandoori / Baked"
    ],
    "Side": [
        "Lentils / Dal", "Chickpeas / Chole / Rajma / Beans", "Soya / Nutrinuggets",
        "Paneer / Chhena / Tofu", "Cheese / Cream", "Besan / Gram Flour",
        "Leafy Greens (Palak, Methi, Saag)", "Potato / Sweet Potato (Aloo)",
        "Gourds / Brinjal / Okra / Pumpkin", "Cauliflower / Cabbage / Broccoli",
        "Peas / Corn / Green Beans", "Tomato (Tamatar)", "Onion / Garlic / Ginger",
        "Mixed Vegetables", "Citrus / Lemon / Tamarind / Amla", "Milk / Curd / Yogurt / Khoa",
        "Ghee / Butter", "Coconut (Fresh / Milk / Desiccated)", "Nuts (Almond, Cashew, Peanut, Walnut)",
        "Sesame / Til / Poppy / Chia Seeds", "Deep Fried / Crispy", "Spicy / Masala / Chilli",
        "Roasted / Tandoori / Baked"
    ],
    "Bread": [
        "Wheat / Roti / Flour", "Millets / Ragi / Jowar / Oats", "Besan / Gram Flour",
        "Rice / Poha / Murmura", "Semolina / Suji / Rava", "Paneer / Chhena / Tofu",
        "Lentils / Dal", "Mutton / Lamb / Keema", "Egg",
        "Leafy Greens (Palak, Methi, Saag)", "Potato / Sweet Potato (Aloo)",
        "Cauliflower / Cabbage / Broccoli", "Peas / Corn / Green Beans",
        "Onion / Garlic / Ginger", "Milk / Curd / Yogurt / Khoa", "Ghee / Butter",
        "Sesame / Til / Poppy / Chia Seeds", "Deep Fried / Crispy",
        "Spicy / Masala / Chilli", "Roasted / Tandoori / Baked"
    ],
    "Snack": [
        "Rice / Poha / Murmura", "Wheat / Roti / Flour", "Millets / Ragi / Jowar / Oats",
        "Semolina / Suji / Rava", "Besan / Gram Flour", "Pasta / Noodles / Vermicelli",
        "Lentils / Dal", "Chickpeas / Chole / Rajma / Beans", "Soya / Nutrinuggets",
        "Paneer / Chhena / Tofu", "Cheese / Cream", "Egg", "Chicken / Poultry",
        "Mutton / Lamb / Keema", "Fish / Prawn / Seafood",
        "Leafy Greens (Palak, Methi, Saag)", "Potato / Sweet Potato (Aloo)",
        "Cauliflower / Cabbage / Broccoli", "Peas / Corn / Green Beans",
        "Tomato (Tamatar)", "Onion / Garlic / Ginger", "Mixed Vegetables",
        "Citrus / Lemon / Tamarind / Amla", "Milk / Curd / Yogurt / Khoa",
        "Ghee / Butter", "Coconut (Fresh / Milk / Desiccated)",
        "Nuts (Almond, Cashew, Peanut, Walnut)", "Sesame / Til / Poppy / Chia Seeds",
        "Deep Fried / Crispy", "Spicy / Masala / Chilli", "Extra Salted / Pickled / Papad",
        "Roasted / Tandoori / Baked"
    ],
    "Dessert": [
        "Milk / Curd / Yogurt / Khoa", "Cheese / Cream", "Paneer / Chhena / Tofu",
        "Ghee / Butter", "Sweetened (Sugar / Jaggery / Honey)",
        "Nuts (Almond, Cashew, Peanut, Walnut)", "Sesame / Til / Poppy / Chia Seeds",
        "Coconut (Fresh / Milk / Desiccated)", "Rice / Poha / Murmura",
        "Wheat / Roti / Flour", "Semolina / Suji / Rava", "Besan / Gram Flour",
        "Millets / Ragi / Jowar / Oats", "Lentils / Dal", "Fresh Fruit (Mango, Banana, Apple, etc.)",
        "Gourds / Brinjal / Okra / Pumpkin", "Egg", "Deep Fried / Crispy",
        "Roasted / Tandoori / Baked"
    ],
    "Soup": [
        "Tomato (Tamatar)", "Leafy Greens (Palak, Methi, Saag)", "Peas / Corn / Green Beans",
        "Cauliflower / Cabbage / Broccoli", "Potato / Sweet Potato (Aloo)",
        "Gourds / Brinjal / Okra / Pumpkin", "Onion / Garlic / Ginger", "Mixed Vegetables",
        "Lentils / Dal", "Chickpeas / Chole / Rajma / Beans", "Millets / Ragi / Jowar / Oats",
        "Pasta / Noodles / Vermicelli", "Chicken / Poultry", "Mutton / Lamb / Keema",
        "Egg", "Paneer / Chhena / Tofu", "Cheese / Cream", "Milk / Curd / Yogurt / Khoa",
        "Ghee / Butter", "Coconut (Fresh / Milk / Desiccated)",
        "Nuts (Almond, Cashew, Peanut, Walnut)", "Citrus / Lemon / Tamarind / Amla",
        "Spicy / Masala / Chilli"
    ],
    "Salad": [
        "Mixed Vegetables", "Tomato (Tamatar)", "Onion / Garlic / Ginger",
        "Leafy Greens (Palak, Methi, Saag)", "Potato / Sweet Potato (Aloo)",
        "Peas / Corn / Green Beans", "Cauliflower / Cabbage / Broccoli",
        "Gourds / Brinjal / Okra / Pumpkin", "Fresh Fruit (Mango, Banana, Apple, etc.)",
        "Citrus / Lemon / Tamarind / Amla", "Chickpeas / Chole / Rajma / Beans",
        "Lentils / Dal", "Paneer / Chhena / Tofu", "Cheese / Cream",
        "Milk / Curd / Yogurt / Khoa", "Egg", "Chicken / Poultry",
        "Pasta / Noodles / Vermicelli", "Nuts (Almond, Cashew, Peanut, Walnut)",
        "Sesame / Til / Poppy / Chia Seeds", "Coconut (Fresh / Milk / Desiccated)",
        "Spicy / Masala / Chilli"
    ],
    "Beverage": [
        "Milk / Curd / Yogurt / Khoa", "Cheese / Cream",
        "Fresh Fruit (Mango, Banana, Apple, etc.)", "Citrus / Lemon / Tamarind / Amla",
        "Coconut (Fresh / Milk / Desiccated)", "Nuts (Almond, Cashew, Peanut, Walnut)",
        "Sesame / Til / Poppy / Chia Seeds", "Millets / Ragi / Jowar / Oats",
        "Semolina / Suji / Rava", "Besan / Gram Flour",
        "Sweetened (Sugar / Jaggery / Honey)", "Spicy / Masala / Chilli",
        "Extra Salted / Pickled / Papad"
    ],
}

EST_FLAGS = [
    "is_nonveg", "is_chicken", "is_poultry", "is_fish", "is_meat", "is_red_meat", "is_egg",
    "is_paneer", "is_cheese_cream", "is_dairy", "is_legume", "is_chickpea_bean", "is_soya",
    "is_rice", "is_wheat", "is_millet", "is_semolina", "is_besan", "is_pasta_noodle",
    "is_vegetable", "is_leafy_green", "is_potato", "is_gourd_squash", "is_cruciferous",
    "is_pea_corn", "is_tomato", "is_onion_garlic", "is_fruit", "is_citrus_tangy",
    "is_nuts_seed", "is_sesame", "is_coconut", "is_ghee_butter",
    "is_salted", "is_cured", "is_spicy", "is_sugar", "is_sweet", "is_fried"
]
EST_NONVEG = ["is_egg", "is_chicken", "is_poultry", "is_meat", "is_red_meat", "is_fish"]


def _make_predictable(model):
    """Let a model saved by an older scikit-learn still predict on a newer one.

    Trees pickled before scikit-learn 1.4 have no ``monotonic_cst``, which newer
    versions look for at predict time; fill in the default so old files still run.
    """
    def fix(est):
        if est is not None and not hasattr(est, "monotonic_cst"):
            try:
                est.monotonic_cst = None
            except Exception:
                pass

    try:
        fix(model)
        for est in getattr(model, "estimators_", None) or []:
            fix(est)
    except Exception:
        pass
    return model


@st.cache_resource(show_spinner=False)
def load_estimator_models() -> dict:
    """All shipped regression models: nutrient -> {pipeline, features, mae, r2}."""
    models = {}
    for nutrient, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.is_file():
            models[nutrient] = _make_predictable(joblib.load(path))
    return models


# Scientifically calibrated ICMR-NIN (IFCT 2017) per-100g cooked ingredient contributions:
# (Sodium_mg, Iron_mg, Calcium_mg, Protein_g, Fibre_g)
IFCT_PRIORS = {
    "is_chicken":       (75.0,  1.4,  18.0, 18.5, 0.0),
    "is_red_meat":      (85.0,  2.6,  22.0, 19.5, 0.0),
    "is_fish":          (90.0,  1.3,  45.0, 17.5, 0.0),
    "is_egg":           (125.0, 1.8,  50.0, 12.5, 0.0),
    "is_paneer":        (40.0,  0.5, 210.0, 14.5, 0.0),
    "is_cheese_cream":  (280.0, 0.3, 240.0, 11.0, 0.0),
    "is_dairy":         (45.0,  0.2, 120.0,  4.2, 0.0),
    "is_soya":          (30.0,  4.2, 110.0, 16.0, 4.5),
    "is_chickpea_bean": (35.0,  2.8,  55.0,  8.5, 5.8),
    "is_legume":        (30.0,  2.4,  38.0,  7.8, 4.4),
    "is_leafy_green":   (65.0,  3.8, 115.0,  2.8, 3.2),
    "is_sesame":        (15.0,  5.2, 340.0,  6.5, 4.0),
    "is_nuts_seed":     (15.0,  2.2,  75.0,  6.0, 3.0),
    "is_millet":        (15.0,  2.9, 130.0,  4.5, 3.8),
    "is_wheat":         (120.0, 2.2,  30.0,  6.8, 3.6),
    "is_besan":         (90.0,  2.6,  45.0,  8.2, 4.2),
    "is_rice":          (15.0,  0.6,  12.0,  2.7, 0.8),
    "is_semolina":      (20.0,  1.2,  22.0,  3.8, 1.4),
    "is_pasta_noodle":  (140.0, 1.1,  18.0,  4.5, 1.5),
    "is_cruciferous":   (35.0,  1.1,  42.0,  2.1, 2.6),
    "is_pea_corn":      (25.0,  1.4,  28.0,  4.2, 3.4),
    "is_potato":        (20.0,  0.7,  14.0,  1.8, 1.8),
    "is_gourd_squash":  (18.0,  0.8,  26.0,  1.2, 2.0),
    "is_tomato":        (25.0,  0.6,  18.0,  1.0, 1.2),
    "is_onion_garlic":  (20.0,  0.5,  22.0,  1.1, 1.3),
    "is_vegetable":     (25.0,  1.0,  32.0,  1.6, 2.4),
    "is_fruit":         (8.0,   0.5,  16.0,  0.8, 2.2),
    "is_citrus_tangy":  (25.0,  0.8,  28.0,  0.6, 1.5),
    "is_coconut":       (20.0,  1.4,  24.0,  2.8, 4.2),
    "is_ghee_butter":   (45.0,  0.1,  15.0,  0.2, 0.0),
    "is_salted":        (260.0, 0.3,  15.0,  0.0, 0.0),
    "is_spicy":         (140.0, 0.6,  20.0,  0.4, 0.6),
    "is_fried":         (95.0,  0.2,   5.0,  0.5, 0.2),
    "is_cured":         (130.0, 0.4,  10.0,  1.5, 0.0),
    "is_sweet":         (10.0,  0.3,  15.0,  0.0, 0.0),
}

DISH_BASE_PRIORS = {
    "Main":     (160.0, 1.1, 35.0, 3.2, 1.4),
    "Side":     (140.0, 1.0, 32.0, 2.2, 1.8),
    "Bread":    (150.0, 1.8, 28.0, 6.5, 3.0),
    "Snack":    (210.0, 1.3, 30.0, 4.5, 2.0),
    "Dessert":  (35.0,  0.6, 65.0, 3.5, 0.8),
    "Soup":     (220.0, 0.8, 24.0, 2.0, 1.0),
    "Salad":    (110.0, 1.0, 30.0, 2.0, 2.2),
    "Beverage": (30.0,  0.3, 45.0, 1.8, 0.4),
}


def compute_ifct_priors(dish_type: str, flags: dict) -> dict:
    """Calculate ICMR-NIN (IFCT 2017) ingredient-weighted priors for 100 g of the dish."""
    base_na, base_fe, base_ca, base_pr, base_fb = DISH_BASE_PRIORS.get(
        dish_type, (120.0, 1.0, 30.0, 3.0, 1.5)
    )
    active = [k for k, v in flags.items() if v == 1.0 and k in IFCT_PRIORS]
    if not active:
        return {
            "prior_Sodium": base_na, "prior_Iron": base_fe,
            "prior_Calcium": base_ca, "prior_Protein": base_pr, "prior_Fibre": base_fb
        }

    # Separate primary bulk ingredients from seasoning/preparation modifiers
    modifiers = {"is_salted", "is_spicy", "is_fried", "is_cured", "is_sweet", "is_ghee_butter", "is_citrus_tangy"}
    bulk = [k for k in active if k not in modifiers]
    mods = [k for k in active if k in modifiers]

    if bulk:
        # Weighted combination: dominant rich ingredients carry higher culinary weight in 100g
        na_vals = [IFCT_PRIORS[k][0] for k in bulk]
        fe_vals = [IFCT_PRIORS[k][1] for k in bulk]
        ca_vals = [IFCT_PRIORS[k][2] for k in bulk]
        pr_vals = [IFCT_PRIORS[k][3] for k in bulk]
        fb_vals = [IFCT_PRIORS[k][4] for k in bulk]

        na = 0.35 * base_na + 0.65 * (0.60 * max(na_vals) + 0.40 * float(np.mean(na_vals)))
        fe = 0.20 * base_fe + 0.80 * (0.65 * max(fe_vals) + 0.35 * float(np.mean(fe_vals)))
        ca = 0.20 * base_ca + 0.80 * (0.65 * max(ca_vals) + 0.35 * float(np.mean(ca_vals)))
        pr = 0.15 * base_pr + 0.85 * (0.65 * max(pr_vals) + 0.35 * float(np.mean(pr_vals)))
        fb = 0.20 * base_fb + 0.80 * (0.60 * max(fb_vals) + 0.40 * float(np.mean(fb_vals)))
    else:
        na, fe, ca, pr, fb = base_na, base_fe, base_ca, base_pr, base_fb

    # Add preparation/seasoning modifiers additively
    for m in mods:
        m_na, m_fe, m_ca, m_pr, m_fb = IFCT_PRIORS[m]
        na += m_na * 0.75
        fe += m_fe * 0.50
        ca += m_ca * 0.50
        pr += m_pr * 0.50
        fb += m_fb * 0.50

    return {
        "prior_Sodium": round(min(na, 650.0), 2),
        "prior_Iron": round(min(fe, 12.0), 2),
        "prior_Calcium": round(min(ca, 480.0), 2),
        "prior_Protein": round(min(pr, 28.0), 2),
        "prior_Fibre": round(min(fb, 14.0), 2),
    }


def _reset_est_ingredients() -> None:
    """Automatically reset selected ingredients whenever Dish type changes."""
    st.session_state["est_ing_sel"] = []


def render_ml_nutrient_estimator(df):
    """Live reactive 100 g nutrient estimator (no button needed; auto-resets on change)."""
    models = load_estimator_models()
    st.markdown(
        section_head(
            "Estimate a dish not in our table",
            "Pick the dish type and what is in it. Estimates update live as you add or remove "
            "ingredients, and automatically reset when you switch dish types."
        ),
        unsafe_allow_html=True
    )
    st.markdown(
        callout(
            "info",
            "<b>Standard 100 g Portion Disclaimer:</b> All food items and nutritional estimates across "
            "NutriMatch are strictly measured per <b>100 g</b> of edible portion (default and fixed at 100 g)."
        ),
        unsafe_allow_html=True
    )

    if not models:
        st.warning("Saved regression models were not found. Check that the .joblib files are inside data/")
        return

    with st.container():
        st.markdown(CARD, unsafe_allow_html=True)
        type_c, ing_c = st.columns([1, 2.2], gap="large")
        available_types = [t for t in ALLOWED_INGREDIENTS_BY_TYPE.keys() if t in df["dish_type"].unique()]
        if not available_types:
            available_types = sorted(df["dish_type"].dropna().astype(str).unique())

        with type_c:
            dish_type = st.selectbox(
                "Dish type",
                available_types,
                key="est_dish_type",
                on_change=_reset_est_ingredients,
            )

        allowed_labels = ALLOWED_INGREDIENTS_BY_TYPE.get(
            dish_type, [label for label, _ in EST_INGREDIENTS]
        )
        if "est_ing_sel" in st.session_state:
            st.session_state["est_ing_sel"] = [
                x for x in st.session_state["est_ing_sel"] if x in allowed_labels
            ]

        with ing_c:
            ingredients = st.multiselect(
                f"What is in it? (filtered for {dish_type.lower()} dishes)",
                allowed_labels,
                key="est_ing_sel",
                placeholder=f"Select 1 or more ingredients for your {dish_type.lower()}..."
            )

    # When no ingredients are selected, reset/hide the cards instead of showing fake numbers!
    if not ingredients:
        st.caption("Select at least one ingredient above to see its live 100 g nutritional estimate.")
        return

    flags = {f: 0.0 for f in EST_FLAGS}
    ing_map = dict(EST_INGREDIENTS)
    for label in ingredients:
        for feat in ing_map.get(label, []):
            flags[feat] = 1.0
    if dish_type == "Dessert":
        flags["is_sweet"] = 1.0
        flags["is_sugar"] = 1.0
    if dish_type in ("Main", "Side", "Soup"):
        flags["is_salted"] = 1.0
    flags["is_nonveg"] = 1.0 if any(flags[f] for f in EST_NONVEG) else 0.0

    priors = compute_ifct_priors(dish_type, flags)

    items = []
    for nutrient, saved in models.items():
        feats = saved.get("features") or []
        pipeline = saved.get("pipeline")
        if pipeline is None or not feats:
            continue

        row = {f: flags.get(f, 0.0) for f in feats if f not in ("dish_type", *priors.keys())}
        row.update(priors)
        row["dish_type"] = dish_type

        try:
            X_one = pd.DataFrame([row])[feats]
            ml_pred = max(0.0, float(np.asarray(pipeline.predict(X_one)).reshape(-1)[0]))
            stem_short = nutrient.split(" ")[0]
            prior_val = priors.get(f"prior_{stem_short}", ml_pred)
            # Blend trained ML model with ICMR-NIN IFCT ingredient composition prior
            pred_100g = 0.65 * ml_pred + 0.35 * prior_val
        except Exception as exc:
            st.error(f"Could not estimate {LABELS.get(nutrient, nutrient)}: {exc}")
            continue

        unit = UNITS.get(nutrient, "")
        mae = saved.get("mae")
        r2 = saved.get("r2")
        sub = "per 100 g"
        if mae is not None:
            sub += f" · ± {float(mae):,.1f} {unit} MAE"
        if r2 is not None:
            sub += f" · R² {float(r2):.2f}"
        items.append((nutrient_label(nutrient), f"{pred_100g:,.1f} {unit}", sub))

    if items:
        st.markdown(stat_strip(items, cols=len(items)), unsafe_allow_html=True)


DIARY_PATH = ROOT / "data" / "diary.csv"   # the data/ folder, not the dataset file (DATA)
DIARY_COLS = ["date", "meal", "dish", "servings"]


def _diary_entry_for(df, base: dict):
    """Rebuild one diary entry (nutrients recomputed from the dish table x servings)."""
    match = df[df["Dish Name"].astype(str) == str(base.get("dish"))]
    if match.empty:
        return None
    row = match.iloc[0]
    try:
        servings = float(base.get("servings"))
    except (TypeError, ValueError):
        return None
    entry = {"date": str(base.get("date"))[:10], "meal": str(base.get("meal")),
             "dish": str(base.get("dish")), "servings": servings}
    for nutrient in NUTRIENTS:
        value = pd.to_numeric(pd.Series([row.get(nutrient)]), errors="coerce").iloc[0]
        entry[nutrient] = float(value) * servings if pd.notna(value) else None
    return entry


def _diary_restore(df) -> list:
    """Load every saved entry from data/diary.csv (diary survives restarts)."""
    if not DIARY_PATH.exists():
        return []
    try:
        rows = pd.read_csv(DIARY_PATH)
    except Exception:
        return []
    entries = []
    for base in rows.to_dict("records"):
        entry = _diary_entry_for(df, base)
        if entry is not None:
            entries.append(entry)
    return entries


def _diary_write() -> None:
    """Persist all logged entries (every day, not just today) back to the CSV."""
    base_rows = [{k: it.get(k) for k in DIARY_COLS}
                 for it in st.session_state.get("daily_food_log", [])]
    pd.DataFrame(base_rows, columns=DIARY_COLS).to_csv(DIARY_PATH, index=False)


def page_daily_nutrition(df, theme: str):

    st.markdown(
        page_header(
            "My Daily Nutrition",
            "Track what you eat, estimate your nutrient intake, "
            "and discover foods that may help fill nutritional gaps."
        ),
        unsafe_allow_html=True
    )

    render_ml_nutrient_estimator(df)

    st.divider()

    today = pd.Timestamp.today().date().isoformat()

    if "daily_food_log" not in st.session_state:
        st.session_state["daily_food_log"] = _diary_restore(df)

    # Keep only today's entries for the current calculation
    today_logs = [
        item for item in st.session_state["daily_food_log"]
        if item["date"] == today
    ]

    # --------------------------------------------------------
    # 1. ADD FOOD
    # --------------------------------------------------------

    st.markdown(
        section_head(
            "Add food to your diary",
            "Enter each food you ate today and its serving quantity."
        ),
        unsafe_allow_html=True
    )

    with st.form("daily_nutrition_form"):

        meal = st.selectbox(
            "Meal",
            ["Breakfast", "Lunch", "Dinner", "Snacks"]
        )

        food_names = sorted(
            df["Dish Name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not food_names:
            st.warning("No food items were found in the nutrition dataset.")
            return

        dish = st.selectbox(
            "What did you eat?",
            food_names
        )

        quantity = st.number_input(
            "Portions (1 portion = 100 g)",
            min_value=0.25,
            max_value=20.0,
            value=1.0,
            step=0.25
        )

        st.caption(
            "All foods are standardized to 100 g per portion (e.g. 1.0 = 100 g, 2.5 = 250 g) "
            "using verified INDB nutritional density values."
        )

        submitted = st.form_submit_button(
            "Add to today's diary",
            type="primary"
        )

    if submitted:

        selected_row = df[
            df["Dish Name"].astype(str) == str(dish)
        ].iloc[0]

        entry = {
            "date": today,
            "meal": meal,
            "dish": str(dish),
            "servings": float(quantity)
        }

        # Calculate nutrients based on servings
        for nutrient in NUTRIENTS:

            value = pd.to_numeric(
                pd.Series([selected_row.get(nutrient)]),
                errors="coerce"
            ).iloc[0]

            if pd.notna(value):
                entry[nutrient] = float(value) * quantity
            else:
                entry[nutrient] = None

        st.session_state["daily_food_log"].append(entry)
        _diary_write()

        st.success(f"Added {dish} to {meal}!")

        st.rerun()

    # Refresh today's food entries
    today_logs = [
        item for item in st.session_state["daily_food_log"]
        if item["date"] == today
    ]

    # --------------------------------------------------------
    # 2. DISPLAY FOOD DIARY
    # --------------------------------------------------------

    _saved = st.session_state["daily_food_log"]
    _days = len({str(it["date"]) for it in _saved})
    if _saved:
        _diary_note = (f"Your food diary for today. Saved to data/diary.csv - "
                       f"{len(_saved)} entr{'y' if len(_saved) == 1 else 'ies'} "
                       f"from {_days} day{'s' if _days != 1 else ''} so far.")
    else:
        _diary_note = ("Your food diary for today. Entries are saved to "
                       "data/diary.csv, so they survive a refresh.")
    st.markdown(
        section_head("Today's meals", _diary_note),
        unsafe_allow_html=True
    )

    if not today_logs:
        st.info(
            "Your food diary is empty. Add the foods you have eaten "
            "to calculate your daily nutritional intake."
        )
        return

    log_df = pd.DataFrame(today_logs)

    display_log = log_df[
        ["meal", "dish", "servings"]
    ].rename(
        columns={
            "meal": "Meal",
            "dish": "Food",
            "servings": "Servings"
        }
    )

    st.dataframe(
        display_log,
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Remove last entry"):

            for i in range(
                len(st.session_state["daily_food_log"]) - 1,
                -1,
                -1
            ):
                if (
                    st.session_state["daily_food_log"][i]["date"]
                    == today
                ):
                    st.session_state["daily_food_log"].pop(i)
                    _diary_write()
                    break

            st.rerun()

    with col2:
        if st.button("Clear today's diary"):

            st.session_state["daily_food_log"] = [
                item
                for item in st.session_state["daily_food_log"]
                if item["date"] != today
            ]
            _diary_write()

            st.rerun()

    # --------------------------------------------------------
    # 3. CALCULATE TOTAL NUTRIENTS
    # --------------------------------------------------------

    st.markdown(
        section_head(
            "Estimated nutrition consumed",
            "Total nutrients from all the foods logged today."
        ),
        unsafe_allow_html=True
    )

    totals = {}

    for nutrient in NUTRIENTS:

        values = pd.to_numeric(
            log_df[nutrient],
            errors="coerce"
        )

        if values.notna().any():
            totals[nutrient] = float(values.sum())
        else:
            totals[nutrient] = None

    # Summary metrics
    metric_columns = st.columns(4)

    summary_nutrients = [
        "Calories (kcal)",
        "Protein (g)",
        "Fibre (g)",
        "Calcium (mg)"
    ]

    for column, nutrient in zip(
        metric_columns,
        summary_nutrients
    ):

        value = totals.get(nutrient)

        if value is None:
            display_value = "—"
        else:
            display_value = (
                f"{value:,.1f} {UNITS.get(nutrient, '')}"
            )

        column.metric(
            LABELS.get(nutrient, nutrient).capitalize(),
            display_value
        )

    # --------------------------------------------------------
    # 4. COMPARE WITH DAILY REFERENCE VALUES
    # --------------------------------------------------------

    st.markdown(
        section_head(
            "Daily nutritional analysis",
            "Compare your estimated intake with general adult reference values."
        ),
        unsafe_allow_html=True
    )

    analysis_rows = []
    low_nutrients = []
    high_nutrients = []

    for nutrient in NUTRIENTS:

        amount = totals.get(nutrient)

        if amount is None:
            continue

        # Skip nutrients without a configured reference value
        if nutrient not in DAILY_REFERENCES:
            continue

        reference, kind = DAILY_REFERENCES[nutrient]

        percentage = (amount / reference) * 100

        if kind == "limit":

            if amount <= reference:
                status = "Within reference limit"
                difference = (
                    f"{reference - amount:,.1f} "
                    f"{UNITS.get(nutrient, '')} remaining"
                )
            else:
                status = "Above reference limit"
                difference = (
                    f"{amount - reference:,.1f} "
                    f"{UNITS.get(nutrient, '')} above limit"
                )
                high_nutrients.append(nutrient)

        else:

            if percentage < 80:
                status = "Below reference"
                low_nutrients.append(nutrient)

            elif percentage < 100:
                status = "Getting close"

            else:
                status = "Reference reached"

            difference = (
                f"{max(0, reference - amount):,.1f} "
                f"{UNITS.get(nutrient, '')} to reference"
            )

        analysis_rows.append({
            "Nutrient": LABELS.get(
                nutrient, nutrient
            ).capitalize(),

            "Consumed": (
                f"{amount:,.1f} {UNITS.get(nutrient, '')}"
            ),

            "Daily Reference": (
                f"{reference:,.0f} "
                f"{UNITS.get(nutrient, '')}"
            ),

            "Progress": f"{percentage:.0f}%",

            "Status": status,

            "Difference": difference
        })

    if analysis_rows:

        st.dataframe(
            pd.DataFrame(analysis_rows),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "These are illustrative general-adult reference values, "
        "not personalized medical targets. Actual requirements vary. "
        "A single day's food log cannot diagnose a nutrient deficiency."
    )

    # --------------------------------------------------------
    # 5. RECOMMEND FOODS FOR POSSIBLE NUTRIENT GAPS
    # --------------------------------------------------------

    st.markdown(
        section_head(
            "What could you eat next?",
            "Find dishes that may help increase nutrients that are "
            "currently below their reference values."
        ),
        unsafe_allow_html=True
    )

    if low_nutrients:

        st.write(
            "Nutrients currently below the reference: "
            + ", ".join(
                LABELS.get(n, n)
                for n in low_nutrients
            )
        )

        recommendations = recommend_multi(
            df,
            low_nutrients,
            [],
            False,
            False,
            False,
            True,
            6
        )

        if recommendations.empty:

            st.info(
                "No matching dishes were found for these nutrients."
            )

        else:

            columns_to_show = [
                "Dish Name",
                *[
                    nutrient
                    for nutrient in low_nutrients
                    if nutrient in recommendations.columns
                ]
            ]

            if "dish_type" in recommendations.columns:
                columns_to_show.append("dish_type")

            recommendations = recommendations[
                columns_to_show
            ]

            recommendations = recommendations.rename(
                columns={"dish_type": "Dish Type"}
            )

            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These suggestions are based on the nutritional values "
                "available in your food dataset."
            )

    else:

        st.success(
            "No major gaps were identified against the configured "
            "reference values in your logged foods. This is not a "
            "medical assessment."
        )

    # --------------------------------------------------------
    # 6. WARN ABOUT VALUES ABOVE LIMITS
    # --------------------------------------------------------

    if high_nutrients:

        st.warning(
            "Your logged intake is above the illustrative reference "
            "limit for: "
            + ", ".join(
                LABELS.get(n, n)
                for n in high_nutrients
            )
            + ". Check your portions and the nutritional data."
        )


# ==========================================================================
# PAGE: REPORT READER (OCR) - read a doctor's nutrient report from PDF / JPG / PNG
# Needs:  pip install pymupdf rapidocr-onnxruntime pillow   (no external programs)
# ==========================================================================
INSTALL_CMD = "pip install pymupdf rapidocr-onnxruntime pillow"


class MissingDeps(Exception):
    pass


def _need(module: str, package: str):
    try:
        return importlib.import_module(module)
    except Exception as exc:  # pragma: no cover
        raise MissingDeps(package) from exc


def missing_ocr_packages() -> list[str]:
    out = []
    for module, package in (("fitz", "pymupdf"), ("rapidocr_onnxruntime", "rapidocr-onnxruntime"), ("PIL", "pillow")):
        try:
            importlib.import_module(module)
        except Exception:
            out.append(package)
    return out


@st.cache_resource(show_spinner=False)
def _ocr_engine():
    return _need("rapidocr_onnxruntime", "rapidocr-onnxruntime").RapidOCR()


def _boxes_to_lines(result) -> str:
    """Group OCR boxes into text lines (keeps table rows together)."""
    items = []
    for box, text, _score in result:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append([sum(ys) / len(ys), min(xs), max(ys) - min(ys), text])
    items.sort(key=lambda t: (t[0], t[1]))
    lines, cur, cur_y = [], [], None
    for yc, x, h, text in items:
        if cur and abs(yc - cur_y) > max(8.0, 0.6 * h):
            lines.append(cur)
            cur = []
        cur.append((x, text))
        cur_y = yc if len(cur) == 1 else (cur_y * (len(cur) - 1) + yc) / len(cur)
    if cur:
        lines.append(cur)
    return "\n".join("  ".join(t for _, t in sorted(line)) for line in lines)


def _ocr_image(img) -> str:
    import numpy as np
    img = img.convert("RGB")
    w, h = img.size
    target = min(max(w, 1400), 3000)
    if w != target:
        img = img.resize((target, max(1, int(h * target / w))))
    result, _ = _ocr_engine()(np.array(img)[:, :, ::-1])
    return _boxes_to_lines(result) if result else ""


@st.cache_data(show_spinner=False)
def extract_text(data: bytes, filename: str, max_pages: int = 8) -> str:
    """Text from a PDF (digital text first, OCR for scanned pages) or an image."""
    image_mod = _need("PIL.Image", "pillow")
    if filename.lower().endswith(".pdf"):
        fitz = _need("fitz", "pymupdf")
        parts = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in list(doc)[:max_pages]:
                text = page.get_text("text")
                if len(text.strip()) >= 60:
                    parts.append(text)
                else:  # scanned page -> render and OCR
                    pix = page.get_pixmap(dpi=200)
                    parts.append(_ocr_image(image_mod.open(io.BytesIO(pix.tobytes("png")))))
        return "\n".join(parts)
    return _ocr_image(image_mod.open(io.BytesIO(data)))


# analyte pattern, display name, NutriMatch nutrient it points to (None = not in the dataset)
ANALYTES = [
    (r"h(?:a)?emoglobin(?!\s*a1c)|\bhgb\b|\bhb\b(?!\s*a1c)", "Haemoglobin", "Iron (mg)"),
    (r"ferritin", "Ferritin", "Iron (mg)"),
    (r"(?:serum\s+|s\.?\s*)?\biron\b", "Serum iron", "Iron (mg)"),
    (r"calcium", "Calcium", "Calcium (mg)"),
    (r"vitamin\s*d\b|25\s*[-\s]?\(?oh\)?", "Vitamin D", None),
    (r"vitamin\s*b\s*-?\s*12|cobalamin", "Vitamin B12", None),
    (r"folate|folic\s*acid", "Folate", "Folate (ug)"),
    (r"vitamin\s*c\b|ascorbic\s*acid", "Vitamin C", "Vitamin C (mg)"),
    (r"sodium", "Sodium", "Sodium (mg)"),
    (r"total\s*protein", "Total protein", "Protein (g)"),
    (r"albumin", "Albumin", "Protein (g)"),
]
SKIP_WORDS = ("corpuscular", "mchc", "binding", "tibc", "saturation", "urine", "urinary", "microalbumin")
_NUM = r"(\d+(?:\.\d+)?)"
_RANGE = re.compile(_NUM + r"\s*(?:-|\u2013|\u2014|to)\s*" + _NUM, re.I)
_UNIT = re.compile(r"(g/dL|mg/dL|ng/mL|pg/mL|mmol/L|mEq/L|[\u00b5\u03bcu]g/dL|mcg/dL|IU/L|g/L|%)", re.I)
_LOW_WORDS = re.compile(r"\b(low|deficien\w*|insufficien\w*|decreased|below)\b", re.I)
_HIGH_WORDS = re.compile(r"\b(high|elevated|raised|increased|above)\b", re.I)
_LOW_MARK = re.compile(r"(?<![\w/])L(?![\w/])|\u2193")
_HIGH_MARK = re.compile(r"(?<![\w/])H(?![\w/])|\u2191")


def _status(line: str, value, low, high) -> str:
    if value is not None and low is not None and high is not None and low < high:
        if value < low:
            return "Low"
        if value > high:
            return "High"
        return "Normal"
    if _LOW_WORDS.search(line) or _LOW_MARK.search(line):
        return "Low"
    if _HIGH_WORDS.search(line) or _HIGH_MARK.search(line):
        return "High"
    return "Unknown"


def parse_report(text: str) -> list[dict]:
    """Find nutrient-related lab values in report text (first match per test)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    found, seen = [], set()
    for idx, line in enumerate(lines):
        lower = line.lower()
        if any(w in lower for w in SKIP_WORDS):
            continue
        for pattern, name, nutrient in ANALYTES:
            m = re.search(pattern, line, re.I)
            if not m or name in seen:
                continue
            rest = re.sub(r"^\s*(\([^)]*\)\s*)+", "", line[m.end():])
            nums = re.search(_NUM, rest)
            if not nums and idx + 1 < len(lines):      # value wrapped onto the next line
                line_used = line + " " + lines[idx + 1]
                rest = lines[idx + 1]
                nums = re.search(_NUM, rest)
            else:
                line_used = line
            value = float(nums.group(1)) if nums else None
            after = rest[nums.end():] if nums else rest
            rng = _RANGE.search(after)
            low = float(rng.group(1)) if rng else None
            high = float(rng.group(2)) if rng else None
            unit = _UNIT.search(rest)
            found.append({
                "analyte": name, "value": value, "unit": unit.group(1) if unit else "",
                "range": f"{rng.group(1)} - {rng.group(2)}" if rng else "",
                "status": _status(line_used, value, low, high), "nutrient": nutrient,
            })
            seen.add(name)
            break
    return found


SAMPLE_REPORT = """PATHOLOGY REPORT - DEMO
Patient: Demo Patient    Age/Sex: 29 / F
Test  Result  Unit  Reference Range
Haemoglobin  9.8  g/dL  12.0 - 15.5  LOW
Serum Ferritin  8  ng/mL  15 - 150  L
Serum Iron  45  ug/dL  60 - 170  Low
Calcium  9.4  mg/dL  8.6 - 10.2
Vitamin B12  180  pg/mL  211 - 911  LOW
Folate  3.1  ng/mL  3.0 - 17.0
Sodium  131  mmol/L  135 - 145  Low
Total Protein  6.9  g/dL  6.0 - 8.3
Vitamin D (25-OH)  14  ng/mL  30 - 100  Deficient
"""


def report_table_html(rows: list[dict]) -> str:
    head = ('<div class="rrow rhead"><span>Test</span><span>Result</span><span>Reference</span>'
            '<span>Status</span><span>Nutrient</span></div>')
    body = ""
    for r in rows:
        val = f'{r["value"]:g} {escape(r["unit"])}'.strip() if r["value"] is not None else "-"
        ref = escape(r["range"]) if r["range"] else "-"
        chip = {"Low": "chip-low", "High": "chip-high", "Normal": "chip-ok"}.get(r["status"], "chip-na")
        nut = escape(nutrient_label(r["nutrient"])) if r["nutrient"] else "Not tracked"
        body += (f'<div class="rrow"><span class="rname">{escape(r["analyte"])}</span><span>{val}</span>'
                 f'<span>{ref}</span><span><b class="chip {chip}">{r["status"]}</b></span>'
                 f'<span class="rmuted">{nut}</span></div>')
    return f'<div class="rtable">{head}{body}</div>'


def _use_sample() -> None:
    st.session_state["ocr_text"] = SAMPLE_REPORT
    st.session_state["ocr_hash"] = "sample"


def _use_pasted() -> None:
    st.session_state["ocr_text"] = st.session_state.get("paste_text", "")
    st.session_state["ocr_hash"] = "pasted"


def page_report(df, theme: str):
    st.markdown(page_header(
        "Read a nutrient report",
        "Upload your doctor's lab report as a PDF, JPG or PNG. NutriMatch reads it, finds the nutrients "
        "that are low, and suggests dishes to help."), unsafe_allow_html=True)

    with st.container():
        st.markdown(CARD, unsafe_allow_html=True)
        up = st.file_uploader("Upload a report", type=["pdf", "png", "jpg", "jpeg"],
                              label_visibility="collapsed", key="report_file")
        st.caption("Your file is read on this computer and is not stored or sent anywhere.")
        st.button("Try a sample report", on_click=_use_sample, key="use_sample")

    if up is not None:
        data = up.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if st.session_state.get("ocr_hash") != digest:
            try:
                with st.spinner("Reading your report..."):
                    text = extract_text(data, up.name)
                st.session_state["ocr_text"] = text
                st.session_state["ocr_hash"] = digest
            except MissingDeps as exc:
                st.markdown(callout("warning", f"Missing package: <code>pip install {escape(str(exc))}</code> "
                                               "then restart the app. You can also paste the report text below."),
                            unsafe_allow_html=True)
            except Exception as exc:
                st.markdown(callout("warning", f"Could not read this file ({escape(type(exc).__name__)}). "
                                               "Try a clearer scan, or paste the text below."),
                            unsafe_allow_html=True)

    with st.expander("Paste report text instead"):
        st.text_area("Report text", key="paste_text", height=140, label_visibility="collapsed",
                     placeholder="Paste the lab values here, one test per line...")
        st.button("Analyse pasted text", on_click=_use_pasted, key="use_pasted")

    text = st.session_state.get("ocr_text", "")
    if not text.strip():
        st.markdown(empty_state("search", "Your results will appear here",
                                "Upload a report above, paste its text, or try the sample report."),
                    unsafe_allow_html=True)
        return

    st.markdown(section_head("What we read", "Fix any misread values here and the results update."),
                unsafe_allow_html=True)
    with st.expander("Extracted text (editable)"):
        st.text_area("Extracted text", key="ocr_text", height=220, label_visibility="collapsed")
    text = st.session_state.get("ocr_text", "")

    rows = parse_report(text)
    if not rows:
        st.markdown(callout("warning", "<b>No nutrient values recognised.</b> Check the extracted text above, "
                                       "or make sure the report lists tests such as haemoglobin, iron, calcium, "
                                       "folate or sodium."), unsafe_allow_html=True)
        return

    n_low = sum(r["status"] == "Low" for r in rows)
    n_high = sum(r["status"] == "High" for r in rows)
    n_ok = sum(r["status"] == "Normal" for r in rows)
    st.markdown(stat_strip([
        ("Indian dishes", f"{len(df):,}", "standardized per 100 g values"),
        ("Nutrients tracked", f"{len(NUTRIENTS)}", "energy, macros, minerals, vitamins"),
        ("Vegetarian", f"{int((df['is_nonveg'] == 0).sum()):,}",
         f"{int(df['is_nonveg'].sum()):,} non-vegetarian"),
        ("Typical energy", f"{df['Calories (kcal)'].median():,.0f} kcal", "median dish, per 100 g"),
    ]), unsafe_allow_html=True)
    st.markdown(f'<div class="foot">Disclaimer: All nutritional values are standardized per 100 g across {len(df):,} Indian dishes.</div>',
                unsafe_allow_html=True)

    low_rows = [r for r in rows if r["status"] == "Low"]
    low_nuts = list(dict.fromkeys(r["nutrient"] for r in low_rows if r["nutrient"]))
    untracked = [r["analyte"] for r in low_rows if not r["nutrient"]]
    if untracked:
        st.markdown(callout("info", f"<b>{escape(', '.join(untracked))}</b> "
                                    f"{'is' if len(untracked) == 1 else 'are'} low, but NutriMatch's dish data does "
                                    "not include it. Ask your doctor about good sources."), unsafe_allow_html=True)

    st.markdown(section_head("Dishes for your low nutrients",
                             "Based on the values marked Low. Add or remove nutrients to explore."),
                unsafe_allow_html=True)
    sel_c, veg_c = st.columns([3, 1.2])
    chosen = sel_c.multiselect("Nutrients to target", NUTRIENTS, default=low_nuts, key="rep_sel",
                               format_func=nutrient_label, placeholder="Choose nutrients")
    veg = veg_c.checkbox("Vegetarian only", key="rep_veg")
    if len(chosen) > 1:
        st.button("Find dishes rich in all of these", type="primary", key="open_all",
                  on_click=open_in_predictor, args=(chosen,))
    if not chosen:
        st.markdown(callout("info", "No tracked nutrient is low in this report. Pick one above to see dishes anyway."),
                    unsafe_allow_html=True)
    for nut in chosen:
        rec = recommend(df, nut, [], veg, False, False, True, 4)
        head_c, btn_c = st.columns([3, 1.3])
        head_c.markdown(f"#### Rich in {LABELS.get(nut, nut)}")
        btn_c.button("Open in Predictor", key=f"open_{nut}", on_click=pick_nutrient, args=(nut,))
        if rec.empty:
            st.caption("No dishes match these filters.")
        else:
            st.markdown(dish_grid_html(df, rec, nut), unsafe_allow_html=True)
            st.write("")

    st.markdown(callout("warning", "<b>Please note:</b> automated reading can make mistakes, and this tool does not "
                                   "give medical advice. Check the values against your original report and follow "
                                   "your doctor's guidance."), unsafe_allow_html=True)


# ==========================================================================
# TOP BAR + MAIN
# ==========================================================================
def render_topbar() -> None:
    # keep the toggle in sync with the active theme (it used to show "off" while in Dark)
    st.session_state.setdefault("theme_toggle", st.session_state.get("theme") == "Dark")
    brand_c, nav_c, theme_c = st.columns([0.9, 5.0, 1.1], gap="small",
                                         vertical_alignment="center")
    with brand_c:
        st.markdown(brand_markup(), unsafe_allow_html=True)
    with nav_c:
        st.radio("Navigation", PAGES, horizontal=True, key="nav",
                 label_visibility="collapsed")
    with theme_c:
        st.toggle("Dark mode", key="theme_toggle", on_change=on_theme_toggle)


def main() -> None:
    theme = init_theme()
    inject_theme(theme)
    render_topbar()

    df, res = load_data()
    page = st.session_state.get("nav", "Home")
    if page == "Predictor":
        page_predictor(df, res, theme)
    elif page == "Report Reader":
        page_report(df, theme)
    elif page == "Graphs and Statistics":
        page_graphs(df, res, theme)
    elif page == "My Daily Nutrition":
        page_daily_nutrition(df, theme)
    else:
        page_home(df, theme)


if __name__ == "__main__":
    main()