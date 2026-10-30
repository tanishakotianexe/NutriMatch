"""Theme system for NutriMatch (modern, flat, teal + turmeric).

Light / Dark palettes are exposed as CSS variables so the whole app restyles
instantly. The theme lives in ``st.session_state["theme"]`` and is toggled from
the top bar. Pure presentation - no data or model logic here.
"""

from __future__ import annotations

import streamlit as st

PALETTES = {
    "Light": {
        "bg": "#f5f8f6",
        "surface": "#ffffff",
        "surface2": "#eef4f1",
        "primary": "#0b6b57",
        "primary_hover": "#095545",
        "primary_soft": "#e2f0ea",
        "accent": "#f2b01e",
        "accent_ink": "#2a2000",
        "text": "#10211c",
        "muted": "#5d726a",
        "border": "#dce6e1",
        "grid": "#e8efeb",
        "success": "#1f8a5b",
        "warning": "#b7791f",
        "info": "#2a6f97",
        "danger": "#c0452f",
        "shadow": "0 1px 2px rgba(16,33,28,.05)",
        "shadow_lg": "0 12px 32px rgba(16,33,28,.10)",
        "hero1": "#083c32",
        "hero2": "#0b6b57",
    },
    "Dark": {
        "bg": "#0b1210",
        "surface": "#121b18",
        "surface2": "#182420",
        "primary": "#6fc9a8",
        "primary_hover": "#8bd8bb",
        "primary_soft": "#173028",
        "accent": "#f2b84b",
        "accent_ink": "#2a2000",
        "text": "#e5eeea",
        "muted": "#8ea49b",
        "border": "#233029",
        "grid": "#1b2823",
        "success": "#7fd1a0",
        "warning": "#e0b25a",
        "info": "#7fb0cf",
        "danger": "#e09484",
        "shadow": "0 1px 2px rgba(0,0,0,.35)",
        "shadow_lg": "0 14px 36px rgba(0,0,0,.45)",
        "hero1": "#0e2d26",
        "hero2": "#15483b",
    },
}

# Used by Plotly charts (lists can't live in PALETTES, which is str-only)
COLORWAY = {
    "Light": ["#0b6b57", "#f2b01e", "#2a6f97", "#c0452f", "#7a9e8f", "#8a6d3b"],
    "Dark": ["#6fc9a8", "#f2b84b", "#7fb0cf", "#e09484", "#5c8a7a", "#c2a266"],
}
HEAT_SCALE = {
    "Light": ["#f5f8f6", "#bfe0d2", "#0b6b57"],
    "Dark": ["#121b18", "#1f5a4a", "#6fc9a8"],
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --bg: __BG__;
  --surface: __SURFACE__;
  --surface2: __SURFACE2__;
  --primary: __PRIMARY__;
  --primary-hover: __PRIMARY_HOVER__;
  --primary-soft: __PRIMARY_SOFT__;
  --accent: __ACCENT__;
  --accent-ink: __ACCENT_INK__;
  --text: __TEXT__;
  --muted: __MUTED__;
  --border: __BORDER__;
  --grid: __GRID__;
  --success: __SUCCESS__;
  --warning: __WARNING__;
  --info: __INFO__;
  --danger: __DANGER__;
  --shadow: __SHADOW__;
  --shadow-lg: __SHADOW_LG__;
  --hero1: __HERO1__;
  --hero2: __HERO2__;
  --display: 'Bricolage Grotesque', 'DM Sans', sans-serif;
}

* { box-sizing: border-box; }

.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
.stApp button, .stApp select, .stApp td, .stApp th {
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
/* keep Streamlit's icon font intact */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

[data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }

/* single-page website: no sidebar, navigation lives in the top bar */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container, [data-testid="stMain"] .block-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 1.1rem 2rem 5rem !important;
}

/* ---------------- typography ---------------- */
h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
  font-family: var(--display) !important;
  color: var(--text) !important;
  letter-spacing: -0.02em;
  font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] h2 { font-size: 1.55rem !important; margin-top: 2.4rem !important; padding: 0 !important; }
[data-testid="stMarkdownContainer"] h3 { font-size: 1.2rem !important; }
[data-testid="stMarkdownContainer"] h4 { font-size: 1.05rem !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text); line-height: 1.6; }
[data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--text) !important; font-weight: 500; font-size: .9rem; }
hr, [data-testid="stDivider"] hr { border-color: var(--border) !important; }

/* ---------------- top bar ---------------- */
[data-testid="stHorizontalBlock"]:has(.brand) { align-items: center; padding: .35rem 0; }
.brand {
  display: flex; align-items: center; gap: .55rem;
  font-family: var(--display); font-weight: 700; font-size: 1.3rem;
  letter-spacing: -0.03em; color: var(--text);
}
.brand-mark {
  width: 26px; height: 26px; border-radius: 9px 9px 9px 3px;
  background: var(--primary); position: relative; flex-shrink: 0;
}
.brand-mark::after {
  content: ""; position: absolute; right: -3px; top: -3px;
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--bg);
}
.topbar-rule { height: 1px; background: var(--border); margin: .5rem 0 1.6rem; }

/* ---------------- inputs ---------------- */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-baseweb="input"] {
  background-color: var(--surface) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background-color: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  min-height: 44px;
  transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover { border-color: var(--primary) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 18%, transparent) !important;
}
[data-baseweb="select"] input { color: var(--text) !important; }
[data-testid="stSelectbox"] span, [data-testid="stMultiSelect"] span { color: var(--text) !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background-color: var(--primary-soft) !important;
  color: var(--primary) !important;
  border-radius: 999px;
  font-weight: 500;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: var(--primary) !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px;
  box-shadow: var(--shadow-lg);
}
[data-baseweb="menu"] li, [role="option"] { color: var(--text) !important; }
[role="option"][aria-selected="true"], [role="option"]:hover { background: var(--primary-soft) !important; }

/* ---------------- buttons ---------------- */
.stApp button[kind], .stApp [data-testid^="stBaseButton"] {
  border-radius: 12px !important;
  font-weight: 600 !important;
  padding: .55rem 1.2rem;
  transition: background .15s ease, border-color .15s ease, transform .15s ease;
}
.stApp button[kind="primary"], .stApp [data-testid="stBaseButton-primary"] {
  background: var(--primary) !important;
  color: #ffffff !important;
  border: 1px solid var(--primary) !important;
  box-shadow: var(--shadow);
}
.stApp button[kind="primary"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover {
  background: var(--primary-hover) !important;
  border-color: var(--primary-hover) !important;
}
.stApp button[kind="primary"] p, .stApp [data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
.stApp button[kind="secondary"], .stApp [data-testid="stBaseButton-secondary"] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.stApp button[kind="secondary"]:hover, .stApp [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--primary) !important;
  color: var(--primary) !important;
}
[data-testid="stDownloadButton"] button { font-size: .88rem; padding: .4rem .9rem; }
/* the one turmeric call-to-action on the Start page */
.st-key-cta_start button {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  padding: .75rem 1.7rem !important;
  font-size: 1rem;
}
.st-key-cta_start button p { color: var(--accent-ink) !important; }
.st-key-cta_start button:hover { filter: brightness(.95); background: var(--accent) !important; border-color: var(--accent) !important; }

/* ---------------- navigation: segmented control ---------------- */
[data-testid="stRadio"] [role="radiogroup"] {
  display: inline-flex; flex-wrap: nowrap; gap: 2px;
  background: var(--primary-soft);
  padding: 4px; border-radius: 999px;
}
[data-testid="stRadio"] label {
  display: inline-flex; align-items: center;
  border-radius: 999px !important;
  padding: .38rem 1rem !important;
  margin: 0 !important;
  cursor: pointer;
  background: transparent !important;
  transition: background .15s ease, color .15s ease;
}
[data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] label p { color: var(--muted) !important; font-weight: 500; font-size: .88rem; white-space: nowrap; }
[data-testid="stRadio"] label:hover p { color: var(--text) !important; }
[data-testid="stRadio"] label:has(input:checked) { background: var(--surface) !important; box-shadow: var(--shadow); }
[data-testid="stRadio"] label:has(input:checked) p { color: var(--primary) !important; font-weight: 600; }

/* ---------------- toggle / slider / checkbox ---------------- */
[data-testid="stToggle"] label p, [data-testid="stCheckbox"] label p { color: var(--text) !important; font-size: .9rem; }
[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--primary) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--primary) !important; box-shadow: none !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: var(--primary); font-weight: 600; }
[data-testid="stCheckbox"] span[data-checked="true"] { background: var(--primary) !important; border-color: var(--primary) !important; }
[data-testid="stCheckbox"] span[data-checked="true"] svg { stroke: #ffffff !important; }
[data-testid="stToggle"] [data-checked="true"] { background-color: var(--primary) !important; }

/* ---------------- tabs ---------------- */
[data-testid="stTabs"] [role="tablist"] { gap: .4rem; }
[data-testid="stTabs"] button { color: var(--muted) !important; padding: .6rem .9rem; }
[data-testid="stTabs"] button p { font-weight: 500; }
[data-testid="stTabs"] button[aria-selected="true"], [data-testid="stTabs"] button[aria-selected="true"] p { color: var(--primary) !important; font-weight: 600; }
[data-baseweb="tab-highlight"] { background-color: var(--primary) !important; height: 3px !important; border-radius: 3px; }
[data-baseweb="tab-border"] { background-color: var(--border) !important; }

/* ---------------- containers ---------------- */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important; border-radius: 14px !important;
  background: var(--surface) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: 18px; }
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  background: var(--surface) !important;
  box-shadow: var(--shadow);
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
  box-shadow: none; border-radius: 14px !important;
}

/* ---------------- metrics ---------------- */
[data-testid="stMetric"] {
  background: var(--surface2) !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--display) !important; font-weight: 600 !important; }

/* ---------------- tables / alerts / images ---------------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--border) !important; border-radius: 14px !important;
  overflow: hidden; background: var(--surface) !important;
}
[data-testid="stAlert"] { border-radius: 14px !important; }
[data-testid="stImage"] img { border-radius: 16px; }
[data-testid="stToast"] { border-radius: 12px; }

/* ---------------- page header ---------------- */
.page-head { margin: .4rem 0 1.4rem; }
.page-title { font-family: var(--display); font-weight: 700; font-size: 2.2rem; letter-spacing: -0.03em; color: var(--text); margin: 0 0 .35rem; line-height: 1.1; }
.page-sub { color: var(--muted); font-size: 1.02rem; max-width: 42em; margin: 0; line-height: 1.55; }

/* ---------------- hero ---------------- */
.hero {
  position: relative; overflow: hidden;
  display: grid; grid-template-columns: 1.15fr .85fr; gap: 2.4rem; align-items: center;
  border-radius: 28px; padding: 3.6rem 3.2rem;
  color: #ffffff;
  background: linear-gradient(135deg, var(--hero1), var(--hero2));
}
.hero::before {
  content: ""; position: absolute; inset: 0;
  background-size: cover; background-position: center;
  opacity: .22; filter: blur(4px) saturate(.8); transform: scale(1.05);
}
.hero > * { position: relative; z-index: 1; }
.hero-title {
  font-family: var(--display) !important; font-weight: 700 !important;
  font-size: 3.1rem !important; line-height: 1.04 !important; letter-spacing: -0.035em !important;
  color: #ffffff !important; margin: 0 0 1.1rem !important; padding: 0 !important;
}
.hero-tag { color: rgba(255,255,255,.8) !important; font-size: 1.08rem !important; line-height: 1.6 !important; margin: 0 !important; max-width: 30em; }
.hero-panel {
  background: rgba(255,255,255,.96); color: #10211c;
  border-radius: 20px; padding: 1.3rem 1.4rem 1.2rem;
  box-shadow: 0 24px 48px rgba(0,0,0,.28);
}
.hp-title { font-family: var(--display); font-weight: 600; font-size: .98rem; color: #10211c; margin-bottom: .9rem; }
.hp-row { margin-bottom: .85rem; }
.hp-line { display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; font-size: .88rem; color: #10211c; font-weight: 500; }
.hp-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hp-val { font-family: var(--display); font-weight: 600; color: #0b6b57; white-space: nowrap; }
.hp-val span { color: #5d726a; font-weight: 500; font-size: .75rem; }
.hp-bar { height: 8px; border-radius: 999px; background: #e8efeb; margin-top: .4rem; overflow: hidden; }
.hp-fill { height: 100%; border-radius: 999px; background: #0b6b57; }
.hp-row:first-of-type .hp-fill { background: #f2b01e; }
.hp-note { color: #5d726a; font-size: .74rem; margin-top: .2rem; }

/* ---------------- feature (steps) ---------------- */
.feature-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; padding: 1.5rem 1.4rem; height: 100%;
}
.fcard-num {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: var(--primary-soft); color: var(--primary);
  font-family: var(--display); font-weight: 700; font-size: .95rem; margin-bottom: 1rem;
}
.fcard-title { font-family: var(--display); font-weight: 600; color: var(--text); margin-bottom: .4rem; font-size: 1.12rem; letter-spacing: -0.01em; }
.fcard-text { color: var(--muted); font-size: .92rem; line-height: 1.6; }

/* ---------------- stat strip ---------------- */
.stat-strip {
  display: grid; grid-template-columns: repeat(4, 1fr);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; overflow: hidden;
}
.stat-item { padding: 1.4rem 1.5rem; border-left: 1px solid var(--border); }
.stat-item:first-child { border-left: none; }
.stat-value { font-family: var(--display); font-weight: 700; font-size: 2rem; letter-spacing: -0.03em; color: var(--text); line-height: 1.1; }
.stat-label { color: var(--text); font-size: .92rem; font-weight: 500; margin-top: .35rem; }
.stat-sub { color: var(--muted); font-size: .8rem; margin-top: .1rem; }

/* ---------------- callouts ---------------- */
.callout {
  border-radius: 14px; padding: 14px 18px; margin: 10px 0;
  font-size: .92rem; line-height: 1.6; color: var(--text);
  background: var(--surface); border: 1px solid var(--border);
}
.callout-success { background: color-mix(in srgb, var(--success) 9%, var(--surface)); border-color: color-mix(in srgb, var(--success) 30%, var(--border)); }
.callout-info    { background: color-mix(in srgb, var(--info) 8%, var(--surface));    border-color: color-mix(in srgb, var(--info) 28%, var(--border)); }
.callout-warning { background: color-mix(in srgb, var(--warning) 11%, var(--surface)); border-color: color-mix(in srgb, var(--warning) 35%, var(--border)); }
.callout-danger  { background: color-mix(in srgb, var(--danger) 9%, var(--surface));  border-color: color-mix(in srgb, var(--danger) 30%, var(--border)); }

/* ---------------- panel titles (predictor form) ---------------- */
.panel-title { font-family: var(--display); font-weight: 600; font-size: 1.05rem; color: var(--text); margin: .1rem 0 .6rem; letter-spacing: -0.01em; }
.panel-sub { color: var(--muted); font-size: .84rem; margin: -.4rem 0 .8rem; }

/* ---------------- dish cards ---------------- */
.dcard {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 18px; padding: 1.1rem 1.15rem; height: 100%;
  transition: border-color .15s ease;
}
.dcard:hover { border-color: var(--primary); }
.dcard-top { border-color: var(--accent); }
.dhead { display: flex; gap: .7rem; align-items: flex-start; margin-bottom: .55rem; }
.drank {
  flex-shrink: 0; min-width: 28px; height: 28px; border-radius: 9px; padding: 0 .4rem;
  background: var(--primary-soft); color: var(--primary);
  font-family: var(--display); font-weight: 700; font-size: .85rem;
  display: flex; align-items: center; justify-content: center;
}
.dcard-top .drank { background: var(--accent); color: var(--accent-ink); }
.dname { font-weight: 600; color: var(--text); font-size: .98rem; line-height: 1.3; padding-top: 3px; overflow-wrap: anywhere; }
.dtags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: .9rem; }
.tag { font-size: .72rem; padding: 3px 10px; border-radius: 999px; background: var(--surface2); color: var(--muted); font-weight: 500; }
.tag-veg { color: var(--success); background: color-mix(in srgb, var(--success) 12%, var(--surface)); }
.tag-nonveg { color: var(--danger); background: color-mix(in srgb, var(--danger) 11%, var(--surface)); }
.dvalue { font-family: var(--display); font-weight: 700; font-size: 1.7rem; letter-spacing: -0.02em; color: var(--text); line-height: 1.1; }
.dunit { font-size: .82rem; font-weight: 500; color: var(--muted); letter-spacing: 0; }
.dbar { height: 6px; border-radius: 999px; background: var(--grid); margin: .7rem 0 .45rem; overflow: hidden; }
.dbarfill { height: 100%; border-radius: 999px; background: var(--primary); }
.dsub { color: var(--muted); font-size: .78rem; }

/* ---------------- empty state ---------------- */
.empty-box {
  padding: 2.6rem 1.8rem; text-align: left;
  border: 1.5px dashed var(--border); border-radius: 20px; background: var(--surface);
}
.empty-title { font-family: var(--display); font-weight: 600; font-size: 1.35rem; letter-spacing: -0.02em; color: var(--text); margin-bottom: .9rem; }
.empty-steps { list-style: none; padding: 0; margin: 0 0 1.1rem; color: var(--text); font-size: .95rem; }
.empty-steps li { margin: .55rem 0; padding-left: 0; color: var(--muted); }
.empty-note { color: var(--muted); font-size: .82rem; border-top: 1px solid var(--border); padding-top: .9rem; }

/* ---------------- focus + responsive ---------------- */
:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; border-radius: 6px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 900px) {
  .hero { grid-template-columns: 1fr; padding: 2.4rem 1.6rem; }
  .hero-title { font-size: 2.3rem !important; }
  .stat-strip { grid-template-columns: repeat(2, 1fr); }
  .stat-item:nth-child(3) { border-left: none; }
  .stat-item:nth-child(n+3) { border-top: 1px solid var(--border); }
}
@media (max-width: 768px) {
  .block-container, [data-testid="stMain"] .block-container { padding: 1rem 1rem 3rem !important; }
  .page-title { font-size: 1.8rem; }
}
</style>
"""


def init_theme() -> str:
    """Return the active theme, defaulting to the configured Streamlit base."""
    if "theme" not in st.session_state:
        base = "light"
        try:
            base = str(st.get_option("theme.base") or "light").lower()
        except Exception:
            base = "light"
        st.session_state["theme"] = "Dark" if base == "dark" else "Light"
    return st.session_state["theme"]


def on_theme_toggle() -> None:
    """Callback for the Dark-mode toggle."""
    st.session_state["theme"] = "Dark" if st.session_state.get("theme_toggle", False) else "Light"


def theme_toggle_widget() -> None:
    """Render the Light/Dark toggle."""
    st.toggle(
        "Dark mode",
        value=(st.session_state.get("theme", "Light") == "Dark"),
        key="theme_toggle",
        on_change=on_theme_toggle,
    )


def inject_theme(theme: str) -> None:
    """Render the CSS for the active theme (Light or Dark)."""
    palette = PALETTES[theme]
    css = _CSS
    for key, value in palette.items():
        css = css.replace("__" + key.upper() + "__", value)
    st.markdown(css, unsafe_allow_html=True)


def apply_plotly_theme(fig, theme: str, height: int | None = None,
                       xlabel: str | None = None, ylabel: str | None = None,
                       title: str | None = None):
    """Apply the active theme to a Plotly figure (fonts, colours, gridlines)."""
    palette = PALETTES[theme]
    dark = theme == "Dark"
    base_layout = dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, 'Segoe UI', sans-serif", color=palette["text"], size=13),
        title_font=dict(family="Bricolage Grotesque, DM Sans, sans-serif", color=palette["text"], size=17),
        title_x=0,
        colorway=COLORWAY[theme],
        margin=dict(t=56 if title else 48, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(bgcolor=palette["surface"], font_color=palette["text"],
                        bordercolor=palette["border"]),
        bargap=0.12,
    )
    if height:
        base_layout["height"] = height
    if title:
        base_layout["title"] = title
    fig.update_layout(**base_layout)
    fig.update_xaxes(
        gridcolor=palette["grid"], zerolinecolor=palette["border"],
        linecolor=palette["border"], tickfont=dict(color=palette["muted"]),
        title_font=dict(color=palette["muted"]),
    )
    fig.update_yaxes(
        gridcolor=palette["grid"], zerolinecolor=palette["border"],
        linecolor=palette["border"], tickfont=dict(color=palette["muted"]),
        title_font=dict(color=palette["muted"]),
    )
    if xlabel is not None:
        fig.update_xaxes(title_text=xlabel)
    if ylabel is not None:
        fig.update_yaxes(title_text=ylabel)
    return fig