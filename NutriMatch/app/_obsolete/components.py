"""Reusable UI markup helpers for NutriMatch.

All helpers return single-line HTML strings themed through CSS variables
(see theme.py), so they work unchanged in Light and Dark mode. No data or
recommendation logic lives here.
"""

from __future__ import annotations

from html import escape


def _h(markup: str) -> str:
    """Collapse indented markup to one line (keeps Streamlit's markdown from
    treating indented HTML as a code block)."""
    return "".join(line.strip() for line in markup.splitlines())


def brand_markup() -> str:
    """Logo + wordmark for the top bar."""
    return _h("""
    <div class="brand"><span class="brand-mark"></span>NutriMatch</div>
    """)


def page_header(title: str, subtitle: str = "") -> str:
    """Consistent title block for inner pages."""
    sub = f'<p class="page-sub">{subtitle}</p>' if subtitle else ""
    return _h(f"""
    <div class="page-head"><h1 class="page-title">{title}</h1>{sub}</div>
    """)


def hero_markup(title: str, tagline: str, preview_title: str = "",
                preview_rows: list[tuple[str, str, str, float]] | None = None,
                preview_note: str = "") -> str:
    """Hero block for the Start page. The optional preview panel shows real
    ranked dishes from the dataset (name, value, unit, percent-of-max)."""
    panel = ""
    if preview_rows:
        rows = "".join(
            f'<div class="hp-row"><div class="hp-line"><span class="hp-name">{escape(n)}</span>'
            f'<span class="hp-val">{v} <span>{u}</span></span></div>'
            f'<div class="hp-bar"><div class="hp-fill" style="width:{max(4.0, min(100.0, p)):.0f}%"></div></div></div>'
            for n, v, u, p in preview_rows
        )
        note = f'<div class="hp-note">{preview_note}</div>' if preview_note else ""
        panel = f'<div class="hero-panel"><div class="hp-title">{preview_title}</div>{rows}{note}</div>'
    return _h(f"""
    <div class="hero">
      <div><h1 class="hero-title">{title}</h1><p class="hero-tag">{tagline}</p></div>
      {panel}
    </div>
    """)


def feature_card(index: str, title: str, text: str) -> str:
    """Step card; the number is real because the steps are a sequence."""
    return _h(f"""
    <div class="feature-card">
      <div class="fcard-num">{index}</div>
      <div class="fcard-title">{title}</div>
      <div class="fcard-text">{text}</div>
    </div>
    """)


def stat_strip(items: list[tuple[str, str, str]]) -> str:
    """One connected strip of headline stats: (label, value, sub) per item."""
    cells = "".join(
        f'<div class="stat-item"><div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        + (f'<div class="stat-sub">{sub}</div>' if sub else "")
        + "</div>"
        for label, value, sub in items
    )
    return f'<div class="stat-strip">{cells}</div>'


def callout(kind: str, text: str) -> str:
    """Styled callout. kind in {info, success, warning, danger}."""
    return f'<div class="callout callout-{kind}">{text}</div>'


def dish_card(rank: int, name: str, dish_type: str, value: str, unit: str,
              pct_of_max: float, percentile: int, veg: int) -> str:
    """Ranked dish card with rank badge, tags, value and progress bar.

    ``pct_of_max`` drives the progress bar (value relative to the top dish).
    ``percentile`` is the nutrient percentile (0-100) shown as text.
    """
    veg_cls = "tag-veg" if veg == 0 else "tag-nonveg"
    veg_lbl = "Vegetarian" if veg == 0 else "Non-veg"
    width = min(100.0, max(0.0, pct_of_max))
    top = " dcard-top" if rank == 1 else ""
    safe = escape(str(name))
    return _h(f"""
    <div class="dcard{top}">
      <div class="dhead"><div class="drank">{rank}</div><div class="dname" title="{safe}">{safe}</div></div>
      <div class="dtags"><span class="tag">{escape(str(dish_type))}</span><span class="tag {veg_cls}">{veg_lbl}</span></div>
      <div class="dvalue">{value} <span class="dunit">{unit}</span></div>
      <div class="dbar"><div class="dbarfill" style="width:{width:.1f}%"></div></div>
      <div class="dsub">Ahead of {percentile}% of dishes</div>
    </div>
    """)


def empty_state(title: str, steps: list[str], note: str) -> str:
    """Quiet empty state shown on the Predictor page before the first search."""
    lis = "".join(f"<li>{s}</li>" for s in steps)
    return _h(f"""
    <div class="empty-box">
      <div class="empty-title">{title}</div>
      <ul class="empty-steps">{lis}</ul>
      <div class="empty-note">{note}</div>
    </div>
    """)