"""Shared design tokens and styling for the payment-router dashboards.

Palette: jet black, cerulean, neon rose.
Rule of thumb: ~85% black, ~10% cerulean, ~5% rose. Rose means "look here".
"""

# --- Core palette -----------------------------------------------------------
BACKGROUND = "#0A0A0C"   # jet black
SURFACE = "#121216"      # cards, sidebar
ELEVATED = "#1A1A20"     # hover, inputs
BORDER = "#26262E"
CERULEAN = "#29A3E8"     # primary data, links, active nav
CERULEAN_DEEP = "#007BA7"
ROSE = "#FF2E93"         # accent: primary action, chosen gateway, key numbers
TEXT = "#F2F4F8"
MUTED = "#8A93A6"

# --- Status colors (always pair with an icon and a label) --------------------
STATUS = {
    "healthy": {"color": "#2EE6A6", "icon": "●", "label": "Healthy"},
    "degraded": {"color": "#FFB020", "icon": "▲", "label": "Degraded"},
    "down": {"color": ROSE, "icon": "✕", "label": "Down"},
}

# --- Gateway colors: fixed everywhere, rose deliberately excluded -----------
GATEWAY_COLORS = {
    "gw_a": "#29A3E8",  # cerulean
    "gw_b": "#9B7BFF",  # violet
    "gw_c": "#22D3C5",  # teal
    "gw_d": "#C9D1E0",  # silver
    "gw_e": "#5B6CFF",  # indigo
}

GRADIENT = f"linear-gradient(90deg, {CERULEAN}, {ROSE})"

FONT_HEADING = "Space Grotesk"
FONT_BODY = "Inter"
FONT_MONO = "JetBrains Mono"

# --- Plotly template ---------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=SURFACE,
    font=dict(family=f"{FONT_BODY}, sans-serif", color=TEXT),
    colorway=list(GATEWAY_COLORS.values()),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
    margin=dict(l=24, r=24, t=48, b=24),
)

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{ font-family: '{FONT_BODY}', sans-serif; }}
h1, h2, h3 {{ font-family: '{FONT_HEADING}', sans-serif; letter-spacing: -0.02em; }}
[data-testid="stMetricValue"], code {{ font-family: '{FONT_MONO}', monospace; }}

h1::after {{
    content: ""; display: block; height: 3px; width: 72px; margin-top: 8px;
    background: {GRADIENT}; border-radius: 2px;
}}

div[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 14px 16px;
}}

.stButton > button[kind="primary"] {{
    box-shadow: 0 0 24px rgba(255, 46, 147, 0.25);
}}
.stButton > button[kind="primary"]:hover {{
    background: {GRADIENT}; border-color: transparent;
}}
</style>
"""


def apply_theme() -> None:
    """Inject fonts and shared CSS. Call once at the top of every Streamlit page."""
    import streamlit as st

    st.markdown(_CSS, unsafe_allow_html=True)


def status_pill(state: str) -> str:
    """Return an HTML pill for a gateway state: 'healthy', 'degraded' or 'down'."""
    s = STATUS[state]
    return (
        f'<span style="color:{s["color"]};border:1px solid {s["color"]};'
        f'border-radius:999px;padding:2px 10px;font-size:0.8rem;">'
        f'{s["icon"]} {s["label"]}</span>'
    )
