"""Live routing dashboard (Stage 10).

    streamlit run dashboards/app.py

Needs the API running (`uvicorn api.main:app`); Redis and PostgreSQL are optional and each
panel says plainly when its source is unreachable. Aggregation lives in dashboards/data.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboards import data, theme  # noqa: E402
from simulator.config import BANKS, CATEGORIES, METHODS  # noqa: E402

st.set_page_config(page_title="Payment Routing Engine", page_icon="◆", layout="wide")
theme.apply_theme()

GATEWAYS = list(theme.GATEWAY_COLORS)
STATE_COLOR = {**{k: v["color"] for k, v in theme.STATUS.items()}, "unknown": theme.MUTED}


def pill(state: str) -> str:
    if state in theme.STATUS:
        return theme.status_pill(state)
    return (f'<span style="color:{theme.MUTED};border:1px solid {theme.MUTED};border-radius:999px;'
            f'padding:2px 10px;font-size:0.8rem;">○ No data</span>')


def style(fig, height=320, **extra):
    fig.update_layout(**{**theme.PLOTLY_LAYOUT, "height": height, **extra})
    return fig


def notice(msg, err):
    st.warning(f"{msg}  \n`{err}`")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Payment Routing")
    st.caption("ML routing with SHAP explanations, circuit breaker and live failover.")
    auto = st.toggle("Auto-refresh (5 s)", value=False)
    if st.button("Refresh now"):
        st.rerun()
    st.caption(f"API: `{data.API_URL}`")

st.title("Intelligent payment routing")

health, health_err = data.api_get("/health", timeout=3)
if health_err and data.is_waking(health_err):
    with st.spinner("Waking the routing API (free hosting sleeps when idle). This takes up to a minute..."):
        health, health_err = data.wait_for_api()
if health_err:
    notice("The routing API is not reachable. Start it with `uvicorn api.main:app`.", health_err)

tab_live, tab_dec, tab_route, tab_model = st.tabs(["Live gateways", "Decisions", "Route a payment", "Model & policies"])


# ------------------------------------------------------------------ live
@st.fragment(run_every=5 if auto else None)
def live():
    gws, err = data.api_get("/gateways")
    if err:
        notice("No live gateway data.", err)
        return
    if health:
        c = st.columns(4)
        c[0].metric("Model", health["model_version"])
        c[1].metric("Health store", health["health_store"])
        c[2].metric("Event bus", health["event_bus"])
        c[3].metric("Uptime", f"{int(health['uptime_s'])} s")
    cards = data.gateway_cards(gws)
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(f"**{card['gateway_id']}**")
            st.markdown(pill(card["state"]), unsafe_allow_html=True)
            st.metric("Success, 15 min", "–" if card["success_pct"] is None else f"{card['success_pct']}%")
            lat = card["avg_latency_ms"]
            st.caption(f"{card['attempts']} attempts · {'–' if lat is None else f'{lat:.0f} ms'} · breaker {card['breaker']}")
            if not card["enabled"]:
                st.caption("Disabled by kill switch")

    series, ev_err = data.load_series(window_min=60, bucket_s=60)
    if ev_err:
        notice("Time series are unavailable (neither the API nor Redis answered).", ev_err)
        return
    if series.empty:
        st.info("No recent events yet. Start traffic with `python -m pipeline.stream --incident gw_d`.")
        return
    left, right = st.columns(2)
    for col, field, title, unit in ((left, "success_pct", "Success rate", "%"),
                                    (right, "avg_latency_ms", "Average latency", " ms")):
        fig = go.Figure()
        for gw in GATEWAYS:
            s = series[series["gateway_id"] == gw].sort_values("bucket")
            if not s.empty:
                fig.add_trace(go.Scatter(x=s["bucket"], y=s[field], name=gw, mode="lines",
                                         line=dict(color=theme.GATEWAY_COLORS[gw], width=2),
                                         hovertemplate=f"{gw}: %{{y:.1f}}{unit}<extra></extra>"))
        style(fig, title=title, hovermode="x unified", legend=dict(orientation="h"))
        if field == "success_pct":
            fig.update_yaxes(range=[0, 101])
        col.plotly_chart(fig, width="stretch")


with tab_live:
    live()


# ------------------------------------------------------------------ decisions
with tab_dec:
    dec, dec_err = data.load_decisions(500)
    if dec_err:
        notice("PostgreSQL is not reachable, so decision history is unavailable.", dec_err)
    elif dec.empty:
        st.info("No live decisions stored yet. Run the pipeline demo: `python -m scripts.pipeline_demo`.")
    else:
        share = data.traffic_share(dec)
        fig = go.Figure()
        for gw in GATEWAYS:
            s = share[share["gateway"] == gw].sort_values("bucket")
            if not s.empty:
                fig.add_trace(go.Bar(x=s["bucket"], y=s["share_pct"], name=gw, marker_color=theme.GATEWAY_COLORS[gw]))
        style(fig, title="Where traffic was routed (share per minute)", barmode="stack", legend=dict(orientation="h"))
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Why a payment went where it did")
        view = dec.head(100).copy()
        view["label"] = (view["decided_at"].dt.strftime("%H:%M:%S") + "  ·  ₹" + view["amount_inr"].map("{:,.0f}".format)
                         + "  ·  " + view["method"] + "  ·  → " + view["chosen_gateway"])
        pick = st.selectbox("Decision", view.index, format_func=lambda i: view.loc[i, "label"])
        row = view.loc[pick]
        expl = row["explanation"] if isinstance(row["explanation"], dict) else {}
        st.markdown(f"**{expl.get('summary', 'No summary stored.')}**")
        a, b = st.columns(2)
        sc = data.scores_frame(row["scores"], row["chosen_gateway"])
        fig = go.Figure(go.Bar(x=sc["score"], y=sc["gateway"], orientation="h",
                               marker_color=[theme.ROSE if c else theme.CERULEAN for c in sc["chosen"]]))
        a.plotly_chart(style(fig, 260, title="Score per gateway (chosen in rose)"), width="stretch")
        dr = data.drivers_frame(expl)
        if dr.empty:
            b.caption("No SHAP drivers were stored for this decision.")
        else:
            fig = go.Figure(go.Bar(x=dr["effect"], y=dr["label"] + ": " + dr["value"].astype(str), orientation="h",
                                   marker_color=[theme.CERULEAN if e > 0 else theme.ROSE for e in dr["effect"]]))
            b.plotly_chart(style(fig, 260, title="SHAP drivers (log-odds; cerulean raises, rose lowers)"),
                           width="stretch")
        with st.expander("Recent decisions"):
            st.dataframe(view[["decided_at", "txn_id", "amount_inr", "method", "bank", "chosen_gateway", "policy"]],
                         width="stretch", hide_index=True)


# ------------------------------------------------------------------ route
with tab_route:
    st.markdown("Send a payment to the live router and see the choice, the alternatives and the reasons.")
    with st.form("route"):
        c = st.columns(4)
        amount = c[0].number_input("Amount (INR)", 1.0, 1_000_000.0, 2500.0, step=100.0)
        method = c[1].selectbox("Method", METHODS)
        bank = c[2].selectbox("Bank", BANKS)
        cat = c[3].selectbox("Merchant category", CATEGORIES)
        go_btn = st.form_submit_button("Route this payment", type="primary")
    if go_btn:
        res, err = data.api_post("/route", {"amount_inr": amount, "method": method, "bank": bank, "merchant_category": cat})
        if err:
            notice("Routing failed.", err)
        else:
            m = st.columns(4)
            m[0].metric("Chosen gateway", res["gateway"])
            m[1].metric("P(success)", f"{100 * res['predicted_success']:.1f}%")
            m[2].metric("Fee", f"₹{res['fee_inr']:.2f}")
            m[3].metric("Decision time", f"{res['decision_ms']:.1f} ms")
            st.markdown(f"**{res['summary']}**  ·  mode `{res['mode']}`")
            if res.get("tripped_gateways"):
                st.warning(f"Circuit breaker open for: {', '.join(res['tripped_gateways'])}")
            alts = data.alternatives_frame(res["alternatives"], res["gateway"])
            st.dataframe(alts, width="stretch", hide_index=True)
            dr = data.drivers_frame({"drivers": res["drivers"]})
            if not dr.empty:
                st.dataframe(dr, width="stretch", hide_index=True)


# ------------------------------------------------------------------ model
with tab_model:
    results = data.load_results()
    if results is None:
        st.info("Run `python -m ml.evaluate` to generate reports/results.json.")
    else:
        pf = data.policy_frame(results)
        st.caption(f"Counterfactual replay on {results['test_transactions']:,} held-out transactions "
                   f"(simulated data). Model {results['model']}.")
        for line in results.get("headline", [])[:2]:
            st.markdown(f"- {line}")
        l, r = st.columns(2)
        for col, field, title in ((l, "success_pct", "Success rate (%)"), (r, "incident_pct", "Success during incidents (%)")):
            colors = [theme.ROSE if k == "ml" else theme.CERULEAN if k == "ml_cost_aware" else theme.MUTED for k in pf["key"]]
            fig = go.Figure(go.Bar(x=pf[field], y=pf["policy"], orientation="h", marker_color=colors,
                                   text=pf[field].map("{:.2f}".format), textposition="outside"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[max(0, pf[field].min() - 5), 100.5])
            col.plotly_chart(style(fig, 340, title=title), width="stretch")
        st.dataframe(pf.drop(columns="key").rename(columns={
            "success_pct": "Success %", "incident_pct": "Incident success %", "fee_pct": "Effective fee %",
            "latency_ms": "Latency ms"}), width="stretch", hide_index=True)
    metrics = data.load_metrics()
    if metrics:
        with st.expander("Training metrics"):
            st.json(metrics)
