"""Gateways, hidden patterns and outage windows for the simulator.

The patterns below are the "ground truth" the ML model must discover from data.
Each gateway has a different trade-off between reliability, speed and cost:

    gw_d  most reliable and fastest, but the most expensive
    gw_e  cheapest, but the least reliable and slowest
    gw_a / gw_b / gw_c  in between, each with a specific weakness
"""
from dataclasses import dataclass, field

import pandas as pd

# --- Transaction attribute spaces -------------------------------------------
METHODS = ["upi", "card", "netbanking"]
METHOD_PROBS = [0.55, 0.30, 0.15]

BANKS = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK", "OTHER"]
BANK_PROBS = [0.24, 0.22, 0.20, 0.14, 0.08, 0.12]

CATEGORIES = ["grocery", "food", "travel", "electronics", "utilities", "fashion"]
CATEGORY_PROBS = [0.22, 0.25, 0.12, 0.10, 0.18, 0.13]
CATEGORY_AMOUNT_SCALE = {
    "grocery": 0.8,
    "food": 0.5,
    "travel": 4.0,
    "electronics": 6.0,
    "utilities": 1.2,
    "fashion": 1.5,
}

MIN_AMOUNT_INR = 10.0
MAX_AMOUNT_INR = 200_000.0
UPI_LIMIT_INR = 100_000.0  # UPI payments above this are moved to cards

HIGH_VALUE_INR = 20_000.0  # threshold for the "high-value card" weakness
PEAK_HOURS = (19, 20, 21, 22)

# Relative traffic by hour of day (0-23) and by weekday (Mon-Sun)
HOUR_WEIGHTS = [
    0.5, 0.3, 0.2, 0.2, 0.3, 0.6, 1.2, 2.0, 3.0, 3.6, 4.0, 4.4,
    5.0, 5.0, 4.4, 4.0, 4.2, 4.8, 5.6, 6.4, 6.6, 6.0, 4.0, 1.8,
]
DOW_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.1, 1.25, 1.2]


@dataclass(frozen=True)
class Gateway:
    id: str
    name: str
    base_success: float          # success probability before any adjustment
    median_latency_ms: float
    fee_pct: float               # fraction of the amount, e.g. 0.02 = 2%
    fee_fixed: float             # flat INR fee per transaction
    methods: tuple               # payment methods this gateway supports
    method_adj: dict = field(default_factory=dict)      # additive, per method
    bank_adj: dict = field(default_factory=dict)        # additive, per bank
    category_adj: dict = field(default_factory=dict)    # additive, per category
    high_value_card_adj: float = 0.0                    # card and amount >= HIGH_VALUE_INR
    peak_adj: float = 0.0                               # during PEAK_HOURS
    peak_latency_mult: float = 1.0


GATEWAYS = [
    Gateway(
        id="gw_a", name="AlphaPay", base_success=0.955, median_latency_ms=380,
        fee_pct=0.020, fee_fixed=0.0, methods=("upi", "card", "netbanking"),
        high_value_card_adj=-0.15,           # weak on high-value cards
        peak_latency_mult=1.1,
    ),
    Gateway(
        id="gw_b", name="BetaPay", base_success=0.945, median_latency_ms=450,
        fee_pct=0.016, fee_fixed=0.0, methods=("upi", "card", "netbanking"),
        method_adj={"upi": 0.02, "netbanking": -0.02},
        peak_adj=-0.12,                      # degrades in the evening peak
        peak_latency_mult=1.5,
    ),
    Gateway(
        id="gw_c", name="CoreGate", base_success=0.940, median_latency_ms=520,
        fee_pct=0.018, fee_fixed=2.0, methods=("upi", "card", "netbanking"),
        method_adj={"netbanking": 0.03, "upi": -0.05},
        category_adj={"travel": -0.06},      # weak on travel merchants
    ),
    Gateway(
        id="gw_d", name="DeltaPG", base_success=0.975, median_latency_ms=260,
        fee_pct=0.024, fee_fixed=0.0, methods=("upi", "card", "netbanking"),
        method_adj={"card": 0.01},
        bank_adj={"SBI": -0.08},             # weak on one bank
    ),
    Gateway(
        id="gw_e", name="EchoPay", base_success=0.915, median_latency_ms=700,
        fee_pct=0.012, fee_fixed=0.0, methods=("upi", "card"),
        method_adj={"card": -0.03},
        peak_adj=-0.04, peak_latency_mult=1.2,
    ),
]
GATEWAYS_BY_ID = {g.id: g for g in GATEWAYS}


@dataclass(frozen=True)
class Outage:
    gateway_id: str
    start_frac: float        # which day: position in the simulated period, 0.0 to 1.0
    start_hour: int          # hour of that day the incident begins (0-23)
    duration_hours: int
    success_prob: float      # success probability while the outage is active
    kind: str                # "outage" (hard down) or "degradation"


# Two incidents fall in the first 80% (training) and two in the last 20%
# (test), so failover can be evaluated on data the model never saw.
OUTAGES = [
    Outage("gw_a", 0.30, 11, 3, 0.10, "outage"),
    Outage("gw_e", 0.55, 14, 6, 0.35, "degradation"),
    Outage("gw_d", 0.86, 18, 4, 0.05, "outage"),
    Outage("gw_b", 0.93, 12, 5, 0.30, "degradation"),
]


@dataclass(frozen=True)
class ResolvedOutage:
    gateway_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    success_prob: float
    kind: str


def resolve_outages(start: pd.Timestamp, days: int, outages=OUTAGES) -> list:
    """Turn fractional outage positions into concrete timestamps."""
    resolved = []
    for o in outages:
        day = min(int(o.start_frac * days), days - 1)
        s = start.normalize() + pd.Timedelta(days=day, hours=o.start_hour)
        resolved.append(
            ResolvedOutage(
                o.gateway_id, s, s + pd.Timedelta(hours=o.duration_hours),
                o.success_prob, o.kind,
            )
        )
    return resolved
