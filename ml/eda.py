"""Exploratory analysis: where does each gateway succeed or struggle?

    python -m ml.eda                      # read from PostgreSQL
    python -m ml.eda --source csv         # read data/*.csv instead
    python -m ml.eda --out reports/eda

Prints the key tables, ranks each gateway's weakest segments by statistical significance, and
saves four charts in the project's dark theme. Segment analysis excludes the simulator's known
incident windows so that outages do not hide steady-state patterns.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from dashboards import theme  # noqa: E402

from .db import load  # noqa: E402
from .features import AMOUNT_BINS, TARGET  # noqa: E402

AMOUNT_LABELS = ["<100", "100-500", "500-1k", "1k-2.5k", "2.5k-5k", "5k-10k",
                 "10k-25k", "25k-50k", "50k+"]
SEGMENT_DIMS = [["method"], ["bank"], ["merchant_category"], ["hour"], ["amount_bucket"],
                ["method", "amount_bucket"]]


def prepare(attempts: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    df = attempts.copy()
    df["hour"] = df["ts"].dt.hour
    df["amount_bucket"] = pd.cut(df["amount_inr"], AMOUNT_BINS, labels=False, right=False).astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df["in_incident"] = incident_mask(df, outages)
    return df


def incident_mask(df: pd.DataFrame, outages: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for o in outages.itertuples():
        mask |= (df["gateway_id"] == o.gateway_id) & (df["ts"] >= o.start) & (df["ts"] < o.end)
    return mask


def weakest_segments(clean: pd.DataFrame, min_n: int = 300) -> pd.DataFrame:
    """Segments where a gateway does worse than its own average, ranked by z-score."""
    rows = []
    for gw, g in clean.groupby("gateway_id"):
        base = g[TARGET].mean()
        for dims in SEGMENT_DIMS:
            agg = g.groupby(dims)[TARGET].agg(["mean", "count"]).reset_index()
            agg = agg[agg["count"] >= min_n]
            for r in agg.to_dict("records"):
                z = (r["mean"] - base) / np.sqrt(base * (1 - base) / r["count"])
                rows.append({
                    "gateway_id": gw,
                    "segment": " & ".join(f"{d}={r[d]}" for d in dims),
                    "attempts": int(r["count"]),
                    "success_rate": round(r["mean"], 4),
                    "gateway_avg": round(base, 4),
                    "delta_pp": round(100 * (r["mean"] - base), 1),
                    "z": round(z, 1),
                })
    return pd.DataFrame(rows).sort_values(["gateway_id", "z"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- charts
def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": theme.BACKGROUND, "savefig.facecolor": theme.BACKGROUND,
        "axes.facecolor": theme.SURFACE, "axes.edgecolor": theme.BORDER,
        "axes.labelcolor": theme.TEXT, "text.color": theme.TEXT,
        "xtick.color": theme.MUTED, "ytick.color": theme.MUTED,
        "axes.grid": True, "grid.color": theme.BORDER, "grid.linewidth": 0.6,
        "legend.facecolor": theme.SURFACE, "legend.edgecolor": theme.BORDER,
    })


def _gw_color(gw: str) -> str:
    return theme.GATEWAY_COLORS.get(gw, theme.MUTED)


def chart_segment_heatmaps(clean: pd.DataFrame, path: Path) -> None:
    """Success rate relative to each gateway's own average, by method, bank and category."""
    base = clean.groupby("gateway_id")[TARGET].mean()
    cmap = LinearSegmentedColormap.from_list("rose_black_cerulean",
                                             [theme.ROSE, theme.SURFACE, theme.CERULEAN])
    cmap.set_bad(theme.BACKGROUND)
    dims = ["method", "bank", "merchant_category"]
    widths = [clean[d].nunique() for d in dims]
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.6), gridspec_kw={"width_ratios": widths})
    for ax, dim in zip(axes, dims):
        pivot = clean.pivot_table(index="gateway_id", columns=dim, values=TARGET, aggfunc="mean")
        delta = pivot.sub(base, axis=0)
        ax.grid(False)
        ax.imshow(np.ma.masked_invalid(delta.to_numpy()), cmap=cmap, vmin=-0.10, vmax=0.10,
                  aspect="auto")
        ax.set_xticks(range(delta.shape[1]), delta.columns, rotation=30, ha="right")
        ax.set_yticks(range(delta.shape[0]), delta.index)
        ax.set_title(f"by {dim.replace('_', ' ')}", loc="left", color=theme.TEXT)
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                v = delta.iat[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{100 * v:+.1f}", ha="center",
                        va="center", fontsize=9, color=theme.MUTED if np.isnan(v) else theme.TEXT)
    fig.suptitle("Success rate vs the gateway's own average (percentage points; rose = worse, "
                 "cerulean = better)", x=0.01, ha="left", color=theme.TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_by_hour(clean: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for gw, g in clean.groupby("gateway_id"):
        s = g.groupby("hour")[TARGET].mean()
        ax.plot(s.index, s.values, marker="o", ms=3, lw=1.8, color=_gw_color(gw), label=gw)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("success rate")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Success rate by hour of day", loc="left")
    ax.legend(ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_by_amount(clean: pd.DataFrame, path: Path, min_n: int = 100) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, (title, sub) in zip(axes, [("all methods", clean),
                                       ("card only", clean[clean["method"] == "card"])]):
        for gw, g in sub.groupby("gateway_id"):
            agg = g.groupby("amount_bucket")[TARGET].agg(["mean", "count"])
            agg = agg[agg["count"] >= min_n]
            ax.plot(agg.index, agg["mean"], marker="o", ms=3, lw=1.8, color=_gw_color(gw), label=gw)
        ax.set_xticks(range(len(AMOUNT_LABELS)), AMOUNT_LABELS, rotation=30, ha="right")
        ax.set_title(f"Success rate by amount ({title})", loc="left")
        ax.set_xlabel("amount (INR)")
    axes[0].set_ylabel("success rate")
    axes[0].legend(ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _hourly_rate(rows: pd.DataFrame, min_attempts: int) -> pd.Series:
    """Hourly success rate; hours with fewer than `min_attempts` are left blank (too noisy)."""
    hourly = rows.set_index("ts")[TARGET].resample("1h")
    return hourly.mean().where(hourly.count() >= min_attempts)


def chart_incidents(df: pd.DataFrame, outages: pd.DataFrame, path: Path,
                    min_attempts: int = 8) -> None:
    """Hourly success rate around each incident, with the incident window shaded.

    Overnight hours with very few attempts are omitted so a single failure does not read as 0%.
    """
    n = len(outages)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.4 * rows), squeeze=False)
    everyone = _hourly_rate(df, min_attempts)
    for ax, o in zip(axes.ravel(), outages.itertuples()):
        lo, hi = o.start - pd.Timedelta(hours=48), o.end + pd.Timedelta(hours=48)
        g = _hourly_rate(df[df["gateway_id"] == o.gateway_id], min_attempts)
        ax.plot(everyone.loc[lo:hi].index, everyone.loc[lo:hi].values, color=theme.MUTED,
                lw=1, ls="--", label="all gateways")
        ax.plot(g.loc[lo:hi].index, g.loc[lo:hi].values, color=_gw_color(o.gateway_id),
                lw=1.8, label=o.gateway_id)
        ax.axvspan(o.start, o.end, color=theme.ROSE, alpha=0.28, lw=0)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{o.gateway_id}: {o.kind}", loc="left")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="lower left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--data", default="data", help="CSV directory (for --source csv)")
    ap.add_argument("--out", default="reports/eda")
    ap.add_argument("--min-n", type=int, default=300, help="smallest segment to rank")
    args = ap.parse_args()

    attempts, _, outages = load(args.source, args.data)
    df = prepare(attempts, outages)
    clean = df[~df["in_incident"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{len(df):,} attempts, {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}; "
          f"{int(df['in_incident'].sum()):,} fall inside the {len(outages)} incident windows\n")

    overview = df.groupby("gateway_id").agg(
        attempts=(TARGET, "size"), success_rate=(TARGET, "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)), avg_fee_inr=("fee_inr", "mean"),
    ).round({"success_rate": 4, "avg_latency_ms": 0, "p95_latency_ms": 0, "avg_fee_inr": 2})
    print("Overview by gateway (all rows)")
    print(overview.to_string(), "\n")

    print("Incident impact")
    for o in outages.itertuples():
        win = df[(df["gateway_id"] == o.gateway_id) & (df["ts"] >= o.start) & (df["ts"] < o.end)]
        normal = clean[clean["gateway_id"] == o.gateway_id][TARGET].mean()
        print(f"  {o.gateway_id} {o.kind:<11} {o.start:%Y-%m-%d %H:%M} to {o.end:%H:%M}: "
              f"{win[TARGET].mean():.3f} success over {len(win)} attempts (normally {normal:.3f})")

    segments = weakest_segments(clean, args.min_n)
    segments.to_csv(out / "weak_segments.csv", index=False)
    print(f"\nWeakest segments per gateway (excluding incidents; z < -3 means unlikely to be noise)")
    for gw, g in segments.groupby("gateway_id"):
        top = g[g["z"] < -3].head(3)
        print(f"  {gw}:")
        if top.empty:
            print("    nothing clearly below its own average")
        for r in top.itertuples():
            print(f"    {r.segment:<36} {r.success_rate:.3f} vs {r.gateway_avg:.3f} "
                  f"({r.delta_pp:+.1f} pp, n={r.attempts:,}, z={r.z})")

    _style()
    chart_segment_heatmaps(clean, out / "segments_heatmap.png")
    chart_by_hour(clean, out / "success_by_hour.png")
    chart_by_amount(clean, out / "success_by_amount.png")
    chart_incidents(df, outages, out / "incidents.png")
    print(f"\nSaved weak_segments.csv and 4 charts to {out}/")


if __name__ == "__main__":
    main()
