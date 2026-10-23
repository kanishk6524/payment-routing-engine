"""Evaluate routing policies on the held-out test period by counterfactual replay.

    python -m ml.evaluate                   # PostgreSQL data, model at ml/artifacts/model.joblib
    python -m ml.evaluate --source csv

Policies compared on the same test transactions:
    random           uniformly random supported gateway (the logged behaviour)
    cheapest         lowest fee for the transaction
    best overall     the gateway with the highest training-period success rate
    best per method  the best gateway per payment method (training period)
    ML               argmax predicted P(success)
    ML cost-aware    argmax P(success) - cost and latency penalties (weights tuned on validation)
    oracle           argmax true P(success): an upper bound, not achievable

Writes reports/results.json and reports/policy_comparison.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dashboards import theme  # noqa: E402

from .db import load  # noqa: E402
from .eda import _style  # noqa: E402
from .features import TARGET, build_features  # noqa: E402
from .model import ARTIFACT_PATH, load_bundle  # noqa: E402
from .replay import (  # noqa: E402
    TXN_COLUMNS, add_ground_truth, add_model_scores, baseline_scores, build_candidates, choose,
    incident_time_mask, ips_success, ml_score, outages_from_df, policy_metrics, tune_weights,
)

POLICY_LABELS = {
    "random": "random (logged)", "cheapest": "cheapest", "best_overall": "best overall",
    "best_per_method": "best per method", "ml": "ML", "ml_cost_aware": "ML cost-aware",
    "oracle": "oracle (ceiling)",
}


def evaluate_policies(test_cands, train, lam_cost, lam_latency, outages, logged):
    scores = baseline_scores(test_cands, train)
    scores["ml"] = ml_score(test_cands)
    scores["ml_cost_aware"] = ml_score(test_cands, lam_cost, lam_latency)
    scores["oracle"] = test_cands["p_true"]

    chosen = {"random": None}
    chosen.update({name: choose(test_cands, s) for name, s in scores.items()})
    results = {}
    for name, rows in chosen.items():
        if rows is None:  # expectation over a uniformly random supported gateway
            res = {"all": policy_metrics(test_cands, 1.0 / test_cands["n_candidates"])}
            for tag, inc in (("incident", True), ("normal", False)):
                m = incident_time_mask(test_cands["ts"], outages) == inc
                res[tag] = policy_metrics(test_cands[m], 1.0 / test_cands.loc[m, "n_candidates"])
            res["ips_success"], res["ips_se"] = float(logged[TARGET].mean()), None
            res["mix"] = (test_cands.assign(w=1.0 / test_cands["n_candidates"])
                          .groupby("gateway_id")["w"].sum() / test_cands["txn_id"].nunique()).to_dict()
        else:
            res = {"all": policy_metrics(rows)}
            inc = incident_time_mask(rows["ts"], outages)
            res["incident"] = policy_metrics(rows[inc])
            res["normal"] = policy_metrics(rows[~inc])
            res["ips_success"], res["ips_se"] = ips_success(rows, logged)
            res["mix"] = rows["gateway_id"].value_counts(normalize=True).to_dict()
        results[name] = res
    return results


def print_report(results, lam_cost, lam_latency, grid, model_label):
    print(f"\nPolicy comparison on the test period ({model_label}; "
          f"cost weight {lam_cost}, latency weight {lam_latency})")
    print(f"{'policy':<18} {'success':>8} {'IPS check':>15} {'value-wtd':>10} {'eff. fee':>9} "
          f"{'latency':>8} {'in incident':>12}")
    for name, r in results.items():
        a = r["all"]
        ips = f"{100 * r['ips_success']:.2f}" + (f" ±{200 * r['ips_se']:.2f}" if r["ips_se"] else "")
        print(f"{POLICY_LABELS[name]:<18} {100 * a['success_rate']:>7.2f}% {ips:>15} "
              f"{100 * a['value_success_rate']:>9.2f}% {a['effective_fee_pct']:>8.3f}% "
              f"{a['avg_latency_ms']:>6.0f}ms {100 * r['incident']['success_rate']:>11.2f}%")
    print("  success = expected success rate by replay; IPS check = same estimated from logged "
          "outcomes only (±95% interval); eff. fee = fees / volume processed successfully")

    print("\nTraffic share by gateway")
    gws = sorted({g for r in results.values() for g in r["mix"]})
    print(f"{'policy':<18}" + "".join(f"{g:>8}" for g in gws))
    for name, r in results.items():
        print(f"{POLICY_LABELS[name]:<18}" + "".join(f"{100 * r['mix'].get(g, 0):>7.1f}%" for g in gws))

    print("\nWeight tuning on the validation period (predicted metrics; lower index is better)")
    show = grid[grid["lambda_latency"].isin([0.0, lam_latency])].copy()
    show = show[["lambda_cost", "lambda_latency", "success_rate", "effective_fee_pct",
                 "avg_latency_ms", "cost_latency_index"]].round(4)
    print(show.to_string(index=False))


def headline(results) -> list:
    ml, ca = results["ml"]["all"], results["ml_cost_aware"]["all"]
    lines = []
    for base in ("cheapest", "best_overall", "best_per_method"):
        b = results[base]["all"]
        lines.append(
            f"vs {POLICY_LABELS[base]}: success {100 * ml['success_rate']:.2f}% vs "
            f"{100 * b['success_rate']:.2f}% ({100 * (ml['success_rate'] - b['success_rate']):+.2f} pp); "
            f"cost-aware router fee {ca['effective_fee_pct']:.3f}% vs {b['effective_fee_pct']:.3f}% "
            f"({100 * (ca['effective_fee_pct'] / b['effective_fee_pct'] - 1):+.1f}%), success "
            f"{100 * ca['success_rate']:.2f}%"
        )
        bi, mi = results[base]["incident"], results["ml"]["incident"]
        lines.append(f"    during incidents: {100 * mi['success_rate']:.1f}% vs {100 * bi['success_rate']:.1f}%")
    return lines


def chart(results, path: Path) -> None:
    order = ["random", "cheapest", "best_overall", "best_per_method", "ml", "ml_cost_aware", "oracle"]
    colors = {"ml": theme.CERULEAN, "ml_cost_aware": theme.ROSE, "oracle": theme.TEXT}
    panels = [("Success rate (test period)", lambda r: 100 * r["all"]["success_rate"], "%"),
              ("Success rate during incidents", lambda r: 100 * r["incident"]["success_rate"], "%"),
              ("Effective fee on processed volume", lambda r: r["all"]["effective_fee_pct"], "%")]
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), sharey=True)
    for ax, (title, fn, unit) in zip(axes, panels):
        for i, name in enumerate(order):
            v = fn(results[name])
            ax.scatter(v, i, s=70, color=colors.get(name, theme.MUTED),
                       edgecolor=theme.BACKGROUND, zorder=3)
            ax.text(v, i - 0.32, f"{v:.2f}{unit}", ha="center", fontsize=8, color=theme.TEXT)
        ax.set_title(title, loc="left", color=theme.TEXT)
        ax.set_yticks(range(len(order)), [POLICY_LABELS[n] for n in order])
        ax.invert_yaxis()
        ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model-path", default=ARTIFACT_PATH)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--tolerance-pp", type=float, default=0.1,
                    help="success the cost-aware router may give up, in percentage points")
    args = ap.parse_args()

    attempts, gateways, outage_df = load(args.source, args.data)
    outages = outages_from_df(outage_df)
    bundle = load_bundle(args.model_path)
    df = build_features(attempts, gateways)
    train = df[df["split"] == "train"]

    def candidates(split):
        part = df[df["split"] == split]
        return part, add_model_scores(build_candidates(part[TXN_COLUMNS], gateways, attempts), bundle)

    _, valid_c = candidates("valid")
    test_part, test_c = candidates("test")
    test_c = add_ground_truth(test_c, outages)

    lam_cost, lam_latency, grid = tune_weights(valid_c, args.tolerance_pp / 100)
    logged = test_part[["txn_id", "gateway_id", TARGET]]
    results = evaluate_policies(test_c, train, lam_cost, lam_latency, outages, logged)

    label = {"xgboost": "XGBoost", "hgb": "scikit-learn stand-in"}.get(bundle["kind"], bundle["kind"])
    print(f"{len(test_part):,} test transactions, {len(test_c):,} candidate (transaction, gateway) pairs")
    print_report(results, lam_cost, lam_latency, grid, label)
    print("\nHeadline")
    lines = headline(results)
    print("\n".join(lines))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": bundle["version"], "lambda_cost": lam_cost, "lambda_latency": lam_latency,
               "tolerance_pp": args.tolerance_pp, "test_transactions": len(test_part),
               "policies": results, "headline": lines}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    chart(results, out / "policy_comparison.png")
    print(f"\nSaved results.json and policy_comparison.png to {out}/")


if __name__ == "__main__":
    main()
