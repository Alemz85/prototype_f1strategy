"""
validate.py — Compare projection accuracy: with vs without tire degradation.

For each lap 1–70, compute projected standings using both methods and measure
Mean Absolute Error against actual final positions.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from projection import get_race_state, project_standings


def main():
    base = os.path.join(os.path.dirname(__file__), "data", "processed")
    laps_df = pd.read_csv(os.path.join(base, "hungary_2024_laps.csv"))
    meta_df = pd.read_csv(os.path.join(base, "hungary_2024_race_meta.csv"))
    with open(os.path.join(base, "hungary_2024_pit_loss.json")) as f:
        pit_data = json.load(f)
    PIT_LOSS_MS = pit_data["estimated_pit_loss_ms"]
    TOTAL_LAPS = int(laps_df["lap"].max())

    # Actual final positions
    actual_positions = meta_df.set_index("driver_code")["final_position"].to_dict()

    results = []

    for lap in range(1, TOTAL_LAPS + 1):
        state = get_race_state(laps_df, meta_df, lap)

        # --- Without degradation ---
        proj_no_deg = project_standings(state, meta_df, PIT_LOSS_MS)
        active_no = proj_no_deg[~proj_no_deg["retired"]].copy()
        active_no["actual_final"] = active_no["driver_code"].map(actual_positions)
        active_no["error"] = (active_no["projected_position"] - active_no["actual_final"]).abs()
        mae_no_deg = active_no["error"].mean()

        # --- With degradation ---
        proj_deg = project_standings(
            state, meta_df, PIT_LOSS_MS,
            laps_df=laps_df, current_lap=lap, total_race_laps=TOTAL_LAPS,
        )
        active_deg = proj_deg[~proj_deg["retired"]].copy()
        active_deg["actual_final"] = active_deg["driver_code"].map(actual_positions)
        active_deg["error"] = (active_deg["projected_position"] - active_deg["actual_final"]).abs()
        mae_deg = active_deg["error"].mean()

        results.append({
            "lap": lap,
            "mae_no_deg": mae_no_deg,
            "mae_with_deg": mae_deg,
        })

    df = pd.DataFrame(results)

    # ── Summary table ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  LAP-BY-LAP MAE COMPARISON")
    print("=" * 60)
    print(f"{'Lap':>4}  {'No Deg MAE':>11}  {'With Deg MAE':>13}  {'Diff':>8}")
    print("-" * 42)
    for _, row in df.iterrows():
        diff = row["mae_no_deg"] - row["mae_with_deg"]
        marker = " ✓" if diff > 0 else ""
        print(
            f"{int(row['lap']):4d}  "
            f"{row['mae_no_deg']:11.3f}  "
            f"{row['mae_with_deg']:13.3f}  "
            f"{diff:+8.3f}{marker}"
        )

    # ── Overall stats ───────────────────────────────────────────────────────
    overall_no = df["mae_no_deg"].mean()
    overall_deg = df["mae_with_deg"].mean()
    mid_race = df[(df["lap"] >= 20) & (df["lap"] <= 50)]
    mid_no = mid_race["mae_no_deg"].mean()
    mid_deg = mid_race["mae_with_deg"].mean()

    print("\n" + "=" * 60)
    print("  OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Overall MAE (all laps):   No Deg = {overall_no:.3f}  |  With Deg = {overall_deg:.3f}")
    print(f"  Mid-race MAE (laps 20-50): No Deg = {mid_no:.3f}  |  With Deg = {mid_deg:.3f}")

    improvement = overall_no - overall_deg
    mid_improvement = mid_no - mid_deg
    if improvement > 0:
        print(f"\n  ✅ Degradation model is MORE accurate overall by {improvement:.3f} MAE ({improvement/overall_no*100:.1f}%)")
    else:
        print(f"\n  ❌ Degradation model is LESS accurate overall by {abs(improvement):.3f} MAE ({abs(improvement)/overall_no*100:.1f}%)")

    if mid_improvement > 0:
        print(f"  ✅ Mid-race improvement: {mid_improvement:.3f} MAE ({mid_improvement/mid_no*100:.1f}%)")
    else:
        print(f"  ❌ Mid-race degradation: {abs(mid_improvement):.3f} MAE ({abs(mid_improvement)/mid_no*100:.1f}%)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["lap"], df["mae_no_deg"], color="#E10600", linewidth=2,
            label=f"Without Degradation (avg {overall_no:.2f})", alpha=0.9)
    ax.plot(df["lap"], df["mae_with_deg"], color="#00D2BE", linewidth=2,
            label=f"With Degradation (avg {overall_deg:.2f})", alpha=0.9)

    # Highlight mid-race zone
    ax.axvspan(20, 50, alpha=0.08, color="#FFD700", label="Mid-race zone (laps 20–50)")

    ax.set_xlabel("Lap", fontsize=12, fontfamily="sans-serif")
    ax.set_ylabel("Mean Absolute Error (positions)", fontsize=12, fontfamily="sans-serif")
    ax.set_title("Projection Accuracy: With vs Without Tire Degradation Model",
                 fontsize=14, fontweight="bold", fontfamily="sans-serif")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(1, TOTAL_LAPS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Dark theme
    fig.patch.set_facecolor("#15151E")
    ax.set_facecolor("#1E1E2E")
    ax.tick_params(colors="#ccc")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#fff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.legend(fontsize=10, loc="upper right", facecolor="#1E1E2E",
              edgecolor="#444", labelcolor="#ccc")

    out_path = os.path.join(os.path.dirname(__file__), "validation_mae_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
