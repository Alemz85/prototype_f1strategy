"""
projection.py — F1 Race Strategy Projection Engine

Pure-function module for the Streamlit race strategy simulator.
All functions receive DataFrames/dicts and return DataFrames/lists — no file I/O.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Function 1: get_race_state
# ─────────────────────────────────────────────────────────────────────────────

def get_race_state(
    laps_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    current_lap: int,
) -> pd.DataFrame:
    """
    Return the state of the race at the end of *current_lap*.

    For each driver: current position, cumulative time, stops done,
    stint info, last lap time.  Drivers who retired before current_lap
    are flagged with ``retired = True``.

    Parameters
    ----------
    laps_df : DataFrame
        The full ``hungary_2024_laps.csv`` data.
    meta_df : DataFrame
        The ``hungary_2024_race_meta.csv`` data (one row per driver).
    current_lap : int
        The lap number to snapshot.

    Returns
    -------
    DataFrame with columns:
        driver_code, driver_name, constructor,
        current_position, cumulative_time_ms, last_lap_time_ms,
        stops_done, stint_number, lap_in_stint,
        grid_position, final_position, retired
    """
    # Get each driver's last available lap at or before current_lap
    available = laps_df[laps_df["lap"] <= current_lap]

    # For each driver, take the row at their maximum lap (handles retirements)
    idx = available.groupby("driver_code")["lap"].idxmax()
    state = available.loc[idx].copy()

    # Determine retirement: a driver is retired if their last lap is before
    # current_lap AND they didn't finish the race (total_laps < race length).
    total_laps_map = meta_df.set_index("driver_code")["total_laps"]
    status_map = meta_df.set_index("driver_code")["status"]

    state["total_laps"] = state["driver_code"].map(total_laps_map)
    state["status"] = state["driver_code"].map(status_map)
    state["retired"] = (state["lap"] < current_lap) & (state["status"] != "Finished")

    # Build clean output
    result = state.rename(columns={
        "position": "current_position",
        "lap_time_ms": "last_lap_time_ms",
        "total_stops_so_far": "stops_done",
    })[[
        "driver_code", "driver_name", "constructor",
        "current_position", "cumulative_time_ms", "last_lap_time_ms",
        "stops_done", "stint_number", "lap_in_stint",
        "grid_position", "final_position", "retired",
    ]].copy()

    return result.sort_values("current_position").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Function 1b: estimate_degradation_rate
# ─────────────────────────────────────────────────────────────────────────────

def estimate_degradation_rate(
    laps_df: pd.DataFrame,
    driver_code: str,
    current_lap: int,
    default_rate: float = 50.0,
) -> float:
    """
    Estimate how much lap time (ms) a driver is losing per lap due to tire
    wear in their current stint.

    Logic:
      1. Collect all laps in the driver's current stint (after last pit).
      2. Exclude first 2 laps (out-lap / warm-up).
      3. Exclude outliers > 5 s slower than the median.
      4. Fit linear regression: lap_in_stint → lap_time_ms; slope = deg rate.
      5. If fewer than 4 clean laps, return *default_rate*.

    Returns
    -------
    float
        Degradation rate in ms per lap (positive = getting slower).
    """
    drv = laps_df[(laps_df["driver_code"] == driver_code) & (laps_df["lap"] <= current_lap)]
    if drv.empty:
        return default_rate

    # Find laps in current stint (after last pit stop)
    pit_laps = drv[drv["is_pit_lap"] == True]["lap"]  # noqa: E712
    if pit_laps.empty:
        stint_start = 1
    else:
        stint_start = int(pit_laps.max()) + 1  # lap after the last pit

    stint = drv[drv["lap"] >= stint_start].copy()
    stint["lap_in_stint"] = stint["lap"] - stint_start + 1

    # Exclude first 2 laps of stint (out-lap + warm-up)
    stint = stint[stint["lap_in_stint"] > 2]

    if len(stint) < 4:
        return default_rate

    # Exclude outliers: laps > 5 s slower than median
    median_time = stint["lap_time_ms"].median()
    clean = stint[stint["lap_time_ms"] <= median_time + 5000]

    if len(clean) < 4:
        return default_rate

    # Linear regression: lap_in_stint → lap_time_ms
    slope = np.polyfit(clean["lap_in_stint"].values, clean["lap_time_ms"].values, 1)[0]

    # Degradation should be positive (getting slower); clamp to >= 0
    return max(slope, 0.0)


def _triangular_cost(deg_rate: float, n_laps: int) -> float:
    """Cumulative degradation cost over *n_laps* laps (triangular sum)."""
    if n_laps <= 0:
        return 0.0
    return deg_rate * n_laps * (n_laps + 1) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Function 2: project_standings
# ─────────────────────────────────────────────────────────────────────────────

def project_standings(
    race_state: pd.DataFrame,
    meta_df: pd.DataFrame,
    pit_loss_ms: float,
    expected_stops_override: Optional[dict] = None,
    laps_df: Optional[pd.DataFrame] = None,
    current_lap: Optional[int] = None,
    total_race_laps: int = 70,
) -> pd.DataFrame:
    """
    Project final standings based on remaining pit stops and (optionally)
    tire degradation.

    For each active driver:
      stops_remaining = expected_total_stops − stops_done  (floor 0)
      projected_time  = cumulative_time + stops_remaining × pit_loss
                        − degradation_savings

    When *laps_df* is provided, a degradation model is applied:
      • Estimate each driver's current deg rate (ms/lap).
      • Compute cumulative degradation cost if they don't stop vs.
        splitting remaining laps across shorter stints.
      • Subtract the savings from projected_time.

    Parameters
    ----------
    race_state, meta_df, pit_loss_ms, expected_stops_override :
        Same as before.
    laps_df : DataFrame, optional
        Full lap-level data (enables degradation model).
    current_lap : int, optional
        Current lap number (required when laps_df is provided).
    total_race_laps : int
        Total laps in the race (default 70).

    Returns
    -------
    DataFrame with columns:
        driver_code, driver_name, constructor,
        current_position, projected_position, position_delta,
        stops_done, stops_remaining, degradation_rate_ms,
        cumulative_time_ms, projected_time_ms,
        projected_gap_to_leader_ms, retired
    """
    df = race_state.copy()

    # Map expected total stops from meta, allowing user overrides
    expected_map = meta_df.set_index("driver_code")["expected_total_stops"].to_dict()
    if expected_stops_override:
        expected_map.update(expected_stops_override)

    df["expected_total_stops"] = df["driver_code"].map(expected_map).fillna(0).astype(int)
    df["stops_remaining"] = (df["expected_total_stops"] - df["stops_done"]).clip(lower=0)

    # ── Degradation model ───────────────────────────────────────────────────
    use_deg = laps_df is not None and current_lap is not None
    remaining_laps = (total_race_laps - current_lap) if use_deg else 0

    deg_rates = {}   # driver_code → deg_rate_ms_per_lap
    deg_costs = {}   # driver_code → total_degradation_cost_ms

    if use_deg and remaining_laps > 0:
        for _, row in df[~df["retired"]].iterrows():
            code = row["driver_code"]
            deg = estimate_degradation_rate(laps_df, code, current_lap)
            deg_rates[code] = deg

            stops_rem = int(row["stops_remaining"])

            # Split remaining laps into expected stints.
            # (stops_remaining + 1) stints: current stint + one after each stop.
            # Each stop resets degradation to zero (fresh tires).
            n_stints = stops_rem + 1
            base_stint_len = remaining_laps // n_stints
            remainder = remaining_laps % n_stints
            total_cost = 0.0
            for s in range(n_stints):
                stint_len = base_stint_len + (1 if s < remainder else 0)
                total_cost += _triangular_cost(deg, stint_len)

            deg_costs[code] = total_cost

    df["degradation_rate_ms"] = df["driver_code"].map(deg_rates).fillna(0.0)

    # Projected time: cumulative + pit cost + degradation cost
    df["projected_time_ms"] = (
        df["cumulative_time_ms"]
        + df["stops_remaining"] * pit_loss_ms
        + df["driver_code"].map(deg_costs).fillna(0.0)
    )

    # Separate active vs retired for ranking
    active = df[~df["retired"]].copy()
    retired = df[df["retired"]].copy()

    # Rank active drivers by projected time
    active = active.sort_values("projected_time_ms").reset_index(drop=True)
    active["projected_position"] = active.index + 1

    # Retired drivers go to the bottom, keep their current position order
    retired = retired.sort_values("current_position").reset_index(drop=True)
    retired["projected_position"] = (
        len(active) + retired.index + 1
    )

    df = pd.concat([active, retired], ignore_index=True)

    # Position delta: positive = gaining positions (moving up)
    df["position_delta"] = df["current_position"] - df["projected_position"]

    # Gap to projected leader
    if not active.empty:
        leader_time = active["projected_time_ms"].iloc[0]
        df["projected_gap_to_leader_ms"] = df["projected_time_ms"] - leader_time
    else:
        df["projected_gap_to_leader_ms"] = 0.0

    return df[[
        "driver_code", "driver_name", "constructor",
        "current_position", "projected_position", "position_delta",
        "stops_done", "stops_remaining", "degradation_rate_ms",
        "cumulative_time_ms", "projected_time_ms",
        "projected_gap_to_leader_ms", "retired",
    ]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Function 3: generate_commentary
# ─────────────────────────────────────────────────────────────────────────────

def generate_commentary(
    projected_standings: pd.DataFrame,
    pit_loss_ms: float,
    remaining_laps: int = 0,
) -> list[str]:
    """
    Generate up to 6 natural-language insight strings from projected standings.

    Insight types (prioritising top-10 drivers):
      • Position change alerts
      • Misleading leaders (P1 on track still needs to stop)
      • Undercut / overcut narratives
      • Close battles (projected gap < 3 s)
      • Degradation alerts (high deg + recoverable time)

    Returns
    -------
    list[str]
        Max 5–6 insight strings with emoji markers.
    """
    insights: list[str] = []
    active = projected_standings[~projected_standings["retired"]].copy()
    top10 = active[active["current_position"] <= 10]

    # ── 1. Misleading leader ────────────────────────────────────────────────
    track_leader = active[active["current_position"] == 1]
    if not track_leader.empty:
        leader = track_leader.iloc[0]
        if leader["stops_remaining"] > 0 and leader["projected_position"] > 1:
            proj_pos = int(leader["projected_position"])
            insights.append(
                f"⚠️ {leader['driver_name']} leads on track but still has "
                f"{int(leader['stops_remaining'])} stop(s) to make — "
                f"projected to drop to P{proj_pos}."
            )

    # ── 2. Position change alerts (top-10 current, big movers) ──────────────
    for _, row in top10.iterrows():
        delta = int(row["position_delta"])
        cur = int(row["current_position"])
        proj = int(row["projected_position"])
        name = row["driver_name"]
        if delta >= 2:
            insights.append(
                f"🔺 {name} is currently P{cur} but projected P{proj} "
                f"once all stops are done (+{delta} places)."
            )
        elif delta <= -2:
            insights.append(
                f"🔻 {name} is currently P{cur} but projected to drop to "
                f"P{proj} ({delta} places)."
            )
        if len(insights) >= 5:
            break

    # ── 3. Undercut / overcut narratives (top-10) ───────────────────────────
    if len(insights) < 5:
        pitted = top10[top10["stops_remaining"] == 0].sort_values("projected_position")
        not_pitted = top10[top10["stops_remaining"] > 0]
        for _, p in pitted.iterrows():
            for _, np_row in not_pitted.iterrows():
                # Driver who pitted is projected ahead of driver who hasn't
                if (p["projected_position"] < np_row["projected_position"]
                        and p["current_position"] > np_row["current_position"]):
                    insights.append(
                        f"🔄 {p['driver_name']} (P{int(p['current_position'])}) "
                        f"has already pitted and is projected P{int(p['projected_position'])}, "
                        f"ahead of {np_row['driver_name']} "
                        f"(P{int(np_row['current_position'])}) who still needs "
                        f"{int(np_row['stops_remaining'])} stop(s)."
                    )
                    if len(insights) >= 5:
                        break
            if len(insights) >= 5:
                break

    # ── 4. Close battles (projected gap < 3 s) ─────────────────────────────
    if len(insights) < 5:
        sorted_active = active.sort_values("projected_time_ms").reset_index(drop=True)
        for i in range(min(len(sorted_active) - 1, 10)):
            gap = (sorted_active.iloc[i + 1]["projected_time_ms"]
                   - sorted_active.iloc[i]["projected_time_ms"])
            if gap < 3000:  # less than 3 seconds
                d1 = sorted_active.iloc[i]
                d2 = sorted_active.iloc[i + 1]
                insights.append(
                    f"⚔️ Tight battle: {d1['driver_name']} and "
                    f"{d2['driver_name']} are projected just "
                    f"{gap / 1000:.1f}s apart "
                    f"(P{int(d1['projected_position'])}–"
                    f"P{int(d2['projected_position'])})."
                )
                if len(insights) >= 5:
                    break

    # ── 5. Degradation alerts ───────────────────────────────────────────────
    if len(insights) < 6 and "degradation_rate_ms" in active.columns and remaining_laps > 0:
        for _, row in top10.sort_values("degradation_rate_ms", ascending=False).iterrows():
            deg = row["degradation_rate_ms"]
            if deg > 200:  # losing > 0.2 s/lap
                # Estimate time recoverable by one extra stop
                cost_no_stop = _triangular_cost(deg, remaining_laps)
                stint_len = remaining_laps // 2
                cost_with_stop = (
                    _triangular_cost(deg, stint_len)
                    + _triangular_cost(deg, remaining_laps - stint_len)
                )
                recoverable = cost_no_stop - cost_with_stop
                if recoverable > 5000:  # > 5 s
                    insights.append(
                        f"⚠️ {row['driver_name']} is losing "
                        f"{deg / 1000:.1f}s/lap on worn tires — "
                        f"an extra stop could recover "
                        f"{recoverable / 1000:.1f}s over remaining laps."
                    )
                    break

    return insights[:6]


# ─────────────────────────────────────────────────────────────────────────────
# Function 4: generate_safety_car_scenario
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_CAR_PIT_LOSS_MS = 8000  # ~8 seconds under SC

def generate_safety_car_scenario(
    race_state: pd.DataFrame,
    meta_df: pd.DataFrame,
    pit_loss_ms: float,
    expected_stops_override: Optional[dict] = None,
    laps_df: Optional[pd.DataFrame] = None,
    current_lap: Optional[int] = None,
    total_race_laps: int = 70,
) -> tuple[pd.DataFrame, str]:
    """
    Simulate a safety car: re-project standings using reduced pit loss
    (~8 s instead of the normal value) because pitting under SC is much
    cheaper.

    Parameters
    ----------
    race_state, meta_df, pit_loss_ms, expected_stops_override :
        Same as ``project_standings``.
    laps_df, current_lap, total_race_laps :
        Passed through to ``project_standings`` for degradation model.

    Returns
    -------
    (projected_df, commentary_str)
        projected_df : same format as ``project_standings``
        commentary_str : explanation of the key changes under SC
    """
    deg_kwargs = dict(laps_df=laps_df, current_lap=current_lap, total_race_laps=total_race_laps)

    # Normal projection (for comparison)
    normal = project_standings(race_state, meta_df, pit_loss_ms, expected_stops_override, **deg_kwargs)

    # Safety-car projection
    sc = project_standings(race_state, meta_df, SAFETY_CAR_PIT_LOSS_MS, expected_stops_override, **deg_kwargs)

    # Build commentary by comparing position changes
    merged = normal[["driver_code", "projected_position"]].merge(
        sc[["driver_code", "projected_position"]],
        on="driver_code",
        suffixes=("_normal", "_sc"),
    )
    merged["sc_delta"] = merged["projected_position_normal"] - merged["projected_position_sc"]
    gainers = merged[merged["sc_delta"] > 0].sort_values("sc_delta", ascending=False)
    losers = merged[merged["sc_delta"] < 0].sort_values("sc_delta")

    lines = [
        f"🟡 Safety Car scenario: pit loss drops from "
        f"{pit_loss_ms / 1000:.1f}s to {SAFETY_CAR_PIT_LOSS_MS / 1000:.1f}s."
    ]

    # Drivers who still need to stop benefit most
    sc_active = sc[~sc["retired"]]
    beneficiaries = sc_active[sc_active["stops_remaining"] > 0]
    if not beneficiaries.empty:
        names = ", ".join(beneficiaries["driver_name"].tolist()[:4])
        lines.append(
            f"Drivers with stops remaining ({names}) benefit most — "
            f"each remaining stop costs {(pit_loss_ms - SAFETY_CAR_PIT_LOSS_MS) / 1000:.1f}s less."
        )

    if not gainers.empty:
        top = gainers.iloc[0]
        lines.append(
            f"Biggest winner: {top['driver_code']} gains "
            f"{int(top['sc_delta'])} projected position(s)."
        )
    if not losers.empty:
        bot = losers.iloc[0]
        lines.append(
            f"Biggest loser: {bot['driver_code']} drops "
            f"{abs(int(bot['sc_delta']))} projected position(s)."
        )

    return sc, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Verification / demo block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os

    base = os.path.join(os.path.dirname(__file__), "data", "processed")
    laps_df = pd.read_csv(os.path.join(base, "hungary_2024_laps.csv"))
    meta_df = pd.read_csv(os.path.join(base, "hungary_2024_race_meta.csv"))

    with open(os.path.join(base, "hungary_2024_pit_loss.json")) as f:
        pit_data = json.load(f)
    PIT_LOSS_MS = pit_data["estimated_pit_loss_ms"]

    DEMO_LAPS = [25, 40, 55]
    SEP = "=" * 70

    for lap in DEMO_LAPS:
        print(f"\n{SEP}")
        print(f"  RACE STATE & PROJECTION AT LAP {lap}")
        print(SEP)

        state = get_race_state(laps_df, meta_df, lap)
        proj = project_standings(
            state, meta_df, PIT_LOSS_MS,
            laps_df=laps_df, current_lap=lap, total_race_laps=70,
        )

        print("\nProjected standings:")
        display_cols = [
            "projected_position", "driver_code", "driver_name", "constructor",
            "current_position", "position_delta",
            "stops_done", "stops_remaining", "degradation_rate_ms",
            "projected_gap_to_leader_ms", "retired",
        ]
        print(proj[display_cols].to_string(index=False))

        print("\nCommentary:")
        remaining = 70 - lap
        for line in generate_commentary(proj, PIT_LOSS_MS, remaining_laps=remaining):
            print(f"  {line}")

        # Safety car scenario
        sc_proj, sc_commentary = generate_safety_car_scenario(
            state, meta_df, PIT_LOSS_MS,
            laps_df=laps_df, current_lap=lap, total_race_laps=70,
        )
        print(f"\n{sc_commentary}")
        print()
