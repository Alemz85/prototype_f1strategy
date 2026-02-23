"""
Prepare clean datasets for the 2024 Hungarian Grand Prix
from the Kaggle F1 dataset.

Outputs:
  - hungary_2024_laps.csv   (one row per driver per lap)
  - hungary_2024_race_meta.csv  (one row per driver)
  - Prints estimated pit lane time loss and summary stats
"""

import pandas as pd
import numpy as np
import os

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed")

# ── 1. Load raw CSVs ────────────────────────────────────────────────────────
print("Loading raw CSVs...")
races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"), na_values="\\N")
lap_times = pd.read_csv(os.path.join(DATA_DIR, "lap_times.csv"), na_values="\\N")
pit_stops = pd.read_csv(os.path.join(DATA_DIR, "pit_stops.csv"), na_values="\\N")
results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"), na_values="\\N")
drivers = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"), na_values="\\N")
constructors = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"), na_values="\\N")
status = pd.read_csv(os.path.join(DATA_DIR, "status.csv"), na_values="\\N")
circuits = pd.read_csv(os.path.join(DATA_DIR, "circuits.csv"), na_values="\\N")

# ── 2. Identify the 2024 Hungarian Grand Prix ───────────────────────────────
# Find Hungaroring circuit
hungary_circuit = circuits[circuits["name"].str.contains("Hungaroring", case=False, na=False)]
if hungary_circuit.empty:
    # Fallback: search by country
    hungary_circuit = circuits[circuits["country"].str.contains("Hungary", case=False, na=False)]
hungary_circuit_id = hungary_circuit["circuitId"].iloc[0]

race_row = races[(races["year"] == 2024) & (races["circuitId"] == hungary_circuit_id)]
if race_row.empty:
    raise ValueError("Could not find 2024 Hungarian Grand Prix in races.csv")

RACE_ID = int(race_row["raceId"].iloc[0])
print(f"Found 2024 Hungarian GP: raceId={RACE_ID}, "
      f"date={race_row['date'].iloc[0]}, "
      f"circuit={hungary_circuit['name'].iloc[0]}")

# ── 3. Filter data to this race ─────────────────────────────────────────────
race_laps = lap_times[lap_times["raceId"] == RACE_ID].copy()
race_pits = pit_stops[pit_stops["raceId"] == RACE_ID].copy()
race_results = results[results["raceId"] == RACE_ID].copy()

# ── 4. Build lookup tables ──────────────────────────────────────────────────
driver_lookup = drivers[["driverId", "code", "surname"]].copy()
driver_lookup.columns = ["driverId", "driver_code", "driver_name"]

constructor_lookup = constructors[["constructorId", "name"]].copy()
constructor_lookup.columns = ["constructorId", "constructor"]

status_lookup = status[["statusId", "status"]].copy()

# Merge results with driver, constructor, and status info
race_results = (
    race_results
    .merge(driver_lookup, on="driverId", how="left")
    .merge(constructor_lookup, on="constructorId", how="left")
    .merge(status_lookup, on="statusId", how="left")
)

# ── 5. Build the laps dataframe ─────────────────────────────────────────────
# Start with lap_times
laps = race_laps.merge(driver_lookup, on="driverId", how="left")

# Add constructor from results
driver_constructor = race_results[["driverId", "constructor"]].drop_duplicates()
laps = laps.merge(driver_constructor, on="driverId", how="left")

# Rename and clean columns
laps = laps.rename(columns={
    "milliseconds": "lap_time_ms",
})

# ── 6. Add pit stop information ─────────────────────────────────────────────
# Build a pit-stop lookup: which driver pitted on which lap
pit_info = race_pits[["driverId", "lap", "stop", "milliseconds"]].copy()
pit_info = pit_info.rename(columns={
    "stop": "pit_stop_number",
    "milliseconds": "pit_duration_ms",
})

# Merge pit info onto laps
laps = laps.merge(pit_info, on=["driverId", "lap"], how="left")

# Create is_pit_lap boolean
laps["is_pit_lap"] = laps["pit_stop_number"].notna()

# ── 7. Compute running / cumulative columns ────────────────────────────────
# Sort by driver then lap for cumulative calculations
laps = laps.sort_values(["driverId", "lap"])

# Cumulative time per driver
laps["cumulative_time_ms"] = laps.groupby("driverId")["lap_time_ms"].cumsum()

# Total stops so far (running count of pit stops)
laps["total_stops_so_far"] = laps.groupby("driverId")["is_pit_lap"].cumsum().astype(int)

# Stint number: starts at 1, increments after each pit stop
# stint_number = total_stops_so_far + 1, but the pit lap itself is the last lap
# of the current stint (you pit at the end of the lap), so:
#   stint_number = (cumulative pit stops BEFORE this lap) + 1
# We shift the cumulative sum by 1 for each driver
laps["stint_number"] = (
    laps.groupby("driverId")["is_pit_lap"]
    .apply(lambda x: x.shift(1, fill_value=False).cumsum() + 1)
    .reset_index(level=0, drop=True)
    .astype(int)
)

# Lap in stint: consecutive count within each stint
laps["lap_in_stint"] = laps.groupby(["driverId", "stint_number"]).cumcount() + 1

# ── 8. Add grid and final position from results ────────────────────────────
grid_final = race_results[["driverId", "grid", "positionOrder"]].copy()
grid_final.columns = ["driverId", "grid_position", "final_position"]
laps = laps.merge(grid_final, on="driverId", how="left")

# ── 9. Select and order final columns for laps CSV ──────────────────────────
laps_out = laps[[
    "lap", "driver_code", "driver_name", "constructor",
    "lap_time_ms", "cumulative_time_ms", "position",
    "is_pit_lap", "pit_stop_number", "pit_duration_ms",
    "total_stops_so_far", "stint_number", "lap_in_stint",
    "grid_position", "final_position"
]].copy()

# Sort by lap then position
laps_out = laps_out.sort_values(["lap", "position"]).reset_index(drop=True)

# ── 10. Build race_meta dataframe ───────────────────────────────────────────
# Total laps and total stops per driver
driver_stats = (
    laps.groupby("driverId")
    .agg(
        total_laps=("lap", "max"),
        total_stops=("is_pit_lap", "sum"),
    )
    .reset_index()
)
driver_stats["total_stops"] = driver_stats["total_stops"].astype(int)

# Pit laps list per driver
pit_laps_list = (
    race_pits.groupby("driverId")["lap"]
    .apply(lambda x: ",".join(x.astype(str).tolist()))
    .reset_index()
    .rename(columns={"lap": "pit_laps"})
)

# Build meta
meta = (
    race_results[["driverId", "driver_code", "driver_name", "constructor",
                   "grid", "positionOrder", "status"]]
    .rename(columns={"grid": "grid_position", "positionOrder": "final_position"})
    .merge(driver_stats[["driverId", "total_stops", "total_laps"]], on="driverId", how="left")
    .merge(pit_laps_list, on="driverId", how="left")
)

# expected_total_stops = total_stops (will be user-overridable in the app)
meta["expected_total_stops"] = meta["total_stops"]

# Fill pit_laps NaN (drivers who never pitted) with empty string
meta["pit_laps"] = meta["pit_laps"].fillna("")

# Select and order final columns
meta_out = meta[[
    "driver_code", "driver_name", "constructor",
    "grid_position", "final_position", "total_stops",
    "expected_total_stops", "total_laps", "status", "pit_laps"
]].copy()

meta_out = meta_out.sort_values("final_position").reset_index(drop=True)

# ── 11. Estimate pit lane time loss ─────────────────────────────────────────
pit_lap_times = laps_out[laps_out["is_pit_lap"] == True]["lap_time_ms"]
non_pit_lap_times = laps_out[laps_out["is_pit_lap"] == False]["lap_time_ms"]

median_pit = pit_lap_times.median()
median_non_pit = non_pit_lap_times.median()
estimated_pit_loss_ms = median_pit - median_non_pit

print(f"\n{'='*60}")
print(f"Estimated pit lane time loss:")
print(f"  Median pit lap time:     {median_pit:.0f} ms ({median_pit/1000:.3f} s)")
print(f"  Median non-pit lap time: {median_non_pit:.0f} ms ({median_non_pit/1000:.3f} s)")
print(f"  Estimated pit loss:      {estimated_pit_loss_ms:.0f} ms ({estimated_pit_loss_ms/1000:.3f} s)")
print(f"{'='*60}")

# ── 12. Save outputs ────────────────────────────────────────────────────────
laps_path = os.path.join(OUTPUT_DIR, "hungary_2024_laps.csv")
meta_path = os.path.join(OUTPUT_DIR, "hungary_2024_race_meta.csv")
pit_loss_path = os.path.join(OUTPUT_DIR, "hungary_2024_pit_loss.json")

laps_out.to_csv(laps_path, index=False)
meta_out.to_csv(meta_path, index=False)

# Save pit loss value as JSON for easy consumption by the Streamlit app
import json
pit_loss_data = {
    "race": "2024 Hungarian Grand Prix",
    "raceId": RACE_ID,
    "circuit": hungary_circuit["name"].iloc[0],
    "estimated_pit_loss_ms": round(estimated_pit_loss_ms, 1),
    "estimated_pit_loss_s": round(estimated_pit_loss_ms / 1000, 3),
    "median_pit_lap_ms": round(median_pit, 1),
    "median_non_pit_lap_ms": round(median_non_pit, 1),
}
with open(pit_loss_path, "w") as f:
    json.dump(pit_loss_data, f, indent=2)

print(f"\nFiles saved:")
print(f"  {laps_path}")
print(f"  {meta_path}")
print(f"  {pit_loss_path}")

# ── 13. Print summary stats ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY STATS — 2024 Hungarian Grand Prix")
print(f"{'='*60}")
print(f"Number of drivers:  {meta_out['driver_code'].nunique()}")
print(f"Total laps in data: {laps_out['lap'].max()}")
print(f"Total rows in laps: {len(laps_out)}")
print()

print("Pit stops per driver:")
pit_summary = meta_out[["driver_code", "driver_name", "constructor",
                         "grid_position", "final_position",
                         "total_stops", "pit_laps", "status"]].to_string(index=False)
print(pit_summary)

print(f"\n{'='*60}")
print("Sample laps data (first 10 rows):")
print(laps_out.head(10).to_string(index=False))
print(f"{'='*60}")
