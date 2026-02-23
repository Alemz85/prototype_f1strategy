"""
F1 Race Strategy Simulator — 2024 Hungarian Grand Prix
Streamlit prototype: explore race state at any lap and run "what if" scenarios.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os

from projection import (
    get_race_state,
    project_standings,
    generate_commentary,
    generate_safety_car_scenario,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config & constants
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Strategy Simulator",
    page_icon="🏎️",  # browser tab icon only
    layout="wide",
    initial_sidebar_state="expanded",
)

CONSTRUCTOR_COLORS = {
    "Red Bull":        "#3671C6",
    "McLaren":         "#FF8700",
    "Ferrari":         "#E8002D",
    "Mercedes":        "#27F4D2",
    "Aston Martin":    "#229971",
    "Alpine F1 Team":  "#FF87BC",
    "Haas F1 Team":    "#B6BABD",
    "RB F1 Team":      "#6692FF",
    "Williams":        "#64C4FF",
    "Sauber":          "#52E252",
}

CONSTRUCTOR_BG = {k: v + "22" for k, v in CONSTRUCTOR_COLORS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — F1 timing-screen aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --f1-bg: #15151E;
    --f1-card: #1E1E2E;
    --f1-border: #2A2A3C;
    --f1-text: #E0E0E0;
    --f1-muted: #888;
    --f1-accent: #E10600;
    --f1-green: #00D26A;
    --f1-red: #E10600;
    --f1-yellow: #FFD700;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
.stMarkdown, .stCaption, .stMetricLabel, .stMetricValue,
button, input, select, textarea, [data-testid="stMetric"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #15151E;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #E10600;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Standings row — grid layout */
.driver-row {
    display: grid;
    align-items: center;
    padding: 7px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 0.86rem;
    font-weight: 500;
    border-left: 4px solid transparent;
    transition: background 0.15s;
}
.driver-row:hover { filter: brightness(1.15); }

/* On-track grid: P DRV TEAM | STOPS | TIRE */
.driver-row.track-row {
    grid-template-columns: 28px 40px 1fr 60px 76px;
    gap: 6px;
}
/* Projected grid: P DRV TEAM | GAP | STOPS | Δ */
.driver-row.proj-row {
    grid-template-columns: 28px 40px 1fr 70px 70px 36px;
    gap: 6px;
}

.driver-row .pos {
    font-weight: 800;
    font-size: 1.05rem;
    text-align: center;
}
.driver-row .code {
    font-weight: 700;
}
.driver-row .team {
    color: #aaa;
    font-size: 0.8rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Sub-columns with subtle left divider */
.driver-row .col-stops,
.driver-row .col-tire,
.driver-row .col-gap,
.driver-row .col-sleft {
    font-size: 0.78rem;
    color: #bbb;
    text-align: center;
    border-left: 1px solid rgba(255,255,255,0.07);
    padding-left: 6px;
}

.driver-row .delta-pos {
    font-weight: 700;
    font-size: 0.85rem;
    text-align: center;
    border-left: 1px solid rgba(255,255,255,0.07);
    padding-left: 4px;
}
.delta-up { color: #00D26A; }
.delta-down { color: #E10600; }
.delta-same { color: #555; }

/* Retired badge */
.retired-badge {
    background: #3a1a1a;
    color: #E10600;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    grid-column: 4 / -1;
    text-align: center;
}

/* Header bar */
.standings-header {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #555;
    padding: 4px 12px 8px 12px;
    border-bottom: 1px solid #2A2A3C;
    margin-bottom: 6px;
    display: grid;
    align-items: center;
}
.standings-header.track-hdr {
    grid-template-columns: 28px 40px 1fr 60px 76px;
    gap: 6px;
}
.standings-header.proj-hdr {
    grid-template-columns: 28px 40px 1fr 70px 70px 36px;
    gap: 6px;
}
.standings-header span { text-align: center; }
.standings-header .team { text-align: left; }

/* Sidebar metric fix — smaller text */
[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 0.95rem;
}

/* SC section */
.sc-banner {
    background: linear-gradient(90deg, #b8860b22 0%, #b8860b11 100%);
    border-left: 4px solid #FFD700;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 10px 0;
    font-size: 0.9rem;
}

/* Commentary boxes */
.commentary-item {
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}

/* Section card */
.section-card {
    background: #1E1E2E;
    border: 1px solid #2A2A3C;
    border-radius: 10px;
    padding: 16px;
}

/* Metric overrides for F1 look */
[data-testid="stMetric"] {
    background: #1E1E2E;
    border: 1px solid #2A2A3C;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "data", "processed")
    laps = pd.read_csv(os.path.join(base, "hungary_2024_laps.csv"))
    meta = pd.read_csv(os.path.join(base, "hungary_2024_race_meta.csv"))
    with open(os.path.join(base, "hungary_2024_pit_loss.json")) as f:
        pit_data = json.load(f)
    return laps, meta, pit_data

laps_df, meta_df, pit_data = load_data()
PIT_LOSS_MS = pit_data["estimated_pit_loss_ms"]
TOTAL_LAPS = int(laps_df["lap"].max())


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render a driver row as HTML
# ─────────────────────────────────────────────────────────────────────────────
def render_track_row(pos, code, team, stops_done, tire_age, retired=False):
    """On-track standings row: P | DRV | TEAM | STOPS | TIRE."""
    color = CONSTRUCTOR_COLORS.get(team, "#888")
    bg = CONSTRUCTOR_BG.get(team, "#22222233")

    if retired:
        return (
            f'<div class="driver-row track-row" style="background:{bg}; border-left-color:{color};">'
            f'  <span class="pos">{pos}</span>'
            f'  <span class="code" style="color:{color};">{code}</span>'
            f'  <span class="team">{team}</span>'
            f'  <span class="retired-badge">RET</span>'
            f'</div>'
        )

    return (
        f'<div class="driver-row track-row" style="background:{bg}; border-left-color:{color};">'
        f'  <span class="pos">{pos}</span>'
        f'  <span class="code" style="color:{color};">{code}</span>'
        f'  <span class="team">{team}</span>'
        f'  <span class="col-stops">{stops_done}</span>'
        f'  <span class="col-tire">{tire_age} laps</span>'
        f'</div>'
    )


def render_proj_row(pos, code, team, gap_str, stops_left, delta, retired=False):
    """Projected standings row: P | DRV | TEAM | GAP | STOPS LEFT | Δ."""
    color = CONSTRUCTOR_COLORS.get(team, "#888")
    bg = CONSTRUCTOR_BG.get(team, "#22222233")

    if retired:
        return (
            f'<div class="driver-row proj-row" style="background:{bg}; border-left-color:{color};">'
            f'  <span class="pos">{pos}</span>'
            f'  <span class="code" style="color:{color};">{code}</span>'
            f'  <span class="team">{team}</span>'
            f'  <span class="retired-badge">RET</span>'
            f'</div>'
        )

    if delta > 0:
        delta_html = f'<span class="delta-pos delta-up">▲{delta}</span>'
    elif delta < 0:
        delta_html = f'<span class="delta-pos delta-down">▼{abs(delta)}</span>'
    else:
        delta_html = f'<span class="delta-pos delta-same">–</span>'

    return (
        f'<div class="driver-row proj-row" style="background:{bg}; border-left-color:{color};">'
        f'  <span class="pos">{pos}</span>'
        f'  <span class="code" style="color:{color};">{code}</span>'
        f'  <span class="team">{team}</span>'
        f'  <span class="col-gap">{gap_str}</span>'
        f'  <span class="col-sleft">{stops_left}</span>'
        f'  {delta_html}'
        f'</div>'
    )


def track_header():
    return (
        '<div class="standings-header track-hdr">'
        '  <span>P</span><span>DRV</span>'
        '  <span class="team">Team</span>'
        '  <span>Stops</span><span>Tire</span>'
        '</div>'
    )


def proj_header():
    return (
        '<div class="standings-header proj-hdr">'
        '  <span>P</span><span>DRV</span>'
        '  <span class="team">Team</span>'
        '  <span>Gap</span><span>Left</span><span>Δ</span>'
        '</div>'
    )


def format_gap(gap_ms):
    """Format a gap in milliseconds to a readable string."""
    if gap_ms == 0:
        return "LEADER"
    return f"+{gap_ms / 1000:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "f1logo.png")
    _, logo_col, _ = st.columns([1, 3, 1])
    with logo_col:
        st.image(logo_path, use_container_width=True)
    st.caption(
        "Explore any race lap by lap. "
        "Adjust strategy assumptions to see how projected standings change."
    )

    st.markdown("---")
    st.markdown("### Race Info")
    col1, col2 = st.columns(2)
    col1.metric("Circuit", "Hungaroring")
    col2.metric("Laps", "70")
    col1.metric("Date", "21 Jul 2024")
    col2.metric("Pit Loss", f"{PIT_LOSS_MS / 1000:.1f}s")
    col1.metric("Min Stops", "1")
    col2.metric("Compounds", "3")

    st.markdown("---")
    st.markdown("### Current Lap")
    current_lap = st.slider(
        "Select lap",
        min_value=1,
        max_value=TOTAL_LAPS,
        value=25,
        key="lap_slider",
        help="Drag to see the race state at any lap",
    )

    # Get race state for current lap (needed for override section)
    race_state = get_race_state(laps_df, meta_df, current_lap)

    st.markdown("---")
    with st.expander("Strategy Overrides", expanded=False):
        st.caption(
            "Change expected total stops below. "
            "Defaults from actual race data."
        )
        active_drivers = race_state[~race_state["retired"]].sort_values("current_position")
        overrides = {}
        for _, drv in active_drivers.iterrows():
            code = drv["driver_code"]
            default_stops = int(
                meta_df.loc[meta_df["driver_code"] == code, "expected_total_stops"].iloc[0]
            )
            new_val = st.number_input(
                f"{code} ({drv['constructor']})",
                min_value=0,
                max_value=5,
                value=default_stops,
                key=f"stops_{code}",
            )
            if new_val != default_stops:
                overrides[code] = new_val

    st.markdown("---")
    show_sc = st.toggle("Safety Car Scenario", value=False)


# Use simple model by default; degradation only when overrides are active
# (simple model is more accurate for default projections — 1.80 vs 2.12 MAE;
#  degradation adds value only for "what-if" stop scenarios.)
has_overrides = bool(overrides)
deg_kwargs = (
    dict(laps_df=laps_df, current_lap=current_lap, total_race_laps=TOTAL_LAPS)
    if has_overrides else {}
)

proj = project_standings(
    race_state, meta_df, PIT_LOSS_MS,
    expected_stops_override=overrides if has_overrides else None,
    **deg_kwargs,
)
remaining_laps = TOTAL_LAPS - current_lap
commentary = generate_commentary(proj, PIT_LOSS_MS, remaining_laps=remaining_laps)

if show_sc:
    sc_proj, sc_commentary = generate_safety_car_scenario(
        race_state, meta_df, PIT_LOSS_MS,
        expected_stops_override=overrides if has_overrides else None,
        **deg_kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main area — tabs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:2px; font-weight:800; font-size:2.2rem;">'
    '🇭🇺 Hungarian Grand Prix 2024</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">'
    f'  <h2 style="margin:0; font-size:1.4rem;">Lap {current_lap} '
    f'    <span style="color:#666; font-size:0.7em;">/ {TOTAL_LAPS}</span>'
    f'  </h2>'
    f'  <span style="background:#E10600; color:#fff; font-size:0.7rem; font-weight:800;'
    f'    padding:3px 10px; border-radius:4px; letter-spacing:1.5px;'
    f'    text-transform:uppercase; line-height:1;">LIVE</span>'
    f'</div>',
    unsafe_allow_html=True,
)
if overrides:
    override_text = ", ".join(f"{k}: {v}-stop" for k, v in overrides.items())
    st.info(f"Strategy overrides active: {override_text}")

tab_standings, tab_commentary, tab_validation = st.tabs(
    ["Race Standings", "Strategy Commentary", "Race Validation"]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Race Standings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_standings:
    n_sc_cols = 1 if show_sc else 0
    cols = st.columns(2 + n_sc_cols, gap="large")

    # ── Left: On-Track Standings ────────────────────────────────────────────
    with cols[0]:
        st.markdown("#### On-Track Standings")
        track = race_state.sort_values("current_position")

        html = track_header()
        for _, row in track.iterrows():
            html += render_track_row(
                pos=int(row["current_position"]),
                code=row["driver_code"],
                team=row["constructor"],
                stops_done=int(row["stops_done"]),
                tire_age=int(row["lap_in_stint"]),
                retired=row["retired"],
            )
        st.markdown(html, unsafe_allow_html=True)

    # ── Right: Projected Standings ──────────────────────────────────────────
    with cols[1]:
        st.markdown("#### Projected Standings")
        proj_sorted = proj.sort_values("projected_position")

        html = proj_header()
        for _, row in proj_sorted.iterrows():
            html += render_proj_row(
                pos=int(row["projected_position"]),
                code=row["driver_code"],
                team=row["constructor"],
                gap_str=format_gap(row["projected_gap_to_leader_ms"]),
                stops_left=int(row["stops_remaining"]),
                delta=int(row["position_delta"]),
                retired=row["retired"],
            )
        st.markdown(html, unsafe_allow_html=True)

    # ── Safety Car column ───────────────────────────────────────────────────
    if show_sc:
        with cols[2]:
            st.markdown("#### SC Projected")
            sc_sorted = sc_proj.sort_values("projected_position")

            html = proj_header()
            for _, row in sc_sorted.iterrows():
                html += render_proj_row(
                    pos=int(row["projected_position"]),
                    code=row["driver_code"],
                    team=row["constructor"],
                    gap_str=format_gap(row["projected_gap_to_leader_ms"]),
                    stops_left=int(row["stops_remaining"]),
                    delta=int(row["position_delta"]),
                    retired=row["retired"],
                )
            st.markdown(html, unsafe_allow_html=True)

    # ── Summary bar ─────────────────────────────────────────────────────────
    st.markdown("---")
    active_proj = proj[~proj["retired"]]
    movers = active_proj[active_proj["position_delta"] != 0]
    n_movers = len(movers)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Drivers Active", f"{len(active_proj)}")
    summary_cols[1].metric(
        "Position Changes",
        f"{n_movers} driver{'s' if n_movers != 1 else ''}",
    )
    gainers = active_proj[active_proj["position_delta"] > 0]
    losers = active_proj[active_proj["position_delta"] < 0]
    summary_cols[2].metric("Gaining", f"{len(gainers)} ▲")
    summary_cols[3].metric("Dropping", f"{len(losers)} ▼")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Strategy Commentary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_commentary:
    st.markdown("#### Strategy Insights")
    if commentary:
        for line in commentary:
            if line.startswith("⚠️"):
                st.warning(line)
            elif line.startswith("🔺"):
                st.success(line)
            elif line.startswith("🔻"):
                st.error(line)
            elif line.startswith("🔄"):
                st.info(line)
            elif line.startswith("⚔️"):
                st.info(line)
            else:
                st.info(line)
    else:
        st.caption("No notable strategy insights at this lap.")

    if show_sc:
        st.markdown("---")
        st.markdown("#### Safety Car Analysis")
        st.markdown(
            f'<div class="sc-banner">{sc_commentary.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: Race Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_validation:
    st.markdown("#### Projection vs Actual Results")

    # Validation table
    val_df = proj[[
        "driver_code", "driver_name", "constructor",
        "current_position", "projected_position", "retired",
    ]].copy()
    val_df = val_df.merge(
        meta_df[["driver_code", "final_position"]],
        on="driver_code",
        how="left",
    )
    val_df["proj_vs_actual"] = val_df["projected_position"] - val_df["final_position"]
    val_df = val_df.sort_values("projected_position")

    # Show table with conditional formatting
    st.dataframe(
        val_df.rename(columns={
            "driver_code": "Driver",
            "driver_name": "Name",
            "constructor": "Team",
            "current_position": "Track Pos",
            "projected_position": "Projected",
            "final_position": "Actual Final",
            "proj_vs_actual": "Difference",
            "retired": "Retired",
        }).style.map(
            lambda v: (
                "color: #00D26A" if isinstance(v, (int, float)) and v < 0
                else "color: #E10600" if isinstance(v, (int, float)) and v > 0
                else ""
            ),
            subset=["Difference"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Projected position evolution chart ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### Projected Position Evolution")

    driver_options = meta_df.sort_values("final_position")["driver_code"].tolist()
    default_drivers = driver_options[:3]
    selected_drivers = st.multiselect(
        "Select driver(s) to display",
        options=driver_options,
        default=default_drivers,
    )

    if selected_drivers:
        @st.cache_data
        def compute_evolution(_laps_df, _meta_df, _pit_loss, _overrides_tuple, total_laps):
            """Compute projected position for every lap for all drivers."""
            records = []
            overrides_dict = dict(_overrides_tuple) if _overrides_tuple else None
            for lap in range(1, total_laps + 1):
                state = get_race_state(_laps_df, _meta_df, lap)
                p = project_standings(state, _meta_df, _pit_loss, overrides_dict)
                for _, row in p.iterrows():
                    records.append({
                        "lap": lap,
                        "driver_code": row["driver_code"],
                        "projected_position": int(row["projected_position"]),
                        "retired": row["retired"],
                    })
            return pd.DataFrame(records)

        overrides_tuple = tuple(overrides.items()) if overrides else ()
        evo_df = compute_evolution(laps_df, meta_df, PIT_LOSS_MS, overrides_tuple, TOTAL_LAPS)

        fig = go.Figure()

        for code in selected_drivers:
            drv_data = evo_df[evo_df["driver_code"] == code]
            team = meta_df.loc[meta_df["driver_code"] == code, "constructor"].iloc[0]
            color = CONSTRUCTOR_COLORS.get(team, "#888")
            actual_final = int(
                meta_df.loc[meta_df["driver_code"] == code, "final_position"].iloc[0]
            )

            fig.add_trace(go.Scatter(
                x=drv_data["lap"],
                y=drv_data["projected_position"],
                mode="lines",
                name=f"{code} (projected)",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{code}</b><br>Lap %{{x}}<br>Projected P%{{y}}<extra></extra>",
            ))

            fig.add_trace(go.Scatter(
                x=[1, TOTAL_LAPS],
                y=[actual_final, actual_final],
                mode="lines",
                name=f"{code} actual (P{actual_final})",
                line=dict(color=color, width=1.5, dash="dot"),
                hoverinfo="skip",
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#15151E",
            plot_bgcolor="#1E1E2E",
            font=dict(family="Outfit"),
            yaxis=dict(
                title="Position",
                autorange="reversed",
                dtick=1,
                gridcolor="#2A2A3C",
            ),
            xaxis=dict(
                title="Lap",
                gridcolor="#2A2A3C",
            ),
            legend=dict(
                bgcolor="#15151E",
                bordercolor="#2A2A3C",
                font=dict(size=11),
            ),
            height=480,
            margin=dict(l=40, r=20, t=30, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Select one or more drivers to see their projected position evolution.")
