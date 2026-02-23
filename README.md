# F1 Race Strategy Simulator

A second-screen companion tool that projects live race standings adjusted for pit stop strategy. Built as a Streamlit prototype using the 2024 Hungarian Grand Prix, it lets you scrub through any lap, override expected pit stops, and instantly see how projected positions shift — including a tire degradation model that kicks in for "what-if" scenarios to balance pit loss against fresh-tire pace.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit UI — sidebar controls, projected standings, strategy commentary, and validation charts |
| `projection.py` | Pure-function engine — race state snapshots, projected standings with optional degradation model, commentary generation, safety car scenarios |
| `prepare_data.py` | Data pipeline — downloads and processes raw Kaggle F1 data into the CSVs and JSON used by the app |
| `data/processed/` | Pre-processed race data (laps, driver metadata, pit loss estimate) |
| `tests/degradation_validation/` | MAE comparison scripts and plots from degradation model testing |

## Data Source

Race data sourced from the [Kaggle Formula 1 World Championship dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020).

## Built With

Python · Streamlit · Pandas · NumPy · Plotly
