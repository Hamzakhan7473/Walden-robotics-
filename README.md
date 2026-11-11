# Walden allocation

Allocate deployable robot units and Field Deployment Engineer hours across competing deployment requests. A deployment pays and produces training data, so the choice is an explore/exploit portfolio under two weekly capacities.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make data
make outputs
make test
make app
```

`make app` serves v2. `streamlit run app.py -- --version v1` loads the original min-max score, the full-duration objective, and the 0.6 novelty floor. The sidebar can switch versions after startup.

`data/DATA_NOTES.md` is intentionally unfinished.

## Layout

- `data/generate_data.py` — seeded synthetic book (seed 42) and the 12-week capacity file
- `src/scoring.py` — hard gates and a weighted score, with each contribution kept
- `src/allocate.py` — integer program (PuLP / CBC), exploration floor, and a reason on every row
- `src/sensitivity.py` — Monte Carlo weight perturbation, re-score, re-solve
- `app.py` — portfolio, score audit, weekly capacity, sensitivity
- `scripts/export_outputs.py` — writes `outputs/` and `reports/figures/`
- `outputs/` — scored book, allocation, utilization, sensitivity, and the v1/v2 funded-set diff
