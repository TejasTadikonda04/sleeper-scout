# DraftSleeper

An NFL draft prospect forecasting tool that predicts career value for skill-position players and surfaces undervalued picks through a Sleeper-style UI. Given a prospect's college production, combine metrics, and draft position, DraftSleeper produces a **Sleeper Score (0–100)** and draft window recommendation.

---

## What It Does

DraftSleeper trains position-group regression models on historical draft data (2000–2019) and evaluates on a hold-out set (2020–2024). For each prospect, it outputs:

- **Weighted AV prediction** — projected career value based on Pro Football Reference's weighted Approximate Value metric
- **Surplus value** — how much a player's projection exceeds (or falls short of) what players drafted at that slot typically produce
- **Sleeper Score** — a 0–100 composite combining surplus value, breakout probability, peak upside, and durability
- **Tier label** — a plain-language classification (High Upside, Safe Floor, Boom or Bust, Developmental, Overdrafted)

Results are served through a FastAPI backend and rendered in a React + Vite frontend.

---

## Source Data

**File:** `data/raw/nfl_prospects.csv`

A pre-joined dataset of 6,387 drafted players combining PFR draft results, combine metrics, college stats, and NFL career outcomes. The project filters to 2,047 skill-position players (QB, WR, TE, RB).

### Key Columns

| Column | Description |
|---|---|
| `player_name`, `position`, `college` | Player identity |
| `draft_year`, `draft_round`, `draft_pick`, `draft_team` | Draft context |
| `height_in`, `weight_lbs` | Measurables |
| `forty_yard`, `vertical_jump`, `broad_jump`, `bench_reps`, `cone_drill`, `shuttle` | Combine metrics |
| `col_pass_completions` through `col_rec_tds` | College production stats |
| `weighted_av` | **Target variable** — weighted career Approximate Value |
| `nfl_games`, `pro_bowls`, `all_pro`, `seasons_started`, `hof` | NFL outcome indicators |

### Data Constraints

- **`career_av` is 100% null** — never used anywhere in the pipeline. `weighted_av` is the only valid target.
- **Combine metrics are heavily missing:** `bench_reps` (54%), `cone_drill` (47%), `shuttle` (46%), `broad_jump` (36%), `vertical_jump` (34%), `forty_yard` (20%). Missing values are flagged, not dropped.
- **`weighted_av` is 11.9% null** in the training window. These rows are excluded from model training but retained for feature analysis.
- Defensive and special teams stat columns (`col_total_tackles`, `col_sacks`, `col_fg_*`, etc.) are 100% null and unused.

---

## Pipeline

### Phase 1 — Data Preparation

Loads `nfl_prospects.csv`, filters to QB/WR/TE/RB, and splits into four CSVs by position group and time period:

| File | Group | Years |
|---|---|---|
| `skill_pass_train.csv` | QB, WR, TE | 2000–2019 |
| `skill_pass_holdout.csv` | QB, WR, TE | 2020–2024 |
| `skill_run_train.csv` | RB | 2000–2019 |
| `skill_run_holdout.csv` | RB | 2020–2024 |

The split is strictly by `draft_year` — never random. A data audit report (`data_audit.txt`) is generated documenting null rates and distributions.

### Phase 2 — Feature Engineering

Builds position-group-specific features from the raw columns:

- **Athletic composites** — speed score, burst score, agility score, BMI
- **Z-scored combine metrics** — normalized within position group (e.g., `forty_yard_z`, `burst_score_z`)
- **College production rates** — yards per carry, rush TD rate, reception rate, dominator rating, completion percentage, TD/INT ratio
- **Draft capital features** — pick value curve, early pick flag, Day 3 flag
- **Missing data flags** — boolean columns like `missing_combine`, `missing_college_stats`, `missing_outcome`

College stat columns that don't apply to a position (e.g., `col_pass_yards` for RBs) are set to NaN, not zero.

### Phase 3 — Modeling + Sleeper Scoring

Two model types are trained per position group:

- **Ridge regression** — baseline model
- **XGBoost** — primary model

Both predict `weighted_av`. Models are trained on the 2000–2019 data only; scalers and encoders are fit on train and applied to hold-out.

**Evaluation metrics (hold-out set):**

| Group | Model | N | RMSE | MAE | Spearman ρ |
|---|---|---|---|---|---|
| skill_pass | Ridge | 281 | 15.29 | 11.52 | 0.52 |
| skill_pass | XGBoost | 281 | 15.76 | 11.61 | 0.57 |
| skill_run | Ridge | 96 | 12.77 | 10.96 | 0.51 |
| skill_run | XGBoost | 96 | 11.18 | 9.38 | 0.57 |

XGBoost outperforms Ridge on Spearman rank correlation for both groups, meaning it ranks prospects more accurately even when point predictions carry noise.

**Sleeper Score formula:**

```
sleeper_score = (
    0.40 × surplus_value_percentile +
    0.20 × pro_bowl_probability +
    0.20 × peak_value_percentile +
    0.20 × availability_factor
) × 100
```

All component columns are retained in the scored output for transparency.

### Phase 4 — API Layer

A FastAPI application serves the scored data. Endpoints:

| Endpoint | Description |
|---|---|
| `GET /prospects` | List prospects with filtering (position, round, name search) |
| `GET /prospects/{id}` | Single prospect profile with full prediction data |
| `GET /prospects/{id}/score` | Sleeper Score breakdown for a prospect |
| `GET /scores/top` | Leaderboard sorted by Sleeper Score |

Storage is SQLite via SQLAlchemy. API docs auto-generate at `/docs`.

### Phase 5 — UI Layer

A React + Vite frontend with Tailwind CSS and Recharts. Three views:

- **Prospect List** — sortable/filterable table showing Sleeper Score, surplus value, breakout probability, and tier label
- **Prospect Detail** — full player profile with athletic radar chart (z-scored combine metrics), surplus value gauge, availability meter, and top projection drivers
- **Top Sleepers** — leaderboard of highest-scoring prospects across both position groups

---

## Directory Structure

```
draftsleeper/
├── actual_app/
│   ├── backend/                      # FastAPI implementation
│   │   ├── data/                     # Scored prospect CSVs
│   │   └── main.py                   # API entry point
│   └── frontend/                     # React + Vite application
│       ├── src/                      # Frontend source code
│       └── package.json              # Frontend dependencies
├── data/                             # Historical raw data
├── models/                           # Model logic and training
└── notebooks/                        # Original research notebooks
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| API | FastAPI, uvicorn |
| Frontend | React, Vite, Tailwind CSS, Recharts |

---

## How to Run

### Prerequisites

- Python 3.13+
- Node.js 18+

### 1. Run the Backend

```bash
# Navigate to the backend directory
cd actual_app/backend

# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the API
python -m uvicorn main:app --reload --port 8000
```
*The API will be available at http://localhost:8000. You can view the interactive documentation at http://localhost:8000/docs.*

### 2. Run the Frontend

In a **new terminal window**:

```bash
# Navigate to the frontend directory
cd actual_app/frontend

# Install dependencies (if not already done)
npm install

# Start the development server
npm run dev
```
*The UI will be available at http://localhost:5173.*

---

## Key Design Decisions

**Why weighted AV?** It's the most widely available career value metric in the source data. `career_av` was 100% null and unusable. Weighted AV front-loads recent seasons, which better captures early-career trajectory — relevant for draft evaluation.

**Why two position groups instead of four?** QB/WR/TE share passing and receiving production features. RB is distinct enough in its feature profile (rushing volume, speed score, burst) to warrant a separate model. Splitting further (e.g., QB alone) would leave too few training samples for reliable evaluation.

**Why XGBoost over neural nets?** With ~1,500 training samples for skill_pass and ~400 for skill_run, gradient boosting outperforms deeper architectures that need more data. XGBoost also handles missing features natively, which matters given combine metric missingness rates of 20–54%.

**Why time-based splits?** Random splits would leak future draft class information into training. A coach evaluating the 2024 class should see predictions built only on historical data, which is what the 2000–2019 / 2020–2024 split enforces.

---

## Feature Importance (Top Drivers)

### skill_pass (QB, WR, TE)

The model relies most heavily on draft position, positional identity (is_QB), and explosive athleticism (broad jump). College passing and receiving volume round out the top features.

### skill_run (RB)

Draft capital dominates — the top five features are all draft-position-related (is_early_pick, is_day_3_pick, draft_pick, draft_round, pick_value). This reflects the reality that for RBs, where you're drafted largely determines your opportunity, which drives AV. Agility metrics (shuttle, agility score) and college rushing volume are the first non-capital features.

---

## Scored Output Columns

The final scored CSVs (`skill_pass_scored.csv`, `skill_run_scored.csv`) include all original columns plus:

| Column | Description |
|---|---|
| `weighted_av_pred` | Model's predicted weighted AV |
| `surplus_value` | Predicted AV minus expected AV for that draft slot |
| `surplus_value_percentile` | Surplus value ranked within position group (0–1) |
| `pro_bowl_probability` | Estimated probability of making a Pro Bowl |
| `peak_value_percentile` | Projected peak single-season value, as a percentile |
| `availability_factor` | Durability projection (0 = high risk, 1 = durable) |
| `sleeper_score` | Final composite score (0–100) |

---

## Limitations

- **Target variable is career-level**, not season-level. The model predicts total weighted AV, not year-by-year trajectories. A future iteration could model season-level AV with year index as an explicit feature.
- **Small RB sample** — only 96 hold-out players for skill_run. Spearman of 0.57 is encouraging but should be interpreted with caution.
- **Combine missingness** — players who skip the combine or individual drills receive imputed/flagged values. Predictions for these players carry wider implicit uncertainty.
- **No availability model** — the `availability_factor` in the Sleeper Score is derived from proxy signals, not a dedicated injury/games-played model. Building one is a natural next step.
- **No conference or opponent adjustments** — college production is raw, not adjusted for strength of schedule.
