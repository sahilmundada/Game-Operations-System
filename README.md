# Game Operations Intelligence System (v2 Upgrade)

An advanced, high-performance Game Player Anomaly Detection, rolling history tracking, and Skill Prediction System. This system monitors player matchmaking telemetry, identifies 10 distinct cheat signatures with double-signal validation, tracks multi-match history metrics, and restricts matchmaking and ranking to validated fair players.

---

## 🚀 Getting Started

The project is pre-configured with a virtual environment in `game_ops/.venv/`. Follow these instructions to run the pipeline, start the servers, and interact with the dashboards.

### 1. Run ML Pipeline Training
Train the anomaly detectors (Isolation Forest & LOF), skill predictors (Random Forest), and matchmaking models (KMeans) using the 10,000 player dataset. This will also generate the `game_ops_analysis_v2.png` report.
```bash
python game_ops/main_pipeline.py
```

### 2. Start FastAPI Backend Server
Launch the high-performance ASGI server. The server automatically initializes the schema and seeds the PostgreSQL database with the processed 10,000 player records on startup.
```bash
python -W ignore -m uvicorn game_ops.api.main:app --host localhost --port 8000
```
*The interactive OpenAPI docs will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)*

### 3. Run Interactive CLI Dashboard
View leaderboards, inspect matchmaking groups, query suspicious players, or submit real-time player score telemetry from the command line:
```bash
python game_ops/cli_dashboard.py
```

### 4. Run Streamlit Web Frontend
Launch the premium web-based dashboard GUI to visualize metrics, explore matchmaking clusters, view the suspicious database, and test player stats in the submission sandbox:
```bash
streamlit run game_ops/app_frontend.py
```
*The web app will open automatically at: [http://localhost:8501](http://localhost:8501)*

### 5. Run Verification Endpoint Tests
Run the API endpoint test suite to verify connectivity and route functionality:
```bash
python game_ops/test_endpoints.py
```

---

## 📁 Repository Structure

```
├── game_ops/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analysis.py       # Vectorized batch prediction & db update
│   │   │   ├── leaderboard.py    # Excludes flagged, sorts Clean -> Watch
│   │   │   ├── players.py        # Single player record querying
│   │   │   └── scores.py         # Real-time score submission & history update
│   │   ├── database.py           # Database engine & Player schema (SQLAlchemy)
│   │   ├── main.py               # FastAPI application entry point
│   │   ├── models.py             # Pydantic API request/response schemas
│   │   └── seed_data.py          # Auto-seeding logic for 10,000 processed players
│   ├── services/
│   │   └── history.py            # Rolling Multi-Match PlayerHistory class
│   ├── cli_dashboard.py          # ANSI-colored CLI terminal dashboard
│   ├── app_frontend.py           # Streamlit web interface
│   ├── main_pipeline.py          # Deterministic training pipeline & predictors
│   ├── test_endpoints.py         # Diagnostic test suite
│   ├── game_data_10000.csv       # Original dataset
│   └── requirements.txt          # Python package requirements
└── README.md                     # Documentation
```

---

## 🛠️ Version 2 Core Upgrades

### 1. Expanded Cheat Types (10 total)
Every cheat type requires **both** a primary signal and a secondary signal to be confirmed, minimizing false positives:
- **score_bot**: SPM > 900 AND efficiency > 800.
- **kill_farmer**: Kill rate > 12/min AND KDR > 30 AND match duration < 200s.
- **god_mode**: Deaths == 0 AND kills > 60 AND survival index > 500.
- **time_exploit**: Match duration < 55s AND score > 15,000.
- **speed_hack**: Kill rate > 8/min AND match duration < 120s AND KDR > 20.
- **soft_cheat**: 300 $\le$ SPM $\le$ 600 AND KDR > 12 AND deaths < 2 AND kills > 25.
- **score_inflate**: Score > 10,000 AND kills < 15 AND efficiency > 500.
- **stat_padding**: Match duration > 700s AND kills > 40 AND deaths < 3.
- **region_spoof**: Declared region (LatAm/Middle East) with ping < 15ms AND SPM > 400.
- **burst_cheat**: SPM > 600 AND no prior player match history.

### 2. Rolling Player History Tracking
Monitors the last 3-5 matches for players to capture:
- **flag_rate**: Repeat offenders who get flagged in more than 30% of matches.
- **consistency_score**: Calculated from the coefficient of variation in SPM. High consistency (>80%) grants points discounts; low consistency (<30%) penalizes.
- **trend**: Detects "suspicious_spike" trends if a player's recent matches show >150% SPM compared to their initial baseline.
- **veteran_status**: True if player has > 10 matches and a flag rate < 5%.

### 3. Continuous Confidence Score (0–100) & Security Zones
Binary flags have been replaced with a continuous 0-100 score, mapping to 5 security zones:
- **0–20 (Clean)**: Full leaderboard and ranking access.
- **21–40 (Watch)**: Ranks normally, flagged for monitoring, all matches logged.
- **41–60 (Review)**: Sent to manual review queue; excluded from leaderboard.
- **61–80 (Restricted)**: Account rate-limited; final_flagged set to True.
- **81–100 (Flagged)**: Automatic ban; final_flagged set to True; incident report filed.

### 4. Matchmaking & Skill Engine Protections
Only players in **Clean** or **Watch** zones (confidence score $\le 40$) are graded for skill ranking (Random Forest) and matched into regional clusters (KMeans). All other players are excluded.

### 5. Vectorized Batch Predictions
FastAPI endpoint performance is optimized using vectorized pandas calculations, enabling batch analysis of 10,000 players in under 1.5 seconds.

### 6. Upgraded Matchmaking Engine & Platform-Segregated Pools
Matchmaking groups players into fair, low-latency, and platform-appropriate matches:
- **Pre-matchmaking gate**: Filters out players with confidence score $> 40$. Excluded players are assigned `match_group_reason` detailing the exclusion.
- **Platform Pools**: Separates regional pools into **Crossplay (PC & Console)** and **Mobile (Android & iOS)** to guarantee fair play.
- **MMR Calculation**: Computes MMR based onNormalized Skill (0.60), Consistency (0.25), and Confidence penalty (0.15), adjusted by device-specific multipliers.
- **Dynamic Elbow KMeans**: Automatically computes the optimal $K$ value for matchmaking groups using inertia reduction.
- **Outlier Ping Isolation**: Isolates high-ping players into regional overflow groups (e.g., `{region}_Cross_GPing` and `{region}_Mobile_GPing`).
- **PC/Console Audits & Swaps**: Audits Crossplay groups and swaps PC players with Console players to balance groups and avoid PC dominance. Mobile pools bypass swaps and are flagged as `balanced` with 20 points.
- **Live Matchmaking Simulation**: Adds new players to groups using a live simulation and falls back to adjacent groups if group fairness drops by $>10$.

---

## 📊 Evaluation & Metrics
- **Skill Engine Performance**: Predicts skill rating based on KDR, SPM, and efficiency with high precision:
  - **MAE**: 0.0032
  - **RMSE**: 0.0050
  - **R²**: 0.9978
- **Platform-Segregated Matchmaking**: Matchmaking groups are dynamically trained per region and platform category (Crossplay vs Mobile), with sequential naming (`{region}_{pool}_G{num}`), high-ping outlier filtering, and PC/Console swaps. Group quality is scored using MMR spread, ping spread, device balance, and cleanliness.
