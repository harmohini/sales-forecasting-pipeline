# Sales Forecasting Pipeline with Automated Retraining

## Project Objective
The **Sales Forecasting Pipeline** is an end-to-end machine learning system designed to automate sales data ingestion, schema validation, preprocessing, database persistence in PostgreSQL, model training/inference, and REST API delivery via FastAPI, with support for automated, scheduled model retraining.

---

## Planned Architecture & Technology Stack

### Technology Stack
- **Data Processing & ML**: Python 3.10+, Pandas, NumPy, Scikit-Learn, Joblib
- **Database**: PostgreSQL, SQLAlchemy, Psycopg2-binary
- **REST API**: FastAPI, Uvicorn, Pydantic
- **Environment & Config**: Python-Dotenv
- **Testing**: Pytest, HTTPX
- **Containerization**: Docker, Docker Compose

---

## Directory Structure

```text
sales-forecasting-pipeline/
│
├── data/
│   ├── incoming/        # Storage for incoming sales CSV files
│   ├── processed/       # Cleaned & feature-engineered datasets
│   ├── archive/         # Processed raw CSV files archive
│   └── README.md
│
├── src/                 # Main Python package
│   ├── __init__.py
│   ├── ingestion/       # File detection, schema validation & logging
│   ├── preprocessing/   # Data cleaning, type conversion & feature engineering
│   ├── database/        # PostgreSQL connection, ORM models & query execution
│   ├── forecasting/     # Model training, inference, evaluation & serialization
│   └── pipeline/        # End-to-end pipeline execution & retraining triggers
│
├── api/                 # FastAPI REST prediction endpoints & routing
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routes.py
│
├── models/              # Saved model binary artifacts (.pkl, .joblib)
├── tests/               # Unit and integration test suite
├── logs/                # Application execution log files
├── scripts/             # CLI executable entry points
├── config/              # Centralized configuration management
│
├── .env.example         # Template for environment variables
├── .gitignore           # Git exclusion definitions
├── requirements.txt     # Dependency specifications
├── README.md            # Project documentation
├── Dockerfile           # Docker container builder definition
├── docker-compose.yml   # Multi-container service definition
└── pyproject.toml       # Python package configuration
```

---

## Future Pipeline Flow

1. **Ingestion**: Incoming CSV files deposited into `data/incoming/` are detected, schema-validated, and logged.
2. **Preprocessing**: Raw data is cleaned, handled for missing values, and augmented with time-series feature engineering.
3. **Database Persistence**: Processed data is stored in PostgreSQL tables via SQLAlchemy models.
4. **Model Training & Evaluation**: Data is extracted to train sales forecasting models, evaluate accuracy metrics, and persist model binaries into `models/`.
5. **API Serving**: FastAPI exposes REST endpoints (`/api/v1/predict`) to serve inference requests.
6. **Automated Retraining**: Scheduled scripts or retraining triggers detect data changes and execute retrains automatically.

---

## Local Setup Instructions

### 1. Prerequisites
- Python 3.9+ installed on your mac System
- PostgreSQL running locally or via Docker

### 2. Create Virtual Environment

```bash
# Navigate to the project root directory
cd sales-forecasting-pipeline

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment on macOS
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to create your local `.env` configuration file:

```bash
cp .env.example .env
```

Edit `.env` to configure your PostgreSQL credentials:

```env
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/sales_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=sales_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
MODEL_PATH=models/
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Data Ingestion

### Incoming File Placement
Place raw input datasets into the `data/incoming/` directory:
- `train.csv`: Historical daily sales records (`Store`, `Date`, `Sales`, `Customers`, etc.)
- `store.csv`: Store metadata (`Store`, `StoreType`, `Assortment`, `CompetitionDistance`, etc.)
- `test.csv`: Test set for inference (`Id`, `Store`, `Date`, `Promo`, etc.)

### Schema & Data Validation
Validation is dataset-specific and enforced by `src/ingestion/validator.py`:
- Checks required column presence and flags unexpected headers.
- Verifies non-empty datasets and validates key columns (`Store`, `Date`, `Id`).
- Ensures valid `YYYY-MM-DD` date formatting.
- Validates numeric boundaries (e.g. `Sales >= 0`, `Customers >= 0`, `Store > 0`).
- Raises explicit `SchemaValidationError` or `DataValidationError` upon failure.

### Database Storage Schema
Data is persisted into raw PostgreSQL tables via SQLAlchemy ORM models (`src/database/models.py`):
- `sales_train_raw`: Composite primary key `(store, date)`
- `stores_raw`: Primary key `store`
- `sales_test_raw`: Primary key `id`, unique constraint `(store, date)`

### Running Data Ingestion
Execute the ingestion script to process, validate, and load incoming data into PostgreSQL:

```bash
python scripts/ingest_data.py
```

To automatically move successfully processed CSV files to `data/archive/`, append the `--archive` flag:

```bash
python scripts/ingest_data.py --archive
```

### Idempotency & Duplicate Prevention
Ingestion is strictly idempotent and safe to execute multiple times:
- Uses PostgreSQL native `ON CONFLICT DO NOTHING` statements matching each table's unique natural keys.
- Skips previously inserted records without raising errors or dropping existing database tables.
---

## Stage 3 — Forecasting Model

### Time-Based Validation & Data Leakage Prevention
Model evaluation uses strict chronological validation splitting:
- **Training Period**: `2013-01-29 → 2015-06-19` (778,511 rows)
- **Validation Period**: `2015-06-20 → 2015-07-31` (40,282 rows, 42-day window)
- **Feature Exclusion**: The `customers` feature is explicitly excluded (`excluded_features = ["sales", "customers", "date"]`) because future customer counts are unavailable at production prediction time.

### Benchmark & Machine Learning Model
- **Baseline**: Seasonal Lag-7 baseline (`prediction = sales_lag_7`) evaluated on the validation period.
- **ML Model**: `HistGradientBoostingRegressor` (from `sklearn.ensemble`) using categorical ordinal encoding fitted strictly on training data (`X_train`).

### Evaluation Metrics
Validation metrics demonstrate that the ML model drastically outperforms the seasonal baseline:
- **Baseline (Seasonal Lag-7)**: MAE = $2,191.58$, RMSE = $2,829.74$, RMSPE = $41.02\%$
- **ML Model (HistGradientBoostingRegressor)**: MAE = $583.64$, RMSE = $846.96$, RMSPE = **$12.29\%$** ($28.73\%$ RMSPE reduction)

### Running Model Training
Execute the training script to train the model, evaluate performance, and serialize artifacts:

```bash
python scripts/train_model.py
```

### Model Artifacts & Prediction Utility
- **Joblib Model Artifact**: [models/sales_forecasting_model.joblib](file:///Users/apple/Pictures/.gemini/antigravity-ide/scratch/sales-forecasting-pipeline/models/sales_forecasting_model.joblib) (contains estimator, fitted categorical encoder, feature names).
- **JSON Metadata**: [models/model_metadata.json](file:///Users/apple/Pictures/.gemini/antigravity-ide/scratch/sales-forecasting-pipeline/models/model_metadata.json).
- **Prediction Function**: Programmatic inference provided by `generate_forecast(df, artifact)` in [src/forecasting/predict.py](file:///Users/apple/Pictures/.gemini/antigravity-ide/scratch/sales-forecasting-pipeline/src/forecasting/predict.py).

---

## Stage 4 — FastAPI REST API

### Production Endpoints
The REST API provides low-latency inference, model info, and service health status:
- **`GET /health`**: Health status and API operational state (`{"status": "healthy", "service": "sales-forecasting-api", "version": "1.0.0"}`).
- **`GET /model-info`**: Detailed active model metadata, feature lists, and validation metrics (`MAE`, `RMSE`, `RMSPE`).
- **`POST /predict`**: Batch sales forecasting given array of store input objects.
- **`GET /docs`**: Interactive Swagger UI documentation.

### Running the REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Stage 5 — Automated Retraining & Safe Model Promotion

### Automated Retraining Workflow
The retraining module ([src/pipeline/retraining.py](file:///Users/apple/Pictures/.gemini/antigravity-ide/scratch/sales-forecasting-pipeline/src/pipeline/retraining.py)) implements safe, reproducible model retraining:
1. **Data Integrity Check**: Verifies availability of processed dataset (`data/processed/sales_features.parquet`).
2. **Current Metric Inspection**: Reads active production model metrics from `models/model_metadata.json`.
3. **Candidate Model Training**: Fits a candidate `HistGradientBoostingRegressor` model on updated data using chronological train/validation split.
4. **Candidate Validation Safety Check**: Verifies candidate model predictions for `NaN`/`Inf` anomalies.
5. **Candidate vs. Production Performance Comparison**: Evaluates candidate RMSPE against active production model RMSPE.
6. **Safe Atomic Promotion**: Candidate is promoted ONLY if candidate RMSPE improves over production RMSPE by at least `MIN_IMPROVEMENT_PERCENT` (1.0%). Upon promotion:
   - Current production model is archived to `models/archive/sales_forecasting_model_v{version}_{timestamp}.joblib`.
   - Semantic version is incremented (e.g. `1.0.0` → `1.1.0`).
   - Candidate model and metadata are saved atomically to `models/sales_forecasting_model.joblib`.
7. **Audit Logging**: Appends complete run metadata, row counts, metric comparisons, and promotion decision to `models/retraining_history.json`.

### Executing Automated Retraining

```bash
python scripts/retrain_model.py
```

### Running Test Suite

```bash
pytest -v
```

---

## Stage 6 — Automated Retraining Schedule

The sales forecasting pipeline includes an automated retraining scheduler using macOS `launchd`.

### Scheduler Architecture & Schedule
- **Technology**: macOS LaunchAgent (`launchd`).
- **Configuration Template**: [deployment/launchd/com.salesforecasting.retrain.plist](file:///Users/apple/Pictures/.gemini/antigravity-ide/scratch/sales-forecasting-pipeline/deployment/launchd/com.salesforecasting.retrain.plist)
- **Installed Agent**: `~/Library/LaunchAgents/com.salesforecasting.retrain.plist`
- **Schedule**: Daily at **02:00 local time** (`StartCalendarInterval`: `Hour=2`, `Minute=0`).
- **Overlapping Run Protection**: `scripts/scheduled_retrain.sh` acquires a process lock file (`logs/scheduler/retrain.lock`) to prevent concurrent training runs.
- **Model Safety Guarantee**: Delegates directly to Stage 5 retraining logic (`python scripts/retrain_model.py`). Candidates are promoted ONLY if candidate RMSPE improves over production RMSPE by $\ge 1.0\%$.

### Logging
- **Standard Output**: `logs/scheduler/retraining_stdout.log`
- **Standard Error**: `logs/scheduler/retraining_stderr.log`
- **Audit Trail Log**: `models/retraining_history.json`

### CLI Helper Commands

#### 1. Install Scheduler
Installs and registers the daily LaunchAgent:
```bash
bash scripts/install_scheduler.sh
```

#### 2. Check Scheduler Status
Inspects launchd service status, process PID, stdout/stderr log snippets, and latest retraining run details:
```bash
bash scripts/scheduler_status.sh
```

#### 3. Manually Trigger Retraining Run
Immediately executes the scheduled retraining job without waiting for 02:00:
```bash
bash scripts/test_scheduler.sh
```

#### 4. Uninstall Scheduler
Safely unloads and removes the LaunchAgent without touching project files or model artifacts:
```bash
bash scripts/uninstall_scheduler.sh
```

---

## Local Development

The project is configured to run directly using Python 3.10+ and a local PostgreSQL instance without requiring Docker.

### 1. Requirements & Setup
- Python 3.10+ (macOS / Linux / Windows)
- Local PostgreSQL instance with database `sales_forecasting`

### 2. Environment Configuration
Copy `.env.example` to `.env` and verify database credentials:
```bash
cp .env.example .env
```
Ensure `POSTGRES_DB=sales_forecasting` and `DATABASE_URL` are configured properly.

### 3. Running the REST API Server
Start the Uvicorn server locally:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access points:
- Health Check: `http://localhost:8000/health`
- Model Metadata: `http://localhost:8000/model-info`
- Prediction Endpoint: `http://localhost:8000/predict`
- Interactive Swagger Docs: `http://localhost:8000/docs`

### 4. Local Scheduled Retraining
Local automated model retraining is managed via macOS `launchd`:
```bash
bash scripts/install_scheduler.sh
```

---

## Production Deployment (Render)

The application is fully prepared for native deployment on **Render** (or any cloud web hosting service) without requiring a local or remote Docker daemon.

> **Note**: Local Docker runtime was not used because the development machine runs macOS 12.7.6, which is outside the supported range of current Docker Desktop releases. The application itself is fully runnable without Docker.

### 1. Render Service Configuration
Deploying to Render can be done using the included `render.yaml` Blueprint or manually creating a Web Service:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Python Version**: `3.11.10`

### 2. Required Environment Variables
Configure the following environment variables in the Render Dashboard:

| Variable | Description | Example / Recommended Value |
| --- | --- | --- |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/sales_forecasting` |
| `POSTGRES_HOST` | Database host | Provided by Render PostgreSQL |
| `POSTGRES_PORT` | Database port | `5432` |
| `POSTGRES_DB` | Database name | `sales_forecasting` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | Secret database password |
| `MODEL_PATH` | Relative path to model directory | `models/` |
| `LOG_LEVEL` | Application logging level | `INFO` |

### 3. Database Setup (PostgreSQL)
Provision a PostgreSQL database instance on Render with the database name `sales_forecasting`. Link its connection string to the `DATABASE_URL` environment variable of your API web service.

### 4. Model Artifact Requirements
- The production server loads the active production model artifact (`models/sales_forecasting_model.joblib`) into memory upon startup.
- The production API does **NOT** retrain the model on server startup.
- Ensure the production model binary `models/sales_forecasting_model.joblib` and metadata `models/model_metadata.json` are present in the repository or attached build workspace.

### 5. Production Health Check & API Endpoints
- **Health Endpoint**: `/health` (Returns HTTP 200 `{"status": "ok", "service": "sales-forecasting-api"}`)
- **Model Metadata Endpoint**: `/model-info` (Returns HTTP 200 with model metrics)
- **Prediction Endpoint**: `/predict` (Accepts JSON payload and returns sales predictions)

Run the included health verification script against your deployed app:
```bash
python scripts/verify_health.py https://<your-render-app>.onrender.com
```

### 6. Production Automated Retraining
Render Web Services cannot run macOS `launchd`. For production automated model retraining:
- Setup a **Render Cron Job** running `python scripts/retrain_model.py` on a periodic schedule (e.g. daily `0 2 * * *`).
- Or trigger `python scripts/retrain_model.py` via an external scheduler or GitHub Actions workflow.

---

## Docker

Docker configuration files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) are maintained in the repository for containerized deployment to supported production environments (such as AWS ECS, GCP Cloud Run, Kubernetes, or Linux servers).

### Quick Start Commands (On Docker-Supported Machines)
```bash
# Build and launch multi-container stack (API + PostgreSQL)
docker compose up -d --build

# View container logs
docker compose logs -f api

# Shutdown services
docker compose down
```






