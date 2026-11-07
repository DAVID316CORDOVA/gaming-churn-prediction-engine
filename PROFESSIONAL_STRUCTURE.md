# Churn Gaming - Professional Project Structure

## Project Overview
LightGBM model predicting weekly Gaming player churn with AUC 0.8745.

## Complete Structure

```
churn_gaming/
├── README.md                  (Spanish documentation)
├── README_EN.md               (English comprehensive doc)
├── PROFESSIONAL_STRUCTURE.md  (This file)
├── requirements.txt           (All dependencies)
├── setup.py                   (Installable package)
├── Dockerfile                 (Production container)
├── Makefile                   (12 commands: test, lint, deploy)
├── .env.example               (Environment template)
├── .gitignore                 (Git ignore patterns)
│
├── src/                       (Production code)
│   ├── __init__.py
│   ├── main.py                (Entry point)
│   ├── model.py               (LightGBM model training)
│   ├── features.py            (Feature engineering)
│   ├── validator.py           (Temporal validation - walk-forward)
│   ├── data/
│   │   └── loader.py          (Athena data loading)
│   └── utils/
│       ├── logging.py         (Structured logging)
│       └── metrics.py         (Performance metrics)
│
├── tests/                     (Test suite)
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_features.py
│   │   └── test_model.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── fixtures/
│       └── mock_data.py
│
├── docs/                      (Documentation)
│   ├── API.md                 (Data/model API)
│   ├── ARCHITECTURE.md        (Technical design)
│   └── EXPERIMENTS.md         (Model iterations)
│
├── scripts/                   (Utility scripts)
│   ├── train.py               (Training script)
│   ├── evaluate.py            (Model evaluation)
│   └── export.py              (Export predictions)
│
├── notebooks/                 (Jupyter notebooks)
│   ├── 01_eda.ipynb           (Exploratory analysis)
│   └── 02_validation.ipynb    (Temporal validation)
│
├── graficos_churn_gaming_v2.png       (Professional 3x3 visualization)
├── generar_graficos_v2.py             (Matplotlib code - commented)
└── data/
    ├── samples/               (Example data)
    └── outputs/               (Model outputs)
```

## Key Files (Production Ready)

### Core Implementation
- `src/model.py` — LightGBM training with balanced class weights
- `src/features.py` — 28-35 features (Recency, Frequency, Monetary, etc.)
- `src/validator.py` — Walk-forward temporal validation (21 weeks)

### Configuration
- `requirements.txt` — All ML dependencies
- `setup.py` — Package installation
- `.env.example` — Environment variables (AWS, logging, etc.)

### Deployment
- `Dockerfile` — Multi-stage, production-grade
- `Makefile` — 12 commands (test, lint, deploy)
- `scripts/train.py` — Training pipeline

### Testing
- `tests/unit/` — Model & feature unit tests
- `tests/integration/` — End-to-end pipeline
- `tests/fixtures/` — Mock Athena responses

### Documentation
- `README_EN.md` — 16KB comprehensive technical
- `docs/API.md` — Data model interface
- `docs/ARCHITECTURE.md` — Design decisions

## Usage

```bash
# Install
pip install -r requirements.txt

# Train model
python scripts/train.py --data_range 2026-01-01:2026-11-06

# Run tests
make test

# Deploy
make deploy

# Docker
docker build -t churn-gaming:latest .
docker run churn-gaming:latest
```

## Model Specifications

- **Algorithm:** LightGBM with class_weight='balanced'
- **Features:** 28-35 (see docs/ARCHITECTURE.md)
- **Validation:** Walk-forward temporal (21 weeks, anti-leakage)
- **AUC:** 0.8745 (21-week average)
- **Precision (Critical):** 89-92%
- **Base Churn:** 66% (highly imbalanced)

## Data Sources

- `dwh_raw_pe.dwh_transaction` — Deposits/withdrawals
- `dwh_raw_pe.dwh_operation_calimaco_cuadre` — Operations
- `dwh_raw_pe.dwh_room_calimaco` — Product catalog
- `dwh_raw_pe.kyc_resumen` — KYC verification

## Status

✅ Production (2026-05-14)  
✅ All tests passing  
✅ Professional documentation  
✅ Docker containerized  
✅ CI/CD ready (Makefile)  

---
**Version:** 1.0 | **Last Updated:** 2026-11-06
