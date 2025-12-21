# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a data science and analytics portfolio workspace containing multiple independent projects demonstrating ML, data engineering, visualization, and dashboard development. All projects are authored by Alexy Louis.

## Project Structure

```
Claude_projects/
└── Portfolio creation/
    ├── data-analysis-portfolio/data-analysis-portfolio/
    │   ├── 01-exploratory-data-analysis/    # Customer churn EDA
    │   ├── 02-classification-ml/            # Churn prediction models
    │   ├── 03-regression-ml/                # House price prediction
    │   ├── 04-data-processing-apis/         # ETL pipeline framework
    │   └── 05-time-series-forecasting/      # Energy forecasting
    ├── Dashboard_ecommerce/                 # Streamlit e-commerce dashboard
    ├── Dashboard_inventory/                 # Retail inventory analytics
    ├── Clinical_Trial_Dashboard/            # Phase III trial analytics
    └── nlp-sentiment-analysis/              # NLP sentiment classification
```

## Common Commands

### Streamlit Dashboards
```bash
# E-commerce dashboard
cd "Portfolio creation/Dashboard_ecommerce"
streamlit run dashboard.py

# Inventory dashboard
cd "Portfolio creation/Dashboard_inventory"
streamlit run retail_dashboard.py

# Clinical trial dashboard
cd "Portfolio creation/Clinical_Trial_Dashboard"
streamlit run clinical_dashboard.py
```

### Data Generation
Each dashboard has a data generator script:
```bash
python generate_data.py              # E-commerce
python generate_retail_data.py       # Inventory
python generate_clinical_data.py     # Clinical trial
```

### NLP Project
```bash
cd "Portfolio creation/nlp-sentiment-analysis/06-nlp-sentiment-analysis"
python scripts/train_classical_ml.py
```

### Jupyter Notebooks
Notebooks are in `data-analysis-portfolio/data-analysis-portfolio/*/`:
- `customer_churn_eda.ipynb` (Project 1)
- `churn_classification_model.ipynb` (Project 2)
- `house_price_regression.ipynb` (Project 3)
- `notebooks/etl_pipeline_demo.ipynb` (Project 4)
- `notebooks/energy_forecasting.ipynb` (Project 5)

## Architecture Notes

### Dashboard Projects
All three dashboards follow the same pattern:
- Single main file (`*_dashboard.py`) using Streamlit
- Data generator script for reproducible sample data
- Plotly for interactive visualizations
- PDF report export capability (e-commerce dashboard)

### Data Analysis Portfolio (Projects 1-5)
- Jupyter notebooks are the primary deliverable
- `src/` directories contain reusable Python modules
- Project 4 (ETL) has a full modular architecture: `DataLoader`, `DataValidator`, `DataTransformer`, `PipelineOrchestrator`
- Project 5 (Time Series) uses multiple forecasting approaches: ARIMA, SARIMA, Prophet, LightGBM, XGBoost, LSTM

### NLP Sentiment Analysis (Project 6)
- Progressive approach: Classical ML → Deep Learning → Transformers
- `src/preprocessing.py` handles text cleaning
- `scripts/` contains training scripts
- Uses synthetic IMDB-like data

## Key Dependencies by Project Type

**Dashboards**: streamlit, pandas, plotly, numpy, scikit-learn
**ML Notebooks**: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn
**Time Series**: prophet, tensorflow/keras (LSTM), statsmodels
**NLP**: nltk, torch, transformers, gradio

## Data Management

- Large datasets are `.gitignore`d and generated locally via scripts
- Sample data kept small (< 10 MB) for demos
- Each project has a `data/` directory with `raw/`, `processed/`, and sometimes `sample/` subdirectories

## Planning Document

`PORTFOLIO_EXTENSION_PLAN.md` contains detailed implementation plans for projects 6-9 (NLP, RAG, Neuroscience CV, Astrophysics CV). Reference this when extending the portfolio.
