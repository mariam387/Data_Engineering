#  Fintech Data Engineering Pipeline Project

This project is a comprehensive data engineering pipeline built to process and analyze a real-world fintech dataset from Kaggle. It is structured into two milestones:

- **Milestone 1**: Exploratory Data Analysis (EDA), Data Cleaning, and Feature Engineering in a Jupyter Notebook.
- **Milestone 2**: Production-grade pipeline using Docker, Apache Airflow, and Apache Superset for scalable, automated processing and visualization.

---

## Project Structure
```
Fintech-Pipeline/
├── Milestone1/
│   ├── fintech_raw.csv
│   ├── fintech_clean.csv
│   ├── lookup_table.csv
├── Milestone2/
│   ├── dags/
│   │   ├── extract_clean.py
│   │   ├── extract_states.py
│   │   ├── combine_sources.py
│   │   ├── encoding.py
│   │   └── load_to_db.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── superset/
│   │   └── dashboard_screenshots/
│   └── README.md
└── requirements.txt
```

## Milestone 1: Data Cleaning & Feature Engineering

Implemented in a Jupyter notebook with clean, modular functions.

### Tasks Performed

- **EDA**: Answered 5 exploratory questions with visualizations and insights.
- **Data Cleaning**:
  - Fixed column names and types.
  - Removed duplicates and handled missing data using domain-aware strategies.
  - Justified missing value types (MCAR, MAR, MNAR).
- **Imputation Strategy**:
  - Imputed with constants, means, or column-based logic.
  - Used a dynamic and reversible lookup table for label encoding and imputation.
- **Feature Engineering**:
  - Created custom features: `Month Number`, `Installment Per Month`, `Letter Grade`, etc.
- **Outlier Detection**: Identified and handled anomalies to improve data quality.
- **Bonus**: Scraped and mapped state codes to full names using an API/web scraping logic.

### Outputs

- `fintech_clean.csv`: Cleaned dataset
- `fintech_clean.parquet`: Cleaned dataset
- `lookup_table.csv`: Reversible encoding and imputation mapping

---

## Milestone 2: Scalable Pipeline with Docker, Airflow & Superset

Converted Milestone 1 into a production-grade pipeline for automated execution and visualization.

### Tools & Technologies

- **Docker**: Containerized the entire environment for reproducibility.
- **Apache Airflow**: Orchestrated data processing steps via DAGs.
- **PostgreSQL**: Stored final transformed data.
- **Apache Superset**: Created interactive dashboards for business intelligence.

### ETL Pipeline Tasks (via Airflow DAG)

1. `extract_clean.py`: Cleans and saves data to `fintech_clean.parquet`
2. `extract_states.py`: Scrapes state data and saves to `fintech_states.parquet`
3. `combine_sources.py`: Joins cleaned data with states info
4. `encoding.py`: Encodes features and stores final dataset as `fintech_encoded.parquet`
5. `load_to_db.py`: Loads the final dataset into PostgreSQL

### Superset Dashboard

- **Basic Stats**: Total loans, average loan amounts, state with highest loan volume
- **Time Series**: Loans over time by amount and frequency
- **Purpose Analysis**: Loan purpose vs amount vs status
- **Custom Insights**:
  - Loan grade distributions
  - Income-loan relationships across states
  - Top 5 states by average loan amounts
---

## Requirements

- Python 3.10+
- Docker & Docker Compose
- Apache Airflow 2+
- Apache Superset
- PostgreSQL
- Pandas, NumPy, Matplotlib, Seaborn

Install all dependencies via:

```bash
pip install -r requirements.txt

