from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from _functions.cleaning import extract_clean, extract_states, combine_sources, encoding, load_to_db

# Define the DAG
default_args = {
    "owner": "data_engineering_team",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 0,
}

with DAG(
    dag_id = 'sales_dag',
    schedule_interval = '@once', # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args = default_args,
    tags = ['pipeline', 'etl', 'sales'],
)as dag:
    # Define the tasks
    extract_clean = PythonOperator(
        task_id = 'extract_clean',
        python_callable = extract_clean,
        op_kwargs = {
            'input_csv': '/opt/airflow/data/fintech_data_32_52_1830.csv',
            'output_parquet': '/opt/airflow/data/df_cleaned.parquet'
        }
    )

    extract_states = PythonOperator(
        task_id = 'extract_states',
        python_callable =  extract_states,
        op_kwargs = {
            'url': 'https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971',
            'output_parquet': '/opt/airflow/data/states_df.parquet'
        }
    )
    combine_sources = PythonOperator(
        task_id = 'combine_sources',
        python_callable =  combine_sources,
        op_kwargs = {
            'fintech_clean_parquet': '/opt/airflow/data/df_cleaned.parquet',
            'fintech_states_parquet': '/opt/airflow/data/states_df.parquet',
            'output_parquet': '/opt/airflow/data/combined_df.parquet'
        }
    )
    encoding = PythonOperator(
        task_id = 'encoding',
        python_callable =  encoding,
        op_kwargs = {
            'input_parquet': '/opt/airflow/data/df_cleaned.parquet',
            'output_parquet': '/opt/airflow/data/df_encoded.parquet'
        }
    )
    load_to_db = PythonOperator(
        task_id = 'load_to_db',
        python_callable = load_to_db,
        op_kwargs = {
            'cleaned_df_path': '/opt/airflow/data/combined_df.parquet',
            'cleaned_table': 'fintech_cleaned',
            'postgres_opt': {
                'user': 'root',
                'password': 'root',
                'host': 'pgdatabase',
                'port': 5432,
                'db': 'data_engineering'
            }
        }
    )

    # Define the task dependencies
    extract_clean >> combine_sources
    extract_states >> combine_sources
    combine_sources >> encoding >> load_to_db