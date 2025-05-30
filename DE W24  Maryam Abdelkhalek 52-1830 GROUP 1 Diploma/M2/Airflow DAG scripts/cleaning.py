import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
import requests

#--- helper methods---
lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Imputed/Encoded'])

def drop_columns(df, columns):
    df.drop(columns, axis=1, inplace=True)

def clean_column_names(df):
    df.columns = df.columns.str.replace(' ', '_').str.lower()

def standardize_values(df, column, replacements):
    df[column] = df[column].replace(replacements).str.lower()

def convert_to_numeric(df, column, regex_pattern):
    df[column] = df[column].str.extract(regex_pattern).fillna(-1).astype(int)

def impute_multivariant_mode(df, col_to_impute, group_col):
    df[col_to_impute].fillna(df.groupby(group_col)[col_to_impute].transform(lambda x: x.mode()[0]), inplace=True)

def impute_multivariant_mean(df, col_to_impute, group_col):
    df[col_to_impute].fillna(df.groupby(group_col)[col_to_impute].transform('mean'), inplace=True)

def handle_outliers_with_log(df, columns):
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = np.log1p(df[col])

def update_lookup_table(column, original, imputed_or_encoded):
    """
    Updates the global lookup table with a record of imputations or encodings
    and saves it to a Parquet file.

    Parameters:
    - column: The column name being updated.
    - original: The original values before imputation/encoding.
    - imputed_or_encoded: The new imputed or encoded values.
    - output_parquet: The file path where the lookup table will be saved.
    """
    global lookup_table
    new_entries = pd.DataFrame({
        'Column': [column],  # Ensure these are lists to avoid shape issues
        'Original': [original],
        'Imputed/Encoded': [imputed_or_encoded]
    })
    
    # Concatenate new entries to the global lookup table
    lookup_table = pd.concat([lookup_table, new_entries], ignore_index=True)
    


def feature_engineer(df):
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    
    df['issue_month'] = df['issue_date'].dt.month
    df['issue_year'] = df['issue_date'].dt.year
    df['issue_day'] = df['issue_date'].dt.day
    df.drop('issue_date', axis=1, inplace=True)
    
    df['salary_can_cover'] = df['annual_inc'] >= df['funded_amount']
    
    bins = [0, 5, 10, 15, 20, 25, 30, 35]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df['letter_grade'] = pd.cut(df['letter_grade'], bins=bins, labels=labels, include_lowest=True)
    
    def calculate_installment(row):
        P = row['loan_amount']  # Loan principal
        r = row['int_rate'] / 12   # Monthly interest rate (annual interest rate divided by 12)
        n = row['term']  # Number of payments (loan term in months)
        
        if r == 0:  # If interest rate is 0, the formula simplifies to P / n
            return P / n
        
        # Monthly installment formula
        M = P * r * (1 + r)**n / ((1 + r)**n - 1)
        return M

    df['installment_per_month'] = df.apply(calculate_installment, axis=1)
    
    return df

#---- clean method----

def extract_clean(input_csv, output_parquet, index_col=None):
    global lookup_table
    df = pd.read_csv(input_csv)
    drop_columns(df, ['Customer Id'])
    clean_column_names(df)
    df = df.rename(columns={'grade': 'letter_grade'})
    standardize_values(df, 'type', {'INDIVIDUAL': 'Individual', 'Joint App': 'JOINT'})
    impute_multivariant_mode(df, 'description', 'purpose')
    impute_multivariant_mean(df, 'int_rate', 'letter_grade')

    original_values = df['annual_inc_joint'].copy()
    df['annual_inc_joint'] = df['annual_inc_joint'].fillna(-1)
    update_lookup_table(
        column=['annual_inc_joint'] * len(original_values),
        original=original_values,
        imputed_or_encoded=df['annual_inc_joint']
    )
    original_values = df['emp_title'].copy()
    df['emp_title'] = df['emp_title'].fillna(np.nan)
    update_lookup_table(
        column=['emp_title'] * len(original_values),
        original=original_values,
        imputed_or_encoded=df['emp_title']
    )

    original_values = df['emp_length'].copy()
    df['emp_length'] = df['emp_length'].str.extract('(\d+)').fillna(-1).astype(int)
    update_lookup_table(
        column=['emp_length'] * len(original_values),
        original=original_values,
        imputed_or_encoded=df['emp_length']
    )

    original_values = df['term'].copy()
    df['term'] = df['term'].str.extract('(\d+)').astype(int)
    update_lookup_table(
        column=['term'] * len(original_values),
        original=original_values,
        imputed_or_encoded=df['term']
    )
    df= feature_engineer(df)
    columns_with_outliers = ['annual_inc', 'avg_cur_bal', 'tot_cur_bal', 'int_rate', 'letter_grade']
    handle_outliers_with_log(df, columns_with_outliers)
    df.to_parquet(output_parquet, index=False)
    return df

#-------- extract_states method----
def extract_states(url: str, output_parquet: str):

    page = requests.get(url)
    
    soup = BeautifulSoup(page.content, 'html.parser')
    
    table = soup.find('table')
    
    world_titles = table.find_all('th')
    world_table_titles = [title.text.strip() for title in world_titles]
    
    world_table_titles_cleaned = world_table_titles[1:4]
    
    df_states = pd.DataFrame(columns=world_table_titles_cleaned)
    
    column_data = table.find_all('tr')
    for row in column_data[1:]:  # Skip header row
        row_data = row.find_all('td')
        individual_row_data = [data.text.strip() for data in row_data]
        df_states.loc[len(df_states)] = individual_row_data

    df_states.to_parquet(output_parquet, index=False)
    
    return df_states


#-------- combine_sources method----

def combine_sources(fintech_clean_parquet: str, fintech_states_parquet: str, output_parquet: str):

    df_cleaned = pd.read_parquet(fintech_clean_parquet)
    df_states = pd.read_parquet(fintech_states_parquet)
    
    state_mapping = dict(zip(df_states['Alpha code'], df_states['State']))
    
    df_cleaned['state_name'] = df_cleaned['state'].map(state_mapping)
    
    df_cleaned.to_parquet(output_parquet, index=False)
    
    return df_cleaned
#-------- encoded method----

def encoding(input_parquet: str, output_parquet: str):

    df = pd.read_parquet(input_parquet)
    
    label_columns = ['verification_status', 'loan_status', 'letter_grade']
    
    le = LabelEncoder()
    
    for column in label_columns:
        original_values = df[column].copy()
        
        df[column] = le.fit_transform(df[column])
        
        update_lookup_table(
            column=[column] * len(original_values),
            original=original_values,
            imputed_or_encoded=df[column]
        )
    
    one_hot_columns = ['emp_title', 'home_ownership', 'zip_code', 'addr_state', 'state', 'type', 'purpose', 'description']
    
    df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)
   
    df.to_parquet(output_parquet, index=False)
    
    return df
#---- load to db ---

import pandas as pd
from sqlalchemy import create_engine

def load_to_db(postgres_opt: dict, cleaned_df_path: str,
               cleaned_table: str):
    # Load the DataFrames from the specified file paths
    cleaned_df = pd.read_parquet(cleaned_df_path) 
    
    # Extracting database connection details
    user, password, host, port, db = postgres_opt.values()
    
    # Creating the engine to connect to the PostgreSQL database
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}')
    
    # Load cleaned_df into PostgreSQL
    cleaned_df.to_sql(cleaned_table, engine, if_exists='replace', index=False)
    print(f"{cleaned_df_path} has been loaded into the database table: {cleaned_table}")
    


    

