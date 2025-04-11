import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file):
    return pd.read_csv(file)

def clean_data(df):
    if 'Loan_Amount_Requested' in df.columns:
        df['Loan_Amount_Requested'] = df['Loan_Amount_Requested'].str.replace(',', '').astype(float)
    
    if 'Risk_Score' in df.columns:
        df['Risk_Score'] = df['Risk_Score'].fillna(df['Risk_Score'].median())
    
    df.drop(columns=['Loan_ID'], inplace=True, errors='ignore')
    return df


def split_data(df, target_column='Loan_Status'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
