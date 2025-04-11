from data_processing import load_data, clean_data, split_data
from model_training import get_models, train_and_evaluate_models
from visualisation import plot_model_comparison
import matplotlib.pyplot as plt

# Step 1: Load and clean data
df = load_data('Loan-Creaditworthiness-classification-main/data/loan.csv')
df = clean_data(df)

# 👉 Check column names
print(df.columns)

# Step 2: Split into train/test
X_train, X_test, y_train, y_test = split_data(df, target_column='high_risk_applicant')

