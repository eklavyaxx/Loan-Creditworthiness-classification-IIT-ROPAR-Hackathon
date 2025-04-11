import streamlit as st
import matplotlib.pyplot as plt
from Archives.data_processing import clean_data, split_data
from Archives.model_training import get_models, train_and_evaluate_models
from Archives.visualisation import plot_model_comparison



def process_and_train(df, target_column='high_risk_applicant'):
    print("Cleaning data...")
    df = clean_data(df)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    print("Getting models...")
    models = get_models()
    
    print("Training models...")
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
    
    print("Plotting results...")
    figs = plot_model_comparison(results)
    print("Done!")
    return figs
