import streamlit as st
import pandas as pd
from src.data_processing import clean_data, split_data
from src.model_training import get_models, train_and_evaluate_models
from src.visualisation import plot_model_comparison, plot_confusion_matrices

# Page configuration
st.set_page_config(page_title="Loan Risk Analyzer", layout="wide")

# Theme toggle in sidebar (default to dark)
theme_mode = st.sidebar.radio("Select Theme", ("🌙 Dark", "🌞 Light"), index=0)

# Define theme-specific colors
if theme_mode == "🌙 Dark":
    bg_color = "#0f1117"
    text_color = "#FFFFFF"
    button_color = "#444"
    success_text_color = "#FFFFFF"
else:
    bg_color = "#F8F4E1"
    text_color = "#000000"
    button_color = "#333333"
    success_text_color = "#000000"

# Inject custom CSS for theming
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stButton>button {{
            background-color: {button_color};
            color: white;
            font-size: 18px;
            padding: 0.5rem 1.5rem;
            border-radius: 12px;
            border: none;
        }}
        header, .block-container {{
            background-color: {bg_color};
            color: {text_color};
        }}
        [data-testid="stAlertContentSuccess"] p {{
            color: {success_text_color} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏦 Loan Creditworthiness Dashboard")
st.markdown("Upload a preprocessed dataset (like `final.csv`), train five classifiers, and visualize their performance to identify high-risk applicants 💡")

# Load default dataset
default_path = "data/Preprocessed/final.csv"
try:
    df = pd.read_csv(default_path)
    st.success("✅ Default dataset loaded successfully!")
except Exception as e:
    st.error(f"Failed to load default dataset: {e}")
    df = None

# Train models and display results
if df is not None and st.button("🚀 Train Models"):
    with st.spinner("Training in progress... Please wait ⏳"):
        df_cleaned = clean_data(df)
        X_train, X_test, y_train, y_test = split_data(df_cleaned, target_column='high_risk_applicant')
        models = get_models()
        results, predictions = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
        figs = plot_model_comparison(results)
        confusion_figs = plot_confusion_matrices(predictions)

    st.success("🎉 Training completed!")

    st.subheader("📊 Model Performance")
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for j, fig in enumerate(figs[i:i+2]):
            cols[j].pyplot(fig)

    st.subheader("📉 Confusion Matrices")
    for i in range(0, len(confusion_figs), 2):
        cols = st.columns(2)
        for j, fig in enumerate(confusion_figs[i:i+2]):
            cols[j].pyplot(fig)
