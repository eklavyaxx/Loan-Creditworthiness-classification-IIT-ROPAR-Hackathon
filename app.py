import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Archives.data_processing import clean_data, split_data
from Archives.model_training import get_models, train_and_evaluate_models
from Archives.visualisation import plot_model_comparison, plot_confusion_matrices

# Page config
st.set_page_config(page_title="Loan Risk Analyzer", layout="wide")

# Theme toggle
theme_mode = st.sidebar.radio("Select Theme", ("🌙 Dark", "🌞 Light"), index=0)

if theme_mode == "🌙 Dark":
    bg_color = "#0f1117"
    text_color = "#FFFFFF"
    button_color = "#444"
    success_text_color = "#FFFFFF"
    plotly_template = "plotly_dark"
else:
    bg_color = "#F8F4E1"
    text_color = "#000000"
    button_color = "#333333"
    success_text_color = "#000000"
    plotly_template = "plotly_white"

st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stButton>button {{
            background-color: {button_color};
            color: white;
            font-size: 16px;
            padding: 0.5rem 1.2rem;
            border-radius: 8px;
            border: none;
        }}
        .stAlert.success {{
            background-color: #d4edda;
            color: {success_text_color} !important;
        }}
        div[data-testid="stAlertContainer"] p {{
            color: {success_text_color} !important;
        }}
        h2, h3, h4, h5, h6, .stMarkdown {{
            color: {text_color} !important;
        }}
        .css-10trblm, .stMetric label, .stMetric div {{
            color: {text_color} !important;
        }}
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📁 Upload and Train")

uploaded_file = st.sidebar.file_uploader("Upload a preprocessed dataset (CSV)", type=["csv"])
default_path = "Loan-Creaditworthiness-classification-main/data/Preprocessed/final.csv"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded successfully!")
else:
    try:
        df = pd.read_csv(default_path)
        st.sidebar.success("✅ Default dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load default dataset: {e}")
        df = None

use_best_model = st.sidebar.checkbox("🚀 Use Best Model Automatically")

if df is not None and st.sidebar.button("⚙️ Train Models"):
    with st.spinner("Training in progress... Please wait ⏳"):
        df_cleaned = clean_data(df)
        X_train, X_test, y_train, y_test = split_data(df_cleaned, target_column='high_risk_applicant')
        models = get_models()
        results, predictions = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)

        figs = plot_model_comparison(results)
        confusion_figs = plot_confusion_matrices(predictions)

        model_scores = {name: score['Accuracy'] for name, score in results.items()}
        model_dict = {name: predictions[name][1] for name in predictions}
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = model_dict[best_model_name]

        selected_model_name = best_model_name
        selected_model = best_model
        st.success(f"✅ Best model auto-selected: **{best_model_name}** (Accuracy: **{model_scores[best_model_name]:.2f}**)")

    st.success("🎉 Training completed!")

    st.subheader("📊 Model Performance (Bar Graphs)")
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for j, fig in enumerate(figs[i:i+2]):
            cols[j].pyplot(fig)

    st.subheader("📉 Confusion Matrices")
    for i in range(0, len(confusion_figs), 2):
        cols = st.columns(2)
        for j, fig in enumerate(confusion_figs[i:i+2]):
            cols[j].pyplot(fig)

    st.subheader("📊 Visual Comparison of All Models")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    data = {metric: [results[model].get(metric, 0) for model in results] for metric in metrics}
    model_names = list(results.keys())

    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            y=model_names,
            x=data[metric],
            name=metric,
            orientation='h'
        ))

    fig.update_layout(
        barmode='group',
        title="Model Performance Comparison",
        xaxis_title="Score",
        yaxis_title="Models",
        template=plotly_template,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<h3 style='color:{text_color};'>📈 Real-Time Performance Comparison</h3>", unsafe_allow_html=True)
    for name, score_dict in results.items():
        cols = st.columns(4)
        cols[0].markdown(f"<div style='color:{text_color}; font-weight:bold;'>{name}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='color:{text_color};'>✅ Accuracy<br><span style='font-size:28px;'><b>{score_dict['Accuracy']:.3f}</b></span></div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div style='color:{text_color};'>🎯 Precision<br><span style='font-size:28px;'><b>{score_dict['Precision']:.3f}</b></span></div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div style='color:{text_color};'>📊 Recall<br><span style='font-size:28px;'><b>{score_dict['Recall']:.3f}</b></span></div>", unsafe_allow_html=True)
