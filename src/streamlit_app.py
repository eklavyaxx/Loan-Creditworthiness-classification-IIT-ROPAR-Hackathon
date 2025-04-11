# streamlit_app.py

import streamlit as st
import pandas as pd
from Archives.streamlit_pipeline import process_and_train

st.title("Loan Creditworthiness Dashboard")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(df)

    if st.button("Train and Evaluate Models"):
        fig = process_and_train(df)
        st.pyplot(fig)
