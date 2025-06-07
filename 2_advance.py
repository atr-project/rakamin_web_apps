import streamlit as st
import pandas as pd

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose page", ["Home", "Data"])

if page == "Home":
    st.write("Welcome Home!")
elif page == "Data":
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob'],
        'Age': [25, 30]
    })
    st.dataframe(df)
