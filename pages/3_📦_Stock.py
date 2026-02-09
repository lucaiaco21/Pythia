import streamlit as st
import pandas as pd
st.title("_📊_Stock Page")


@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is None:
        st.info("Upload a file", icon='i')
        st.stop()


df = load_data(uploaded_file)
with st.expander("Data Preview"):
    st.dataframe(df)
