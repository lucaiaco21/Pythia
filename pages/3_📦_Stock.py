import streamlit as st
import pandas as pd


@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is None:
        st.info("Upload a file")
        st.stop()


df = load_data(uploaded_file)
with st.expander("Data Preview"):
    st.dataframe(df)


cafeterias = df['restaurant'].unique().tolist()


col1, col2 = st.columns(2)

with col1:
    selected = st.selectbox("Select competitor cafeteria:", cafeterias)
with col2:
    compare_all = st.button("Gain insights on all competitor", type="primary", use_container_width=True)


if compare_all: 
    st.subheader("Analysis of all the positive reviews of all your competitor to check common trends across all restaurants")
    st.caption(f"Analyzing {len(df)} reviews across {len(cafeterias)}")
