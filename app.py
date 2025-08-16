import streamlit as st
import pandas as pd

st.title("Hello, Streamlit Dashboard!")

# Load dataset
IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
COLUMN_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(IRIS_URL, names=COLUMN_NAMES)

# View dataset
st.header("Step 1: Iris Dataset Loaded!")
st.write("Here is what the data looks like:")
st.write(df.head())

# Exploring the dataset - EDA
st.subheader("A Quick Look at Dataset Shape and Structure")
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")

# Data types & missing values
st.subheader("Data Types and Null Values Info")
st.write(df.dtypes)
st.write(df.isnull().sum())

# descriptive statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# value counts for species
st.subheader("Species Distribution in Iris Dataset")
st.write(df["species"].value_counts())

# make it interactive : expand/collapse sections
with st.expander("Show full dataset"):
    st.write(df)

# display a datatable
st.subheader("Iris Dataset Table View")
st.dataframe(df[["sepal_length", "sepal_width", "species"]])


import numpy as np

# Step 1: Add a Slider to Select Sample Size
sample_size = st.slider(
    "Select number of samples to display:", min_value=10, max_value=len(df), value=20
)
st.line_chart(df[["sepal_length", "sepal_width"]].head(sample_size))

# Step 2: Let Users Pick Features Dynamically with selectbox
feature = st.selectbox(
    "Choose a feature to plot:",
    ["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
st.line_chart(df[feature].head(sample_size))

# filter data by species
species = st.radio("Select species:", df["species"].unique())
filtered_df = df[df["species"] == species]
st.dataframe(filtered_df)

# multi select for multiple species
species_options = st.multiselect(
    "Select species:", df["species"].unique(), default=df["species"].unique()
)
filtered_df = df[df["species"].isin(species_options)]
st.dataframe(filtered_df)

# Step 4: Organise Interactive Elements in a Sidebar
with st.sidebar:
    st.header("Controls")
    sample_size = st.slider("Number of samples:", 10, len(df), 20)
    feature = st.selectbox("Feature:", df.columns[:-1])
    species = st.multiselect(
        "Species:", df["species"].unique(), default=df["species"].unique()
    )

# Add Titles, Headers, Subheaders and Markdown
st.title("🌸 Iris Dataset Dashboard")
st.header("A quick tour of flower data analysis")
st.subheader("Visual Insights and Metrics")
st.markdown("Hand-crafted with Streamlit.")


# 2. Page-Wide Customization with st.set_page_config
st.set_page_config(
    page_title="Iris Dataset Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 3. Layout: Columns and Containers for Clean Organization
col1, col2 = st.columns(2)
with col1:
    st.write("## Sepal Features")
    st.line_chart(df[["sepal_length", "sepal_width"]])
with col2:
    st.write("## Petal Features")
    st.area_chart(df[["petal_length", "petal_width"]])
