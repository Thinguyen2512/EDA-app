import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from io import StringIO

st.title("Exploratory Data Analysis (EDA) Tool")

# Data Uploading
st.sidebar.header("Upload Data")
st.subheader("Upload Data File")
uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read data from the uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='windows-1252')
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
    st.write("Data has been uploaded:")
    st.dataframe(df)
    
    # Data Dictionary
    if st.sidebar.checkbox("Show Data Dictionary"):
        st.write("### Data Dictionary")
        st.write(pd.DataFrame({"Column Name": df.columns, "Data Type": df.dtypes.values}))
    
    # Univariate Analysis
    if st.sidebar.checkbox("Show Univariate Analysis"):
        st.write("### Univariate Analysis")
        selected_column = st.selectbox("Select a column", df.columns)
        if df[selected_column].dtype in ['int64', 'float64']:
            st.write(df[selected_column].describe())
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(df[selected_column], kde=True, ax=ax[0])
            ax[0].set_title("Histogram")
            sns.boxplot(y=df[selected_column], ax=ax[1])
            ax[1].set_title("Boxplot")
            st.pyplot(fig)
    
    # Bivariate Analysis
    if st.sidebar.checkbox("Show Bivariate Analysis"):
        st.write("### Bivariate Analysis")
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select X variable", df.columns)
        y_col = col2.selectbox("Select Y variable", df.columns)
        if x_col and y_col:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            st.pyplot(fig)
            
            # Correlation Heatmap
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    
    # Multiple Linear Regression
    if st.sidebar.checkbox("Show Multiple Linear Regression"):
        st.write("### Multiple Linear Regression")
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Feature Variables", df.columns.drop(target))
        if target and features:
            X = df[features]
            X = sm.add_constant(X)
            y = df[target]
            model = sm.OLS(y, X).fit()
            st.write(model.summary())















