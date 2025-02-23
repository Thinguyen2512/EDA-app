import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from io import StringIO

def clean_and_prepare_data(df, x_cols, y_col, cat_list):
    try:
        X = df[x_cols]
        for col in x_cols:
            if col in cat_list:
                X = pd.get_dummies(X, columns=[col], drop_first=True)
        y = df[y_col]
        combined_data = pd.concat([X, y], axis=1).dropna()
        X_cleaned = combined_data.iloc[:, :-1]
        y_cleaned = combined_data.iloc[:, -1]
        return X_cleaned, y_cleaned
    except Exception as e:
        st.error(f"Error cleaning and preparing data: {e}")
        return None, None

st.title("Exploratory Data Analysis (EDA) Tool")

st.sidebar.header("Upload Data")
st.subheader("Upload Data File")
uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='windows-1252')
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    st.write("Data has been uploaded:")
    st.dataframe(df)
    
    if st.sidebar.checkbox("Show Data Dictionary"):
        st.write("### Data Dictionary")
        st.write(pd.DataFrame({"Column Name": df.columns, "Data Type": df.dtypes.values}))
    
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
            
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    
    if st.sidebar.checkbox("Show Multiple Linear Regression"):
        st.write("### Multiple Linear Regression")
        num_list = df.select_dtypes(include=["number"]).columns.tolist()
        cat_list = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        if cat_list:
            st.sidebar.header("Categorical Variables")
            selected_cat_cols = st.sidebar.multiselect("Select columns to create dummy variables:", cat_list)
            if selected_cat_cols:
                df = pd.get_dummies(df, columns=selected_cat_cols, drop_first=True)
        
        x_cols = st.multiselect("Select Independent Variables (X):", num_list + cat_list)
        y_col = st.selectbox("Select Dependent Variable (Y):", ["Select Variable"] + num_list)
        
        if x_cols and y_col != "Select Variable":
            try:
                X, y_values = clean_and_prepare_data(df, x_cols, y_col, cat_list)
                if X is not None and y_values is not None:
                    X = sm.add_constant(X)
                    model = sm.OLS(y_values, X).fit()
                    st.write(model.summary())
                    
                    st.write("### Residual Plot")
                    residuals = model.resid
                    fig, ax = plt.subplots()
                    sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, ax=ax, line_kws={"color": "red"})
                    ax.set_xlabel("Fitted Values")
                    ax.set_ylabel("Residuals")
                    ax.set_title("Residuals vs Fitted")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during regression analysis: {e}")
















