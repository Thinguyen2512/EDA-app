import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Handle categorical columns
        if cat_cols:
            st.sidebar.header("Categorical Variables")
            selected_cat_cols = st.sidebar.multiselect("Select columns to create dummy variables:", cat_cols)
            if selected_cat_cols:
                df = pd.get_dummies(df, columns=selected_cat_cols, drop_first=True)
        
        x_cols = st.multiselect("Select Independent Variables (X):", numeric_cols)
        y_col = st.selectbox("Select Dependent Variable (Y):", numeric_cols)
        
        if x_cols and y_col:
            X = df[x_cols].dropna()
            y = df[y_col].dropna()
            
            # Find common index to ensure both X and y have matching rows
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # Add constant (intercept) to independent variables
            X = sm.add_constant(X)
            
            # Fit the model
            model = sm.OLS(y, X).fit()
            st.write("### Regression Results")
            st.write(pd.DataFrame({
                "Coefficients": model.params,
                "P-Values": model.pvalues,
                "T-Statistics": model.tvalues,
                "Confidence Interval (2.5%)": model.conf_int()[0],
                "Confidence Interval (97.5%)": model.conf_int()[1],
            }))
            st.write("### Model Performance")
            st.write(f"R-squared: {model.rsquared:.4f}")
            st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            st.write(f"F-statistic: {model.fvalue:.2f}")
            st.write(f"F-statistic p-value: {model.f_pvalue:.4e}")
            
            # Variance Inflation Factor (VIF)
            st.write("### Multicollinearity Check (VIF)")
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            st.table(vif_data)
            
            # Residual Plot
            st.write("### Residual Plot")
            residuals = model.resid
            fig, ax = plt.subplots()
            sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, ax=ax, line_kws={"color": "red"})
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")
            st.pyplot(fig)















