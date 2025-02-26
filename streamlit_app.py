import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import uuid
from io import StringIO
from statsmodels.graphics.mosaicplot import mosaic

def descriptive_stats(df, feature):
    stats_dict = {
        "Mean": df[feature].mean(),
        "Median": df[feature].median(),
        "Mode": df[feature].mode().values[0],
        "Min": df[feature].min(),
        "Max": df[feature].max(),
        "Range": df[feature].max() - df[feature].min(),
        "Std Dev": df[feature].std(),
        "Q1": df[feature].quantile(0.25),
        "Q3": df[feature].quantile(0.75)
    }
    return stats_dict

def confidence_interval_mean(df, feature, confidence=0.95):
    sample_mean = np.mean(df[feature])
    sample_std = np.std(df[feature], ddof=1)
    n = len(df[feature])
    margin_of_error = stats.t.ppf((1 + confidence) / 2., n - 1) * (sample_std / np.sqrt(n))
    return sample_mean - margin_of_error, sample_mean + margin_of_error

def hypothesis_test_mean(df, feature, pop_mean, alpha=0.05):
    t_stat, p_value = stats.ttest_1samp(df[feature].dropna(), pop_mean)
    return t_stat, p_value, p_value < alpha

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

def generate_valid_filename(name):
    return "_".join(name.split())

def save_plot_as_jpg(fig):
    buf = StringIO()
    fig.savefig(buf, format="jpg")
    buf.seek(0)
    return buf

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

    filter_col = st.sidebar.selectbox("Filter by Column", ["None"] + df.columns.tolist())
    if filter_col != "None":
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            min_val, max_val = st.sidebar.slider(
                f"Select range for {filter_col}",
                float(df[filter_col].min()),
                float(df[filter_col].max()),
                (float(df[filter_col].min()), float(df[filter_col].max()))
            )
            df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
        else:
            unique_values = list(df[filter_col].dropna().unique())
            if unique_values:
                all_selected = st.sidebar.checkbox(f"Select All {filter_col}", value=True)
                if all_selected:
                    selected_values = unique_values
                else:
                    selected_values = st.sidebar.multiselect(
                        f"Select values for {filter_col}",
                        options=unique_values,
                        default=[]
                    )
                if selected_values:
                    df = df[df[filter_col].isin(selected_values)]
            else:
                st.warning(f"No unique values found in column {filter_col}.")
    
    analysis_option = st.sidebar.radio("Choose Analysis Type:", [
        "Data Dictionary", "Univariate Analysis", "Bivariate Analysis", "Linear Regression"
    ])
    
    if analysis_option == "Data Dictionary":
        st.write("### Data Dictionary")

        # Section to upload a Data Dictionary file
        st.subheader("Upload Data Dictionary File")
        dict_file = st.file_uploader("Upload a CSV or Excel file for the Data Dictionary", type=["csv", "xlsx"])
        
        if dict_file is not None:
            if dict_file.name.endswith('.csv'):
                data_dict_df = pd.read_csv(dict_file)
            else:
                data_dict_df = pd.read_excel(dict_file, engine='openpyxl')

            st.write("#### Uploaded Data Dictionary:")
            st.dataframe(data_dict_df)
        else:
            # Prompt for dataset description
            dataset_description = st.text_area(
                "Provide a detailed description of the dataset:",
                "Include details such as what it is, what each row represents, how/when it was collected, who owns it, and conditions of use."
            )

            # Display dataset description if provided
            if dataset_description:
                st.write("#### Dataset Description")
                st.write(dataset_description)

            # Create a DataFrame for the column information
            column_info = []
            for col in df.columns:
                col_description = st.text_area(
                    f"Describe the column '{col}':",
                    "Include details such as data type, meaning, units, and possible value range."
                )
                column_info.append({"Column Name": col, "Data Type": str(df[col].dtype), "Description": col_description})

            # Display the Data Dictionary table
            st.write("#### Column Details")
            st.write(pd.DataFrame(column_info))
    
    if analysis_option == "Univariate Analysis":
        feature = st.selectbox("Select a Column:", df.columns)
        
        if np.issubdtype(df[feature].dtype, np.number):
            st.write("### Numerical Variable Analysis")
            st.write("#### Descriptive Statistics")
            st.json(descriptive_stats(df, feature))
            
            st.write("#### Histogram")
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            st.pyplot(fig)
            
            st.write("#### Boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[feature], ax=ax)
            st.pyplot(fig)
            
            st.write("#### Confidence Interval (95%)")
            ci_lower, ci_upper = confidence_interval_mean(df, feature)
            st.write(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
            
            pop_mean = st.number_input("Enter Population Mean for Hypothesis Test:", value=df[feature].mean())
            t_stat, p_value, reject = hypothesis_test_mean(df, feature, pop_mean)
            st.write(f"T-Statistic: {t_stat:.2f}, P-Value: {p_value:.4f}")
            st.write("Reject Null Hypothesis" if reject else "Fail to Reject Null Hypothesis")
        
        else:
            st.write("### Categorical Variable Analysis")
            st.write("#### Frequency Table")
            st.dataframe(df[feature].value_counts(normalize=True).reset_index())
            
            st.write("#### Bar Chart")
            fig, ax = plt.subplots()
            df[feature].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
            
    if analysis_option == "Bivariate Analysis":
        x_var = st.selectbox("Select X Variable:", df.columns)
        y_var = st.selectbox("Select Y Variable:", df.columns, index=1)
        
        x_is_num = np.issubdtype(df[x_var].dtype, np.number)
        y_is_num = np.issubdtype(df[y_var].dtype, np.number)
        
        if x_is_num and y_is_num:
            st.write("### Numerical vs Numerical")
            st.write("#### Scatter Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
            st.pyplot(fig)
            
            st.write("#### Pearson Correlation")
            corr, p_value = stats.pearsonr(df[x_var].dropna(), df[y_var].dropna())
            st.write(f"Pearson Correlation: {corr:.2f}, P-Value: {p_value:.4f}")
            
        elif x_is_num and not y_is_num:
            st.write("### Numerical vs Categorical")
            st.write("#### Boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[y_var], y=df[x_var], ax=ax)
            st.pyplot(fig)
            
        elif not x_is_num and not y_is_num:
            st.write("### Categorical vs Categorical")
            st.write("#### Mosaic Plot")
            fig, ax = plt.subplots()
            mosaic(df, [x_var, y_var], ax=ax)
            st.pyplot(fig)

    
    if analysis_option == "Linear Regression":
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



















