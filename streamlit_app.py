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
        st.write("### Univariate Analysis")
        feature = st.selectbox("Select variable to plot:", df.columns)

        if np.issubdtype(df[feature].dtype, np.number):
            st.write("### Numerical Variable Options")
            plot_type = st.selectbox("Select plot type:", ["Histogram", "Box Plot"])
            plt.figure(figsize=(10, 6))

            if plot_type == "Histogram":
                sns.histplot(df[feature], kde=True)
                plt.title(f'Histogram of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')

            elif plot_type == "Box Plot":
                sns.boxplot(y=df[feature])
                plt.title(f'Boxplot of {feature}')

        else:
            st.write("### Categorical Variable Options")
            plot_type = st.selectbox("Select plot type:", ["Bar Chart", "Pie Chart"])
            plt.figure(figsize=(10, 6))

            if plot_type == "Bar Chart":
                df[feature].value_counts().plot(kind='bar')
                plt.title(f'Bar Chart of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Count')

            elif plot_type == "Pie Chart":
                df[feature].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {feature}')

        st.pyplot(plt)

        if st.button("Download Plot as JPG"):
            valid_feature_name = generate_valid_filename(feature)
            buf = save_plot_as_jpg(plt.gcf())
            st.download_button(
                label="Download JPG",
                data=buf,
                file_name=f"{valid_feature_name}_plot.jpg",
                mime="image/jpeg",
                key=str(uuid.uuid4())
            )

    if analysis_option == "Bivariate Analysis":
        st.write("### Bivariate Analysis")
        x_axis = st.selectbox("Select X variable:", df.columns)
        y_axis = st.selectbox("Select Y variable:", df.columns, index=1)

        x_is_numeric = np.issubdtype(df[x_axis].dtype, np.number)
        y_is_numeric = np.issubdtype(df[y_axis].dtype, np.number)

        if x_is_numeric and y_is_numeric:
            st.write("### Two Numerical Variables Options")
            plot_type = st.selectbox("Select plot type:", ["Scatter Plot", "Line Graph", "Area Chart"])
            plt.figure(figsize=(10, 6))

            if plot_type == "Scatter Plot":
                sns.scatterplot(data=df, x=x_axis, y=y_axis)
                plt.title(f'Scatter Plot of {y_axis} vs {x_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Line Graph":
                plt.plot(df[x_axis], df[y_axis])
                plt.title(f'Line Graph of {y_axis} vs {x_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Area Chart":
                df[[x_axis, y_axis]].plot(kind='area')
                plt.title(f'Area Chart of {y_axis} vs {x_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            st.pyplot(plt)

        elif x_is_numeric and not y_is_numeric:
            st.write("### One Numerical and One Categorical Variable Options")
            plt.figure(figsize=(10, 6))
            grouped_data = df.groupby(y_axis)[x_axis].mean().sort_values()
            grouped_data.plot(kind='bar')
            plt.title(f'Bar Chart of {x_axis} by {y_axis}')
            plt.xlabel(y_axis)
            plt.ylabel(x_axis)
            st.pyplot(plt)

        elif not x_is_numeric and not y_is_numeric:
            st.write("### Two Categorical Variables Options")
            plt.figure(figsize=(10, 6))
            mosaic(df, [x_axis, y_axis])
            plt.title(f'Mosaic Plot of {x_axis} and {y_axis}')
            st.pyplot(plt)
    
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



















