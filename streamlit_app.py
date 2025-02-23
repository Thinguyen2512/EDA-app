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
    
    st.sidebar.header("Sidebar for Analysis Options")
    
    if st.sidebar.checkbox("Show Data Dictionary"):
        st.write("### Data Dictionary")
        st.write(pd.DataFrame({"Column Name": df.columns, "Data Type": df.dtypes.values}))
    
    if st.sidebar.checkbox("Show Univariate Analysis"):
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
    
    if st.sidebar.checkbox("Show Bivariate Analysis"):
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

    if st.sidebar.checkbox("Show Multiple Linear Regression"):
        st.write("### Multiple Linear Regression")



















