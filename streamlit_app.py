pip install seaborn matplotlib
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #FFFFE0;  /* background */
    }
    .stApp {
        color: #000000;  /* Black text color */
        font-family: 'Times New Roman';  /* Font style */
    }
    h1, h2, h3 {
        color: #3CB371;  /* Green color for headers */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hàm để phân tích dữ liệu
def generate_analysis(feature, data):
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

    mean_value = data[feature].mean()
    std_value = data[feature].std()
    median_value = data[feature].median()
    trend = "increasing" if data[feature].iloc[-1] > data[feature].iloc[0] else "decreasing"
    
    description = [
        f"The mean of {feature} is {mean_value:.2f}, with a standard deviation of {std_value:.2f}.",
        f"The median value of {feature} is {median_value:.2f}.",
        f"The trend is {trend} over the selected period.",
        f"This indicates that {feature} has shown a {trend} trend recently."
    ]

    return " ".join(description)

# App title
st.title("EDA TOOL")

# Navigation menu
menu = ["About Us", "Upload Your Data", "Contact Us"]
choice = st.sidebar.selectbox("Select feature", menu)

# Upload Data section
if choice == "Upload Your Data":
    st.subheader("Upload Data File")
    uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read data from the uploaded file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, encoding='windows-1252')
        else:
            data = pd.read_excel(uploaded_file, engine='openpyxl')
        
        st.write("Data has been uploaded:")
        st.dataframe(data)

        # Sidebar for analysis options
        st.sidebar.header("Select Analysis Option")
        analysis_option = st.sidebar.selectbox("Choose analysis type", [
            "Summary Statistics",
            "Plot Feature Distributions",
            "Filter Rows",
            "Compare Two Variables",
            "AI Analysis"
        ])

        # Summary Statistics
        if analysis_option == "Summary Statistics":
            st.subheader("Summary Statistics")
            st.write(data.describe())

        # Plot Feature Distributions
        elif analysis_option == "Plot Feature Distributions":
            st.subheader("Plot Feature Distributions")
            feature = st.selectbox("Select feature for distribution plot:", data.columns)

            plt.figure(figsize=(10, 6))
            sns.histplot(data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            st.pyplot(plt)

        # Filter Rows
        elif analysis_option == "Filter Rows":
            st.subheader("Filter Rows by Value Range")
            filter_column = st.selectbox("Select column to filter", data.select_dtypes(include=['float64', 'int64']).columns)
            min_value = st.number_input(f"Minimum value for {filter_column}:", value=float(data[filter_column].min()))
            max_value = st.number_input(f"Maximum value for {filter_column}:", value=float(data[filter_column].max()))

            filtered_data = data[(data[filter_column] >= min_value) & (data[filter_column] <= max_value)]
            st.write("Filtered Data:")
            st.dataframe(filtered_data)

        # Compare Two Variables
        elif analysis_option == "Compare Two Variables":
            st.subheader("Compare Two Variables")
            x_axis = st.selectbox('Select X variable:', data.columns)
            y_axis = st.selectbox('Select Y variable:', data.columns)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=x_axis, y=y_axis)
            plt.title(f'Relationship between {x_axis} and {y_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.grid()
            st.pyplot(plt)

        # AI Analysis
        elif analysis_option == "AI Analysis":
            st.subheader("AI Analysis")
            feature = st.selectbox("Select feature for analysis:", data.columns)
            analysis = generate_analysis(feature, data)
            st.write(analysis)

# Contact Us section
elif choice == "Contact Us":
    st.subheader("Contact Information")
    st.write("If you have any questions, please contact us via email: support@example.com")
