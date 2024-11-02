import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates, andrews_curves

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

# Function to generate analysis description
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
            "Plot One Variable",
            "Plot Two Variables",
            "Plot Three Variables",
            "Plot More Than Three Variables",
            "AI Analysis"
        ])

        # Summary Statistics
        if analysis_option == "Summary Statistics":
            st.subheader("Summary Statistics")
            st.write(data.describe())

        # Plot One Variable
        elif analysis_option == "Plot One Variable":
            st.subheader("Plot One Variable")
            st.write("Select a variable to visualize. Choose a numerical or categorical variable.")
            feature = st.selectbox("Select variable to plot:", data.columns)

            if data[feature].dtype in [np.number, 'float64', 'int64']:
                st.write("### Numerical Variable Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Line Chart",
                    "Histogram",
                    "Box Plot",
                    "Density Plot",
                    "Area Chart",
                    "Dot Plot",
                    "Frequency Polygon"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Line Chart":
                    plt.plot(data[feature])
                    plt.title(f'Line Chart of {feature}')
                    plt.xlabel('Index')
                    plt.ylabel(feature)

                elif plot_type == "Histogram":
                    sns.histplot(data[feature], kde=True)
                    plt.title(f'Histogram of {feature}')
                    plt.xlabel(feature)
                    plt.ylabel('Frequency')

                elif plot_type == "Box Plot":
                    sns.boxplot(y=data[feature])
                    plt.title(f'Boxplot of {feature}')

                elif plot_type == "Density Plot":
                    sns.kdeplot(data[feature])
                    plt.title(f'Density Plot of {feature}')

                elif plot_type == "Area Chart":
                    data[feature].plot(kind='area')
                    plt.title(f'Area Chart of {feature}')
                    plt.xlabel('Index')
                    plt.ylabel(feature)

                elif plot_type == "Dot Plot":
                    plt.plot(data.index, data[feature], 'o')
                    plt.title(f'Dot Plot of {feature}')
                    plt.xlabel('Index')
                    plt.ylabel(feature)

                elif plot_type == "Frequency Polygon":
                    sns.histplot(data[feature], kde=False, bins=30)
                    plt.title(f'Frequency Polygon of {feature}')
                    plt.xlabel(feature)
                    plt.ylabel('Frequency')

            else:
                st.write("### Categorical Variable Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Bar Chart",
                    "Pie Chart"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Bar Chart":
                    data[feature].value_counts().plot(kind='bar')
                    plt.title(f'Bar Chart of {feature}')
                    plt.xlabel(feature)
                    plt.ylabel('Count')

                elif plot_type == "Pie Chart":
                    data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%')
                    plt.title(f'Pie Chart of {feature}')

            st.pyplot(plt)

        # Plot Two Variables
        elif analysis_option == "Plot Two Variables":
            st.subheader("Plot Two Variables")
            st.write("Select two variables to visualize their relationship.")
            x_axis = st.selectbox("Select X variable:", data.columns)
            y_axis = st.selectbox("Select Y variable:", data.columns, index=1)

            if data[x_axis].dtype in [np.number, 'float64', 'int64'] and data[y_axis].dtype in [np.number, 'float64', 'int64']:
                st.write("### Numerical vs Numerical Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Scatter Plot",
                    "Box Plot",
                    "Line Graph",
                    "Grouped Bar Chart",
                    "Bubble Chart",
                    "Violin Chart"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Scatter Plot":
                    sns.scatterplot(data=data, x=x_axis, y=y_axis)
                    plt.title(f'Scatter Plot of {y_axis} vs {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)

                elif plot_type == "Box Plot":
                    sns.boxplot(data=data, x=x_axis, y=y_axis)
                    plt.title(f'Box Plot of {y_axis} by {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)

                elif plot_type == "Line Graph":
                    plt.plot(data[x_axis], data[y_axis])
                    plt.title(f'Line Graph of {y_axis} vs {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)

                elif plot_type == "Grouped Bar Chart":
                    data.groupby(x_axis)[y_axis].mean().plot(kind='bar')
                    plt.title(f'Grouped Bar Chart of {y_axis} by {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(f'Mean {y_axis}')

                elif plot_type == "Bubble Chart":
                    plt.scatter(data[x_axis], data[y_axis], s=data[y_axis]*10, alpha=0.5)
                    plt.title(f'Bubble Chart of {y_axis} vs {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)

                elif plot_type == "Violin Chart":
                    sns.violinplot(x=x_axis, y=y_axis, data=data)
                    plt.title(f'Violin Chart of {y_axis} by {x_axis}')

                st.pyplot(plt)

            elif data[x_axis].dtype in [np.number, 'float64', 'int64'] and data[y_axis].dtype in ['object']:
                st.write("### Numerical vs Categorical Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Box Plot",
                    "Violin Plot"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Box Plot":
                    sns.boxplot(data=data, x=x_axis, y=y_axis)
                    plt.title(f'Box Plot of {y_axis} by {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)

                elif plot_type == "Violin Plot":
                    sns.violinplot(x=x_axis, y=y_axis, data=data)
                    plt.title(f'Violin Plot of {y_axis} by {x_axis}')

                st.pyplot(plt)

            elif data[x_axis].dtype in ['object'] and data[y_axis].dtype in ['object']:
                st.write("### Categorical vs Categorical Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Grouped Bar Chart"
                ])

                plt.figure(figsize=(10, 6))
                data.groupby([x_axis, y_axis]).size().unstack().plot(kind='bar', stacked=True)
                plt.title(f'Grouped Bar Chart of {y_axis} by {x_axis}')
                plt.xlabel(x_axis)
                plt.ylabel('Count')

                st.pyplot(plt)

        # Plot Three Variables
        elif analysis_option == "Plot Three Variables":
            st.subheader("Plot Three Variables")
            st.write("Select three variables to visualize. You can select numerical or categorical variables.")

            num_or_cat = st.selectbox("Choose variable type:", ["Numerical", "Categorical"])
            if num_or_cat == "Numerical":
                numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
                selected_vars = st.multiselect("Select three numerical variables:", numerical_features, max_selections=3)

                if len(selected_vars) == 3:
                    plot_type = st.selectbox("Select plot type:", [
                        "3D Scatter Plot",
                        "Parallel Coordinates Plot"
                    ])

                    if plot_type == "3D Scatter Plot":
                        fig = plt.figure(figsize=(10, 6))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(data[selected_vars[0]], data[selected_vars[1]], data[selected_vars[2]])
                        ax.set_xlabel(selected_vars[0])
                        ax.set_ylabel(selected_vars[1])
                        ax.set_zlabel(selected_vars[2])
                        plt.title('3D Scatter Plot of Selected Numerical Variables')

                    elif plot_type == "Parallel Coordinates Plot":
                        plt.figure(figsize=(10, 6))
                        parallel_coordinates(data[selected_vars], class_column=selected_vars[0])
                        plt.title('Parallel Coordinates Plot')

                    st.pyplot(plt)

            elif num_or_cat == "Categorical":
                categorical_features = data.select_dtypes(include=['object']).columns.tolist()
                selected_vars = st.multiselect("Select three categorical variables:", categorical_features, max_selections=3)

                if len(selected_vars) == 3:
                    st.write("### Categorical vs Categorical Options")
                    plot_type = st.selectbox("Select plot type:", [
                        "Grouped Bar Chart"
                    ])

                    plt.figure(figsize=(10, 6))
                    data.groupby(selected_vars).size().unstack().plot(kind='bar', stacked=True)
                    plt.title('Grouped Bar Chart of Selected Categorical Variables')
                    plt.xlabel('Categories')
                    plt.ylabel('Count')

                    st.pyplot(plt)

        # Plot More Than Three Variables
        elif analysis_option == "Plot More Than Three Variables":
            st.subheader("Plot More Than Three Variables")
            st.write("Select features to visualize. You must select at least one numerical and one categorical variable.")
            feature_columns = st.multiselect("Select features to plot:", data.columns)

            if len(feature_columns) > 1:
                # Separate numerical and categorical features
                numerical_features = [col for col in feature_columns if data[col].dtype in [np.number, 'float64', 'int64']]
                categorical_features = [col for col in feature_columns if data[col].dtype in ['object']]

                if numerical_features and categorical_features:
                    plot_type = st.selectbox("Select plot type:", [
                        "Parallel Coordinates Plot",
                        "Scatterplot Matrix",
                        "Andrews Curves"
                    ])

                    if plot_type == "Parallel Coordinates Plot":
                        plt.figure(figsize=(10, 6))
                        parallel_coordinates(data[feature_columns], class_column=categorical_features[0])
                        plt.title('Parallel Coordinates Plot')
                        plt.xlabel('Features')
                        plt.ylabel('Values')
                        st.pyplot(plt)

                    elif plot_type == "Scatterplot Matrix":
                        plt.figure(figsize=(10, 10))
                        sns.pairplot(data[feature_columns])
                        plt.title('Scatterplot Matrix')
                        st.pyplot(plt)

                    elif plot_type == "Andrews Curves":
                        plt.figure(figsize=(10, 6))
                        andrews_curves(data[feature_columns], class_column=categorical_features[0])
                        plt.title('Andrews Curves Plot')
                        st.pyplot(plt)

                else:
                    st.error("You must select at least one numerical and one categorical variable.")

        # AI Analysis
        elif analysis_option == "AI Analysis":
            st.subheader("AI Analysis")
            st.write("Select a feature for AI analysis to receive statistical insights.")
            feature = st.selectbox("Select feature for AI analysis:", data.columns)
            analysis_description = generate_analysis(feature, data)
            st.write(analysis_description)

# About Us section
elif choice == "About Us":
    st.subheader("About Us")
    st.write("""
        This EDA tool helps you visualize and analyze your data easily.
        It provides various statistical and graphical analyses to enhance your understanding of the data.
    """)

# Contact Us section
elif choice == "Contact Us":
    st.subheader("Contact Us")
    st.write("For inquiries, please email us at contact@example.com.")





