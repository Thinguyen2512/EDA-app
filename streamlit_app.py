import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates, andrews_curves
from scipy.cluster import hierarchy
import plotly.express as px

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
            feature = st.selectbox("Select feature to plot:", data.columns)
            plot_type = st.selectbox("Select plot type:", [
                "Line Chart",
                "Bar Chart",
                "Pie Chart",
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

            elif plot_type == "Bar Chart":
                data[feature].value_counts().plot(kind='bar')
                plt.title(f'Bar Chart of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Count')

            elif plot_type == "Pie Chart":
                data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {feature}')

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

            st.pyplot(plt)

        # Plot Two Variables
        elif analysis_option == "Plot Two Variables":
            st.subheader("Plot Two Variables")
            x_axis = st.selectbox('Select X variable:', data.columns)
            y_axis = st.selectbox('Select Y variable:', data.columns, index=1)
            plot_type = st.selectbox("Select plot type:", [
                "Scatter Plot",
                "Box Plot",
                "Line Graph",
                "Grouped Bar Chart",
                "Heat Map",
                "Bubble Chart",
                "Stacked Bar Chart",
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

            elif plot_type == "Heat Map":
                numeric_data = data[[x_axis, y_axis]].select_dtypes(include=[np.number])

                if numeric_data.shape[0] == 0:
                    st.error("Selected variables do not contain numeric data.")
                else:
                    correlation_matrix = numeric_data.corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
                    plt.title('Heat Map of Correlation between Selected Variables')
                    st.pyplot(plt)

            elif plot_type == "Bubble Chart":
                plt.scatter(data[x_axis], data[y_axis], s=data[y_axis]*10, alpha=0.5)
                plt.title(f'Bubble Chart of {y_axis} vs {x_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Stacked Bar Chart":
                data.groupby([x_axis, y_axis]).size().unstack().plot(kind='bar', stacked=True)
                plt.title(f'Stacked Bar Chart of {y_axis} by {x_axis}')
                plt.xlabel(x_axis)
                plt.ylabel('Count')

            elif plot_type == "Violin Chart":
                sns.violinplot(x=x_axis, y=y_axis, data=data)
                plt.title(f'Violin Chart of {y_axis} by {x_axis}')

            st.pyplot(plt)

        # Plot Three Variables
        elif analysis_option == "Plot Three Variables":
            st.subheader("Plot Three Variables")
            x_axis = st.selectbox('Select X variable:', data.columns)
            y_axis = st.selectbox('Select Y variable:', data.columns, index=1)
            z_axis = st.selectbox('Select Z variable (size or color):', data.columns)

            plot_type = st.selectbox("Select plot type:", [
                "3D Scatter Plot",
                "Surface Plot",
                "Bubble Chart",
                "Ternary Plot",
                "Contour Plot",
                "Raster Plot",
                "Tile Plot"
            ])

            plt.figure(figsize=(10, 6))
            
            if plot_type == "3D Scatter Plot":
                ax = plt.axes(projection='3d')
                ax.scatter(data[x_axis], data[y_axis], data[z_axis])
                ax.set_title(f'3D Scatter Plot of {y_axis}, {x_axis}, {z_axis}')
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_zlabel(z_axis)

            elif plot_type == "Surface Plot":
                # Ensure you have enough points and valid data
                x_unique = np.unique(data[x_axis])
                y_unique = np.unique(data[y_axis])
                X, Y = np.meshgrid(x_unique, y_unique)

                # Use griddata to interpolate
                Z = plt.tricontourf(data[x_axis], data[y_axis], data[z_axis], levels=15)
                plt.colorbar()
                plt.title(f'Surface Plot of {y_axis} over {x_axis} and {y_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Bubble Chart":
                plt.scatter(data[x_axis], data[y_axis], s=data[z_axis]*10, alpha=0.5)
                plt.title(f'Bubble Chart of {y_axis} vs {x_axis} with Size based on {z_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Ternary Plot":
                # Ternary plot is usually for 3 components that sum to 1
                from matplotlib import cm
                from matplotlib.ticker import MaxNLocator
                from math import sqrt
                
                # Normalize the data to sum to 1
                ternary_data = data[[x_axis, y_axis, z_axis]].copy()
                ternary_data = ternary_data.div(ternary_data.sum(axis=1), axis=0)

                # Create the ternary plot
                fig, tax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='ternary'))
                tax.scatter(ternary_data[x_axis], ternary_data[y_axis], marker='o', alpha=0.5)
                tax.set_title('Ternary Plot')
                tax.left_axis_label(y_axis)
                tax.right_axis_label(z_axis)
                tax.bottom_axis_label(x_axis)

            elif plot_type == "Contour Plot":
                sns.kdeplot(x=data[x_axis], y=data[y_axis], fill=True)
                plt.title('Contour Plot')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Raster Plot":
                plt.imshow(data[[x_axis, y_axis]].values, aspect='auto')
                plt.title('Raster Plot')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Tile Plot":
                sns.heatmap(data[[x_axis, y_axis]].corr(), annot=True)
                plt.title('Tile Plot of Correlation')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            st.pyplot(plt)

        # Plot More Than Three Variables
        elif analysis_option == "Plot More Than Three Variables":
            st.subheader("Plot More Than Three Variables")
            feature_columns = st.multiselect("Select features to plot:", data.columns)

            if len(feature_columns) > 1:
                plot_type = st.selectbox("Select plot type:", [
                    "Parallel Coordinates Plot",
                    "Heatmap with Dendrograms",
                    "Scatterplot Matrix",
                    "Andrews Curves",
                    "Glyph Plots",
                    "Interactive Plot"
                ])

                if plot_type == "Parallel Coordinates Plot":
                    plt.figure(figsize=(10, 6))
                    parallel_coordinates(data[feature_columns], class_column=feature_columns[0])
                    plt.title('Parallel Coordinates Plot')
                    plt.xlabel('Features')
                    plt.ylabel('Values')
                    st.pyplot(plt)

                elif plot_type == "Heatmap with Dendrograms":
                    plt.figure(figsize=(10, 8))
                    sns.clustermap(data[feature_columns], method='ward', cmap='viridis', figsize=(10, 10))
                    plt.title('Heatmap with Dendrograms')
                    st.pyplot(plt)

                elif plot_type == "Scatterplot Matrix":
                    plt.figure(figsize=(10, 10))
                    sns.pairplot(data[feature_columns])
                    plt.title('Scatterplot Matrix')
                    st.pyplot(plt)

                elif plot_type == "Andrews Curves":
                    plt.figure(figsize=(10, 6))
                    andrews_curves(data[feature_columns], class_column=feature_columns[0])
                    plt.title('Andrews Curves')
                    st.pyplot(plt)

                elif plot_type == "Glyph Plots":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for index, row in data.iterrows():
                        ax.scatter(row[feature_columns[0]], row[feature_columns[1]], marker='o', alpha=0.5)
                        ax.text(row[feature_columns[0]], row[feature_columns[1]], str(index), fontsize=8)
                    plt.title('Glyph Plot')
                    plt.xlabel(feature_columns[0])
                    plt.ylabel(feature_columns[1])
                    st.pyplot(plt)

                elif plot_type == "Interactive Plot":
                    st.subheader("Interactive Scatter Plot")
                    fig = px.scatter(data, x=feature_columns[0], y=feature_columns[1], color=feature_columns[2] if len(feature_columns) > 2 else None)
                    st.plotly_chart(fig)

        # AI Analysis
        elif analysis_option == "AI Analysis":
            st.subheader("AI Analysis")
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




