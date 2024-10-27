import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from pandas.plotting import parallel_coordinates

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

# Function to generate data analysis
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
st.title("EDA Tool")

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
                "Line Chart", "Bar Chart", "Pie Chart", "Histogram",
                "Box Plot", "Density Plot", "Area Chart", "Dot Plot", "Frequency Polygon"
            ])

            plt.figure(figsize=(10, 6))

            plot_funcs = {
                "Line Chart": lambda: plt.plot(data[feature]),
                "Bar Chart": lambda: data[feature].value_counts().plot(kind='bar'),
                "Pie Chart": lambda: data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%'),
                "Histogram": lambda: sns.histplot(data[feature], kde=True),
                "Box Plot": lambda: sns.boxplot(y=data[feature]),
                "Density Plot": lambda: sns.kdeplot(data[feature]),
                "Area Chart": lambda: data[feature].plot(kind='area'),
                "Dot Plot": lambda: plt.plot(data.index, data[feature], 'o'),
                "Frequency Polygon": lambda: sns.histplot(data[feature], kde=False, bins=30)
            }

            plot_funcs[plot_type]()
            plt.title(f'{plot_type} of {feature}')
            plt.xlabel('Index' if plot_type != "Pie Chart" else '')
            plt.ylabel(feature if plot_type != "Pie Chart" else '')

            st.pyplot(plt)

        # Plot Two Variables
        elif analysis_option == "Plot Two Variables":
            st.subheader("Plot Two Variables")
            x_axis = st.selectbox('Select X variable:', data.columns)
            y_axis = st.selectbox('Select Y variable:', data.columns, index=1)
            plot_type = st.selectbox("Select plot type:", [
                "Scatter Plot", "Box Plot", "Line Graph", "Grouped Bar Chart",
                "Heat Map", "Bubble Chart", "Stacked Bar Chart", "Violin Chart"
            ])

            plt.figure(figsize=(10, 6))

            if plot_type == "Scatter Plot":
                sns.scatterplot(data=data, x=x_axis, y=y_axis)
            elif plot_type == "Box Plot":
                sns.boxplot(data=data, x=x_axis, y=y_axis)
            elif plot_type == "Line Graph":
                plt.plot(data[x_axis], data[y_axis])
            elif plot_type == "Grouped Bar Chart":
                data.groupby(x_axis)[y_axis].mean().plot(kind='bar')
            elif plot_type == "Heat Map":
                numeric_data = data[[x_axis, y_axis]].select_dtypes(include=[np.number])
                if numeric_data.shape[0] == 0:
                    st.error("Selected variables do not contain numeric data.")
                else:
                    correlation_matrix = numeric_data.corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
                    st.pyplot(plt)
            elif plot_type == "Bubble Chart":
                plt.scatter(data[x_axis], data[y_axis], s=data[y_axis]*10, alpha=0.5)
            elif plot_type == "Stacked Bar Chart":
                data.groupby([x_axis, y_axis]).size().unstack().plot(kind='bar', stacked=True)
            elif plot_type == "Violin Chart":
                sns.violinplot(x=x_axis, y=y_axis, data=data)

            plt.title(f'{plot_type} of {y_axis} vs {x_axis}')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

            st.pyplot(plt)

        # Plot Three Variables
        elif analysis_option == "Plot Three Variables":
            st.subheader("Plot Three Variables")
            x_axis = st.selectbox('Select X variable:', data.columns)
            y_axis = st.selectbox('Select Y variable:', data.columns, index=1)
            z_axis = st.selectbox('Select Z variable (size or color):', data.columns)

            plot_type = st.selectbox("Select plot type:", [
                "3D Scatter Plot", "Surface Plot", "Bubble Chart", "Ternary Plot",
                "Contour Plot", "Raster Plot", "Tile Plot"
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
                x_unique = np.unique(data[x_axis])
                y_unique = np.unique(data[y_axis])
                X, Y = np.meshgrid(x_unique, y_unique)
                Z = griddata((data[x_axis], data[y_axis]), data[z_axis], (X, Y), method='linear')
                plt.contourf(X, Y, Z, levels=14, cmap='viridis')
                plt.colorbar(label=z_axis)
                plt.title(f'Surface Plot of {z_axis} by {x_axis} and {y_axis}')
            elif plot_type == "Bubble Chart":
                plt.scatter(data[x_axis], data[y_axis], s=data[z_axis]*10, alpha=0.5)
            elif plot_type == "Ternary Plot":
                if data[[x_axis, y_axis, z_axis]].isnull().values.any():
                    st.error("Data contains null values. Please clean your data before plotting.")
                else:
                    import ternary
                    fig, tax = ternary.figure()
                    tax.scatter(data[[x_axis, y_axis, z_axis]].values, marker='o')
                    tax.set_title('Ternary Plot')
                    tax.right_axis_label(z_axis)
                    tax.left_axis_label(y_axis)
                    tax.bottom_axis_label(x_axis)
                    tax.show()
            elif plot_type == "Contour Plot":
                plt.tricontour(data[x_axis], data[y_axis], data[z_axis])
                plt.title(f'Contour Plot of {z_axis} by {x_axis} and {y_axis}')
            elif plot_type == "Raster Plot":
                plt.imshow(data[[x_axis, y_axis, z_axis]], aspect='auto', cmap='viridis')
                plt.title(f'Raster Plot of {z_axis} by {x_axis} and {y_axis}')
            elif plot_type == "Tile Plot":
                sns.histplot(data[[x_axis, y_axis]], bins=30, pmax=0.8, cmap='coolwarm', cbar=True)
                plt.title(f'Tile Plot of {y_axis} by {x_axis}')

            st.pyplot(plt)

        # Plot More Than Three Variables
        elif analysis_option == "Plot More Than Three Variables":
            st.subheader("Plot More Than Three Variables")
            feature_columns = st.multiselect("Select features to plot:", data.columns)
            if len(feature_columns) > 1:
                plt.figure(figsize=(10, 6))
                parallel_coordinates(data[feature_columns], class_column=feature_columns[0])
                plt.title('Parallel Coordinates Plot')
                plt.xlabel('Features')
                plt.ylabel('Values')
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



