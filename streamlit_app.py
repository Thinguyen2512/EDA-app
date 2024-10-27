import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
            "Plot One Variable",
            "Plot Two Variables",
            "Plot Three Variables",
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
            y_axis = st.selectbox('Select Y variable:', data.columns)
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
                # Select only numeric columns for correlation
                numeric_data = data[[x_axis, y_axis]].select_dtypes(include=[np.number])

                # Check if there is enough numeric data
                if numeric_data.shape[0] == 0:
                    st.error("Selected variables do not contain numeric data.")
                else:
                    # Compute correlation matrix
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
                "3D Heatmap",
                "Grid Plot",
                "Contour Plot",
                "Raster Plot",
                "Tile Plot"
            ])

            fig = plt.figure(figsize=(10, 6))
            
            if plot_type == "3D Scatter Plot":
                ax = fig.add_subplot(111, projection='3d')
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

                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='viridis')
                ax.set_title(f'Surface Plot of {z_axis} vs {x_axis} and {y_axis}')
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_zlabel(z_axis)

            elif plot_type == "Bubble Chart":
                plt.scatter(data[x_axis], data[y_axis], s=data[z_axis]*10, alpha=0.5)
                plt.title(f'Bubble Chart of {y_axis} vs {x_axis} with size {z_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "3D Heatmap":
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data[x_axis], data[y_axis], data[z_axis])
                ax.set_title('3D Heatmap (scatter representation)')
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_zlabel(z_axis)

            elif plot_type == "Grid Plot":
                numeric_data = data.select_dtypes(include=[np.number])

                if numeric_data.shape[1] < 2:
                    st.error("Not enough numeric data to compute the correlation matrix.")
                else:
                    correlation_matrix = numeric_data.corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
                    plt.title('Grid Plot of Correlations')
                    st.pyplot(plt)

            elif plot_type == "Contour Plot":
                plt.tricontourf(data[x_axis], data[y_axis], data[z_axis], cmap='viridis')
                plt.title(f'Contour Plot of {z_axis} vs {x_axis} and {y_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)

            elif plot_type == "Raster Plot":
                plt.imshow(data.pivot_table(values=z_axis, index=y_axis, columns=x_axis), cmap='viridis', aspect='auto')
                plt.title(f'Raster Plot of {z_axis} vs {x_axis} and {y_axis}')
                plt.colorbar()

            elif plot_type == "Tile Plot":
                plt.imshow(data.pivot_table(values=z_axis, index=y_axis, columns=x_axis), cmap='viridis', aspect='equal')
                plt.title(f'Tile Plot of {z_axis} vs {x_axis} and {y_axis}')
                plt.colorbar()

            st.pyplot(fig)

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

