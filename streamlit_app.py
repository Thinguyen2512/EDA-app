import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates
from statsmodels.graphics.mosaicplot import mosaic

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
st.title("Exploratory Data Analysis (EDA) Tool")

# Navigation menu
menu = ["About Us", "Upload Your Data", "Contact Us"]
choice = st.sidebar.selectbox("Select feature", menu)

# About Us section
if choice == "About Us":
    st.subheader("About Us")
    st.write("""
        This EDA tool helps you visualize and analyze your data easily.
        It provides various statistical and graphical analyses to enhance your understanding of the data.
    """)

# Upload Data section
elif choice == "Upload Your Data":
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
            st.write("Select a variable to visualize. Choose a numerical or categorical variable.")
            feature = st.selectbox("Select variable to plot:", data.columns)

            if data[feature].dtype in [np.number, 'float64', 'int64']:
                st.write("### Numerical Variable Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Histogram",
                    "Box Plot"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Histogram":
                    sns.histplot(data[feature], kde=True)
                    plt.title(f'Histogram of {feature}')
                    plt.xlabel(feature)
                    plt.ylabel('Frequency')

                elif plot_type == "Box Plot":
                    sns.boxplot(y=data[feature])
                    plt.title(f'Boxplot of {feature}')

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

            # Grouping options for two variables
            if data[x_axis].dtype in [np.number, 'float64', 'int64'] and data[y_axis].dtype in [np.number, 'float64', 'int64']:
                st.write("### Two Numerical Variables Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Scatter Plot",
                    "Line Graph",
                    "Area Chart"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Scatter Plot":
                    sns.scatterplot(data=data, x=x_axis, y=y_axis)
                    plt.title(f'Scatter Plot of {y_axis} vs {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)

                elif plot_type == "Line Graph":
                    plt.plot(data[x_axis], data[y_axis])
                    plt.title(f'Line Graph of {y_axis} vs {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)

                elif plot_type == "Area Chart":
                    data[[x_axis, y_axis]].plot(kind='area')
                    plt.title(f'Area Chart of {y_axis} vs {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                st.pyplot(plt)
                
            elif data[x_axis].dtype in [np.number, 'float64', 'int64'] and data[y_axis].dtype in ['object']:
                st.write("### One Numerical and One Categorical Variable Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Bar Chart"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Bar Chart":
                    data.groupby(x_axis)[y_axis].value_counts().unstack().plot(kind='bar')
                    plt.title(f'Grouped Bar Chart of {y_axis} by {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel('Count')

                st.pyplot(plt)

            elif data[x_axis].dtype in ['object'] and data[y_axis].dtype in ['object']:
                st.write("### Two Categorical Variables Options")
                plot_type = st.selectbox("Select plot type:", [
                    "Grouped Bar Chart",
                    "Mosaic Plot"
                ])

                plt.figure(figsize=(10, 6))

                if plot_type == "Grouped Bar Chart":
                    data.groupby([x_axis, y_axis]).size().unstack().plot(kind='bar', stacked=True)
                    plt.title(f'Grouped Bar Chart of {y_axis} by {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel('Count')

                elif plot_type == "Mosaic Plot":
                    mosaic(data, [x_axis, y_axis])
                    plt.title(f'Mosaic Plot of {x_axis} and {y_axis}')
                st.pyplot(plt)

        # Plot Three Variables
        elif analysis_option == "Plot Three Variables":
            st.subheader("Plot Three Variables")
            st.write("Select three variables to visualize. Choose among the following options:")

            plot_type = st.selectbox("Select Analysis Type for 3 Variables:", [
                "3 Numerical Variables",
                "2 Categorical Variables and 1 Numerical Variable",
                "2 Numerical Variables and 1 Categorical Variable"
            ])

            if plot_type == "3 Numerical Variables":
                numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

                selected_vars = st.multiselect("Select three numerical variables:", numerical_features, max_selections=3)

                if len(selected_vars) == 3:
                    st.write("### Plot for 3 Numerical Variables")
                    plot_choice = st.selectbox("Select plot type:", [
                        "3D Scatter Plot",
                        "Contour Plot",
                        "Bubble Chart"
                    ])

                    if plot_choice == "3D Scatter Plot":
                        fig = plt.figure(figsize=(10, 6))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(data[selected_vars[0]], data[selected_vars[1]], data[selected_vars[2]])
                        ax.set_xlabel(selected_vars[0])
                        ax.set_ylabel(selected_vars[1])
                        ax.set_zlabel(selected_vars[2])
                        plt.title('3D Scatter Plot of Selected Numerical Variables')
                        st.pyplot(plt)

                   elif plot_choice == "Contour Plot":
                        fig = plt.figure(figsize=(10, 6))
                        ax = fig.add_subplot(111)
    
                        # Chọn 3 biến từ dữ liệu
                        x_var, y_var, z_var = selected_vars
    
                        # Chuyển dữ liệu thành dạng phù hợp với meshgrid
                        # Đảm bảo các giá trị x, y, z đều được sắp xếp
                        x = data[x_var].dropna().values
                        y = data[y_var].dropna().values
                        z = data[z_var].dropna().values

                        # Tạo lưới (grid) cho biểu đồ contour
                        X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))

                        # Dự đoán các giá trị Z cho lưới X, Y (sử dụng interpolation)
                        from scipy.interpolate import griddata
                        Z = griddata((x, y), z, (X, Y), method='linear')

                        # Vẽ contour plot
                        contour = ax.contour(X, Y, Z, levels=10, cmap='coolwarm')
                        plt.title(f'Contour Plot of {z_var} vs {x_var} and {y_var}')
                        plt.xlabel(x_var)
                        plt.ylabel(y_var)

                        # Thêm colorbar
                        plt.colorbar(contour, ax=ax)

                        # Hiển thị biểu đồ
                        st.pyplot(fig)

                        
                    elif plot_choice == "Bubble Chart":
                        x, y, z = selected_vars
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Bubble size: Use the z variable for bubble sizes
                        ax.scatter(data[x], data[y], s=data[z] * 10, alpha=0.5, c='blue', edgecolors="w", linewidth=0.5)
                        ax.set_xlabel(x)
                        ax.set_ylabel(y)
                        plt.title(f'Bubble Chart of {y} vs {x} with Bubble Size based on {z}')
                        st.pyplot(plt)

            elif plot_type == "2 Categorical Variables and 1 Numerical Variable":
                categorical_features = data.select_dtypes(include=['object']).columns.tolist()
                numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

                selected_vars = st.multiselect("Select two categorical variables and one numerical variable:", 
                                              categorical_features + numerical_features, max_selections=3)

                if len(selected_vars) == 3:
                    st.write("### Plot for 2 Categorical Variables and 1 Numerical Variable")
                    plot_choice = st.selectbox("Select plot type:", [
                        "Grid Plot (Heatmap)"
                    ])

                    if plot_choice == "Grid Plot (Heatmap)":
                        x, y, z = selected_vars
                        # Create grid data using pivot_table
                        grid_data = data.pivot_table(values=z, index=x, columns=y, aggfunc='sum')
                        
                        # Grid plot using heatmap
                        plt.figure(figsize=(10, 6))
                        sns.heatmap(grid_data, annot=True, cmap="coolwarm", fmt=".1f")
                        plt.title(f'Grid Plot (Heatmap) of {z} by {x} and {y}')
                        plt.xlabel(y)
                        plt.ylabel(x)
                        st.pyplot(plt)

            elif plot_type == "2 Numerical Variables and 1 Categorical Variable":
                numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_features = data.select_dtypes(include=['object']).columns.tolist()

                selected_vars = st.multiselect("Select two numerical variables and one categorical variable:", 
                                              numerical_features + categorical_features, max_selections=3)

                if len(selected_vars) == 3:
                    st.write("### Plot for 2 Numerical Variables and 1 Categorical Variable")
                    plot_choice = st.selectbox("Select plot type:", [
                        "Area Chart",
                        "Stacked Bar Chart",
                        "Stacked Column Chart"
                    ])

                    if plot_choice == "Area Chart":
                        x, y, z = selected_vars
                        # Grouping by categorical variable and summing numerical ones
                        area_data = data.groupby([z, x])[y].sum().unstack()
                        area_data.plot(kind='area', figsize=(10, 6), stacked=True)
                        plt.title(f'Area Chart of {y} vs {x} grouped by {z}')
                        plt.xlabel(x)
                        plt.ylabel(y)
                        st.pyplot(plt)

                    elif plot_choice == "Stacked Bar Chart":
                        x, y, z = selected_vars
                        # Stacked Bar Chart
                        stacked_data = data.groupby([z, x])[y].sum().unstack()
                        stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6))
                        plt.title(f'Stacked Bar Chart of {y} vs {x} grouped by {z}')
                        plt.xlabel(x)
                        plt.ylabel(y)
                        st.pyplot(plt)

                    elif plot_choice == "Stacked Column Chart":
                        x, y, z = selected_vars
                        # Stacked Column Chart
                        stacked_data = data.groupby([z, x])[y].sum().unstack()
                        stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6), orientation='vertical')
                        plt.title(f'Stacked Column Chart of {y} vs {x} grouped by {z}')
                        plt.xlabel(x)
                        plt.ylabel(y)
                        st.pyplot(plt)

        # AI Analysis Placeholder (Optional)
        elif analysis_option == "AI Analysis":
            st.subheader("AI-based Analysis Placeholder")
            st.write("AI analysis options will be available soon.")

# Contact Us section
elif choice == "Contact Us":
    st.subheader("Contact Us")
    st.write("For inquiries, please email us at contact@example.com.")















