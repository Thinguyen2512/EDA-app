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
                    st.write("### Options for 3 Numerical Variables")
                    plot_type = st.selectbox("Select plot type:", [
                        "3D Scatter Plot",
                        "Contour Plot",
                        "Bubble Chart"
                    ])

                    if plot_type == "3D Scatter Plot":
                        fig = plt.figure(figsize=(10, 6))
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(data[selected_vars[0]], data[selected_vars[1]], data[selected_vars[2]])
                        ax.set_xlabel(selected_vars[0])
                        ax.set_ylabel(selected_vars[1])
                        ax.set_zlabel(selected_vars[2])
                        plt.title('3D Scatter Plot of Selected Numerical Variables')

                    elif plot_type == "Contour Plot":
                        x = np.linspace(data[selected_vars[0]].min(), data[selected_vars[0]].max(), 100)
                        y = np.linspace(data[selected_vars[1]].min(), data[selected_vars[1]].max(), 100)
                        X, Y = np.meshgrid(x, y)

                        # We need to create Z by using the third variable, interpolating over X and Y.
                        Z = np.random.random(X.shape)  # Replace this with actual Z computation logic

                        plt.contour(X, Y, Z)
                        plt.title(f'Contour Plot of {selected_vars[0]} and {selected_vars[1]} vs {selected_vars[2]}')

                    elif plot_type == "Bubble Chart":
                        plt.figure(figsize=(10, 6))
                        plt.scatter(data[selected_vars[0]], data[selected_vars[1]], 
                                    s=data[selected_vars[2]] * 10, alpha=0.5)  # Size of bubble proportional to third variable
                        plt.title(f'Bubble Chart of {selected_vars[0]} vs {selected_vars[1]} by {selected_vars[2]}')
                        plt.xlabel(selected_vars[0])
                        plt.ylabel(selected_vars[1])

                    st.pyplot(plt)

            elif plot_type == "2 Categorical Variables and 1 Numerical Variable":
                categorical_features = data.select_dtypes(include=['object']).columns.tolist()
                selected_vars = st.multiselect("Select two categorical and one numerical variable:", 
                                              categorical_features + numerical_features, max_selections=3)

                if len(selected_vars) == 3:
                    st.write("### Plot for 2 Categorical Variables and 1 Numerical Variable")
                    plot_type = st.selectbox("Select plot type:", [
                        "Bar Chart",
                        "Stacked Bar Chart"
                    ])

                    if plot_type == "Bar Chart":
                        sns.barplot(data=data, x=selected_vars[0], y=selected_vars[2], hue=selected_vars[1])
                        plt.title(f'Bar Chart of {selected_vars[2]} by {selected_vars[0]} and {selected_vars[1]}')
                        plt.xlabel(selected_vars[0])
                        plt.ylabel(selected_vars[2])

                    elif plot_type == "Stacked Bar Chart":
                        data.groupby([selected_vars[0], selected_vars[1]])[selected_vars[2]].sum().unstack().plot(kind='bar', stacked=True)
                        plt.title(f'Stacked Bar Chart of {selected_vars[2]} by {selected_vars[0]} and {selected_vars[1]}')
                        plt.xlabel(selected_vars[0])
                        plt.ylabel(selected_vars[2])

                st.pyplot(plt)

        # AI Analysis Placeholder (Optional)
        elif analysis_option == "AI Analysis":
            st.subheader("AI-based Analysis Placeholder")
            st.write("AI analysis options will be available soon.")

















