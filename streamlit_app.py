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
            st.write("Select three variables to visualize. You can select continuous or categorical variables.")
            
            variable_type = st.selectbox("Choose variable type:", [
                "Three Numerical Variables", 
                "Two Categorical and One Numerical Variable", 
                "Two Numerical and One Categorical Variable"
            ])
            
            if variable_type == "Three Numerical Variables":
                numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
                selected_vars = st.multiselect("Select three numerical variables:", numerical_features, max_selections=3)

                if len(selected_vars) == 3:
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
                        X, Y = np.meshgrid(data[selected_vars[0]], data[selected_vars[1]])
                        Z = data[selected_vars[2]]
                        plt.contour(X, Y, Z)
                        plt.title('Contour Plot of Selected Numerical Variables')

                    elif plot_type == "Bubble Chart":
                        plt.scatter(data[selected_vars[0]], data[selected_vars[1]], s=data[selected_vars[2]]*10, alpha=0.5)
                        plt.title('Bubble Chart of Selected Numerical Variables')

                    st.pyplot(plt)

            elif variable_type == "Two Categorical and One Numerical Variable":
                categorical_features = data.select_dtypes(include=['object']).columns.tolist()
                numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
                selected_vars = st.multiselect("Select two categorical variables and one numerical variable:", 
                                              categorical_features + numerical_features, max_selections=3)

                if len(selected_vars) == 3:
                    plot_type = st.selectbox("Select plot type:", [
                        "Box Plot",
                        "Violin Plot",
                        "Stacked Bar Chart"
                    ])

                    if plot_type == "Box Plot":
                        sns.boxplot(x=selected_vars[0], y=selected_vars[2], data=data, hue=selected_vars[1])
                        plt.title(f"Box Plot of {selected_vars[2]} by {selected_vars[0]} and {selected_vars[1]}")

                    elif plot_type == "Violin Plot":
                        sns.violinplot(x=selected_vars[0], y=selected_vars[2], data=data, hue=selected_vars[1])
                        plt.title(f"Violin Plot of {selected_vars[2]} by {selected_vars[0]} and {selected_vars[1]}")

                    elif plot_type == "Stacked Bar Chart":
                        data.groupby([selected_vars[0], selected_vars[1]]).size().unstack().plot(kind='bar', stacked=True)
                        plt.title(f"Stacked Bar Chart of {selected_vars[0]} and {selected_vars[1]} by {selected_vars[2]}")

                    st.pyplot(plt)

            elif variable_type == "Two Numerical and One Categorical Variable":
                numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_features = data.select_dtypes(include=['object']).columns.tolist()
                selected_vars = st.multiselect("Select two numerical variables and one categorical variable:", 
                                              numerical_features + categorical_features, max_selections=3)

                if len(selected_vars) == 3:
                    plot_type = st.selectbox("Select plot type:", [
                        "Area Chart",
                        "Stacked Bar Chart",
                        "Stacked Column Chart"
                    ])

                    if plot_type == "Area Chart":
                        data[selected_vars].plot(kind='area')
                        plt.title(f"Area Chart of {', '.join(selected_vars)}")

                    elif plot_type == "Stacked Bar Chart":
                        data.groupby([selected_vars[0], selected_vars[1]]).size().unstack().plot(kind='bar', stacked=True)
                        plt.title(f"Stacked Bar Chart of {selected_vars[0]} and {selected_vars[1]} by {selected_vars[2]}")

                    elif plot_type == "Stacked Column Chart":
                        data.groupby([selected_vars[0], selected_vars[1]]).size().unstack().plot(kind='bar', stacked=True, orientation='horizontal')
                        plt.title(f"Stacked Column Chart of {selected_vars[0]} and {selected_vars[1]} by {selected_vars[2]}")

                    st.pyplot(plt)

        # AI Analysis (This section is just a placeholder)
        elif analysis_option == "AI Analysis":
            st.subheader("AI Analysis")
            st.write("AI-powered analysis can be implemented here.")
            













