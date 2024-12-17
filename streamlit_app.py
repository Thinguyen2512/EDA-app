import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates
from statsmodels.graphics.mosaicplot import mosaic
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import uuid 
from scipy.stats import ttest_1samp
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import base64
import os
import openai
from openai import OpenAI

# Set up OpenAI API Key (Replace with your API Key)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to save chart and return Base64 encoded string
def chart_to_base64(fig, filename="chart.jpg"):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=300, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded

# Function to send image Base64 and prompt for AI Analysis
import openai

def ai_analysis(image_base64, data_summary, trend_prediction):
    # Make sure you use the correct model, like "gpt-3.5-turbo" or "gpt-4"
    response = openai.ChatCompletion.create(
        model="gpt-4",  # You can choose "gpt-4" or "gpt-3.5-turbo" depending on your plan
        messages=[
            {"role": "system", "content": "You are an AI analyst who provides detailed insights on charts and predicts future trends based on patterns."},
            {"role": "user", "content": f"Analyze this chart with the following details:\n\n{data_summary}\n\nThe data shows a trend of {trend_prediction}. Predict what might happen in the future."}
        ],
        max_tokens=300,  # You can adjust this depending on the output length you need
        temperature=0.7   # Adjust temperature for more or less creativity
    )

    # Return the AI analysis response
    return response['choices'][0]['message']['content']


# Predict future trend using Linear Regression
def predict_trend(data, column):
    # Prepare data for prediction
    x = np.array(range(len(data))).reshape(-1, 1)  # Time index as X
    y = data[column].dropna().values  # Target variable
    if len(y) < 2:
        return "Not enough data to predict trend."

    # Fit a simple Linear Regression model
    model = LinearRegression()
    model.fit(x[:len(y)], y)

    # Predict future values
    future_x = np.array(range(len(data) + 10)).reshape(-1, 1)  # Next 10 points
    future_y = model.predict(future_x)

    return "The trend shows a general increase." if model.coef_[0] > 0 else "The trend shows a decrease."

# Add AI Analysis to the chart
def add_ai_analysis(fig, data, selected_column, title="AI Analysis"):
    # Save chart to Base64
    image_base64 = chart_to_base64(fig)
    
    # Summarize data statistics
    data_summary = data[selected_column].describe().to_string()
    trend_prediction = predict_trend(data, selected_column)

    # Display chart
    st.write("### Generated Chart:")
    st.pyplot(fig)

    # AI Analysis
    if st.button("Run AI Analysis"):
        st.write("### AI Analysis Result:")
        analysis_result = ai_analysis(image_base64, data_summary, trend_prediction)
        st.write(analysis_result)

# Ensure that your column is cleaned and contains valid numeric values
def clean_and_validate_column(data, column):
    # Convert to numeric, forcing errors to NaN
    data[column] = pd.to_numeric(data[column], errors='coerce')
    clean_data = data[column].dropna()  # Drop NaN values
    return clean_data
    
# Helper function to save plot as JPG
def save_plot_as_jpg(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

# Function for combined variable comparison
def plot_combined_comparison(data, selected_columns, plot_type):
    plt.figure(figsize=(12, 6))
    
    if plot_type == "Density Plot":
        for idx, col in enumerate(selected_columns):
            if np.issubdtype(data[col].dtype, np.number):  # Check if the column is numeric
                sns.kdeplot(data[col], label=col, shade=True, alpha=0.6)
                
        plt.title("Combined Density Plot")
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.legend(title="Variables", loc="upper right")

    elif plot_type == "Boxplot":
        sns.boxplot(data=data[selected_columns], orient="h")
        plt.title("Combined Box Plot")
        plt.xlabel("Values")
        plt.ylabel("Variables")

    st.pyplot(plt)
    
# Helper function to generate valid filenames
def generate_valid_filename(name):
    return ''.join(e if e.isalnum() else '_' for e in name)

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

# Helper function to clean and convert columns to numeric if they contain symbols
def clean_numeric_column(data, column):
    data[column] = pd.to_numeric(data[column].replace({r'[^\d.]': ''}, regex=True), errors='coerce')
    return data[column]
    
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
            "Variables Comparison",
            "Subgroup Analysis",
            "Linear Regression"
        ])

        # General filter with multiple options
        filter_col = st.sidebar.selectbox("Filter by Column", ["None"] + data.columns.tolist())
        if filter_col != "None":
            if pd.api.types.is_numeric_dtype(data[filter_col]):  # Numerical data
                min_val, max_val = st.sidebar.slider(
                    f"Select range for {filter_col}",
                    float(data[filter_col].min()),
                    float(data[filter_col].max()),
                    (float(data[filter_col].min()), float(data[filter_col].max()))
                )
                data = data[(data[filter_col] >= min_val) & (data[filter_col] <= max_val)]
            else:  # Categorical data
                # Handle categorical filtering with "Select All"
                unique_values = list(data[filter_col].dropna().unique())  # Ensure it's a list
                if unique_values:  # Only proceed if there are unique values
                    all_selected = st.sidebar.checkbox(f"Select All {filter_col}", value=True)

                    if all_selected:
                        selected_values = unique_values  # Select all values
                    else:
                        selected_values = st.sidebar.multiselect(
                            f"Select values for {filter_col}",
                            options=unique_values,
                            default=[]  # No default values
                        )
                    # Apply filter only if selected_values is not empty
                    if selected_values:
                        data = data[data[filter_col].isin(selected_values)]
                else:
                    st.warning(f"No unique values found in column {filter_col}.")

        # Display filtered data
        st.write("Filtered Data:")
        st.dataframe(data)
            
        # Summary Statistics
        if analysis_option == "Summary Statistics":
            st.subheader("Summary Statistics")
            st.write(data.describe())

        # Plot One Variable
        elif analysis_option == "Plot One Variable":
            st.subheader("Plot One Variable")
            feature = st.selectbox("Select variable to plot:", data.columns)

            if np.issubdtype(data[feature].dtype, np.number):
                st.write("### Numerical Variable Options")
                plot_type = st.selectbox("Select plot type:", ["Histogram", "Box Plot"])
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
                plot_type = st.selectbox("Select plot type:", ["Bar Chart", "Pie Chart"])
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

            # Add a button to download the plot as JPG
            if st.button("Download Plot as JPG"):
                valid_feature_name = generate_valid_filename(feature)  # Ensure valid filename
                buf = save_plot_as_jpg(plt.gcf())
                st.download_button(
                    label="Download JPG",
                    data=buf,
                    file_name=f"{valid_feature_name}_plot.jpg",  # Use valid file name
                    mime="image/jpeg",
                    key=str(uuid.uuid4())  # Ensure the key is unique
                )

        # Plot Two Variables
        elif analysis_option == "Plot Two Variables":
            st.subheader("Plot Two Variables")
            x_axis = st.selectbox("Select X variable:", data.columns)
            y_axis = st.selectbox("Select Y variable:", data.columns, index=1)

            x_is_numeric = np.issubdtype(data[x_axis].dtype, np.number)
            y_is_numeric = np.issubdtype(data[y_axis].dtype, np.number)

            if x_is_numeric and y_is_numeric:
                st.write("### Two Numerical Variables Options")
                plot_type = st.selectbox("Select plot type:", ["Scatter Plot", "Line Graph", "Area Chart"])
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

            elif x_is_numeric and not y_is_numeric:
                st.write("### One Numerical and One Categorical Variable Options")
                plot_type = st.selectbox("Select plot type:", ["Bar Chart"])
                plt.figure(figsize=(10, 6))

                if plot_type == "Bar Chart":
                    grouped_data = data.groupby(y_axis)[x_axis].mean().sort_values()
                    grouped_data.plot(kind='bar')
                    plt.title(f'Bar Chart of {x_axis} by {y_axis}')
                    plt.xlabel(y_axis)
                    plt.ylabel(x_axis)

                st.pyplot(plt)

            elif not x_is_numeric and not y_is_numeric:
                st.write("### Two Categorical Variables Options")
                plot_type = st.selectbox("Select plot type:", ["Grouped Bar Chart", "Mosaic Plot"])
                plt.figure(figsize=(10, 6))

                if plot_type == "Grouped Bar Chart":
                    data.groupby([x_axis, y_axis]).size().unstack().plot(kind='bar', stacked=True)
                    plt.title(f'Grouped Bar Chart of {x_axis} and {y_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel('Count')

                elif plot_type == "Mosaic Plot":
                    mosaic(data, [x_axis, y_axis])
                    plt.title(f'Mosaic Plot of {x_axis} and {y_axis}')

                st.pyplot(plt)

            # Add a button to download the plot as JPG
            if st.button("Download Plot as JPG"):
                valid_feature_name = generate_valid_filename(f"{x_axis}_vs_{y_axis}")  # Ensure valid filename
                buf = save_plot_as_jpg(plt.gcf())
                st.download_button(
                    label="Download JPG",
                    data=buf,
                    file_name=f"{valid_feature_name}_plot.jpg",  # Use valid file name
                    mime="image/jpeg",
                    key=str(uuid.uuid4())  # Ensure the key is unique
                )

       # Plot Three Variables
        elif analysis_option == "Plot Three Variables":
            st.subheader("Plot Three Variables")
            st.write("Select variables to visualize and assign them to the X, Y, and Z axes.")

            x_axis = st.selectbox("Select variable for X axis:", data.columns)
            y_axis = st.selectbox("Select variable for Y axis:", data.columns, index=1)
            z_axis = st.selectbox("Select variable for Z axis (e.g., size, color, or value):", data.columns, index=2)

            x_is_numeric = np.issubdtype(data[x_axis].dtype, np.number)
            y_is_numeric = np.issubdtype(data[y_axis].dtype, np.number)
            z_is_numeric = np.issubdtype(data[z_axis].dtype, np.number)

            if x_is_numeric and y_is_numeric and z_is_numeric:
                st.write("### All Variables Are Numerical")
                plot_choice = st.selectbox("Select plot type:", ["3D Scatter Plot", "Contour Plot", "Bubble Chart"])

                if plot_choice == "3D Scatter Plot":
                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(data[x_axis], data[y_axis], data[z_axis])
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    ax.set_zlabel(z_axis)
                    plt.title('3D Scatter Plot of Selected Variables')
                    st.pyplot(fig)

                elif plot_choice == "Contour Plot":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = data[x_axis].dropna().values
                    y = data[y_axis].dropna().values
                    z = data[z_axis].dropna().values
                    X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
                    Z = griddata((x, y), z, (X, Y), method='linear')
                    contour = ax.contourf(X, Y, Z, cmap='coolwarm')
                    plt.colorbar(contour, ax=ax)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    plt.title(f'Contour Plot of {z_axis} by {x_axis} and {y_axis}')
                    st.pyplot(fig)

                elif plot_choice == "Bubble Chart":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(data[x_axis], data[y_axis], s=data[z_axis]*10, alpha=0.6, c=data[z_axis], cmap='coolwarm', edgecolors='w')
                    plt.colorbar(scatter, ax=ax, label=z_axis)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    plt.title(f'Bubble Chart of {y_axis} vs {x_axis} with Size by {z_axis}')
                    st.pyplot(fig)

            elif not x_is_numeric and not y_is_numeric and z_is_numeric:
                st.write("### Two Categorical Variables and One Numerical Variable")
                plot_choice = st.selectbox("Select plot type:", ["Grid Plot (Heatmap)"])

                if plot_choice == "Grid Plot (Heatmap)":
                    grid_data = data.pivot_table(values=z_axis, index=x_axis, columns=y_axis, aggfunc='mean')
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(grid_data, annot=True, cmap="coolwarm", fmt=".1f")
                    plt.title(f'Heatmap of {z_axis} by {x_axis} and {y_axis}')
                    plt.xlabel(y_axis)
                    plt.ylabel(x_axis)
                    st.pyplot(plt)

            elif x_is_numeric and y_is_numeric and not z_is_numeric:
                st.write("### Two Numerical Variables and One Categorical Variable")
                plot_choice = st.selectbox("Select plot type:", ["Stacked Bar Chart", "Stacked Column Chart"])

                if plot_choice == "Stacked Bar Chart":
                    stacked_data = data.groupby([z_axis, x_axis])[y_axis].sum().unstack()
                    plt.figure(figsize=(10, 6))
                    stacked_data.plot(kind='bar', stacked=True)
                    plt.title(f'Stacked Bar Chart of {y_axis} vs {x_axis} grouped by {z_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                    st.pyplot(plt)

                elif plot_choice == "Stacked Column Chart":
                    stacked_data = data.groupby([z_axis, x_axis])[y_axis].sum().unstack()
                    plt.figure(figsize=(10, 6))
                    stacked_data.plot(kind='barh', stacked=True)
                    plt.title(f'Stacked Column Chart of {y_axis} vs {x_axis} grouped by {z_axis}')
                    plt.ylabel(x_axis)
                    plt.xlabel(y_axis)
                    st.pyplot(plt)

            # Add a button to download the plot as JPG
            if st.button("Download Plot as JPG"):
                valid_feature_name = generate_valid_filename(f"{x_axis}_vs_{y_axis}_vs_{z_axis}")  # Ensure valid filename
                buf = save_plot_as_jpg(fig)
                st.download_button(
                    label="Download JPG",
                    data=buf,
                    file_name=f"{valid_feature_name}_plot.jpg",  # Use valid file name
                    mime="image/jpeg",
                    key=str(uuid.uuid4())  # Ensure the key is unique
                )

         # Variables Comparison
        elif analysis_option == "Variables Comparison":
            st.subheader("Variables Comparison")
            selected_columns = st.multiselect("Select variables for comparison", data.columns.tolist())
            plot_type = st.selectbox("Select comparison plot type", ["Density Plot", "Boxplot"])
            if selected_columns:
                plt.figure(figsize=(12, 6))
                plot_combined_comparison(data, selected_columns, plot_type)
                fig = plt.gcf()
                selected_column = selected_columns[0]
                add_ai_analysis(fig, data, selected_column, title="AI Analysis for Variables Comparison")
                
        # Add a button to download the plot as JPG
            if st.button("Download Plot as JPG", key="variables_comparison_download"):
                valid_feature_name = generate_valid_filename('_'.join(selected_columns))  # Đảm bảo tên hợp lệ
                buf = save_plot_as_jpg(fig)
                st.download_button(
                    label="Download JPG",
                    data=buf,
                    file_name=f"{valid_feature_name}_combined_plot.jpg",
                    mime="image/jpeg",
                    key="variables_comparison_file"
                )

# Subgroup Analysis
        elif analysis_option == "Subgroup Analysis":
            st.sidebar.header("Subgroup Analysis Settings")
            subgroup_col = st.sidebar.selectbox("Select Subgroup Column", data.columns)
            metric_col = st.sidebar.selectbox("Select Metric Column", data.columns)
            try:
                data[metric_col] = pd.to_numeric(data[metric_col], errors='coerce')
                data = data[[subgroup_col, metric_col]].dropna()
                if data.empty:
                    st.warning("No valid data available for the selected columns. Please check your input data.")
                else:
                    subgroup_stats = data.groupby(subgroup_col)[metric_col].agg(['mean', 'sum', 'std']).reset_index()
                    st.write("### Subgroup Statistics")
                    st.dataframe(subgroup_stats)
                    chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart"])
                    if chart_type == "Bar Chart":
                        metric = st.sidebar.selectbox("Select Metric for Bar Chart", ['mean', 'sum', 'std'])
                        if metric in subgroup_stats.columns:
                            plt.figure(figsize=(10, 6))
                            sns.barplot(x=subgroup_col, y=metric, data=subgroup_stats)
                            plt.title(f"Bar Chart of {metric.capitalize()} by {subgroup_col}")
                            plt.ylabel(metric.capitalize())
                            plt.xlabel(subgroup_col)
                            fig = plt.gcf()
                            st.pyplot(fig)
                        else:
                            st.error(f"Metric '{metric}' does not exist in the subgroup statistics.")
                    elif chart_type == "Pie Chart":
                        metric = st.sidebar.selectbox("Select Metric for Pie Chart", ['mean', 'sum', 'std'])
                        if metric in subgroup_stats.columns:
                            subgroup_stats = subgroup_stats.dropna(subset=[metric])
                            if subgroup_stats.empty:
                                st.warning("No valid data available for the selected metric.")
                            else:
                                fig, ax = plt.subplots(figsize=(8, 8))
                                ax.pie(
                                    subgroup_stats[metric],
                                    labels=subgroup_stats[subgroup_col],
                                    autopct='%1.1f%%',
                                    startangle=140
                                )
                                ax.set_title(f"Pie Chart of {metric.capitalize()} by {subgroup_col}")
                                ax.axis('equal')
                                st.pyplot(fig)
                        else:
                            st.error(f"Metric '{metric}' does not exist in the subgroup statistics.")
                                
            except ValueError as e:
                st.error(f"An error occurred: {str(e)}. Please check your data and try again.")

        # Add a button to download the plot as JPG
            if st.button("Download Plot as JPG"):
                valid_feature_name = generate_valid_filename(feature)  # Ensure valid filename
                buf = save_plot_as_jpg(fig)
                st.download_button(
                    label="Download JPG",
                    data=buf,
                    file_name=f"{valid_feature_name}_plot.jpg",  # Use valid file name
                    mime="image/jpeg",
                    key=str(uuid.uuid4())  # Ensure the key is unique
                )
                
# Linear Regression Analysis
        elif analysis_option == "Linear Regression":
            st.subheader("Linear Regression Analysis")

            # Separate numeric and categorical columns
            numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
            
            # Handle categorical columns
            if cat_cols:
                st.sidebar.header("Categorical Variables")
                selected_cat_cols = st.sidebar.multiselect("Select columns to create dummy variables:", cat_cols)
                if selected_cat_cols:
                    data = pd.get_dummies(data, columns=selected_cat_cols, drop_first=True)
            
            # Only Multiple Regression        
            x_cols = st.multiselect("Select Independent Variables (X):", numeric_cols)
            y_col = st.selectbox("Select Dependent Variable (Y):", numeric_cols)

            if x_cols and y_col:
                # Drop missing values
                X = data[x_cols].dropna()
                y = data[y_col].dropna()
                
                # Find common index to ensure both X and y have matching rows
                common_index = X.index.intersection(y.index)
                X = X.loc[common_index]
                y = y.loc[common_index]
                
                # Add constant (intercept) to independent variables
                X = sm.add_constant(X)
                
                # Fit the model
                model = sm.OLS(y, X).fit()
                st.write("### Regression Results")
                st.write(
                    pd.DataFrame(
                        {
                            "Coefficients": model.params,
                            "P-Values": model.pvalues,
                            "T-Statistics": model.tvalues,
                            "Confidence Interval (2.5%)": model.conf_int()[0],
                            "Confidence Interval (97.5%)": model.conf_int()[1],
                        }
                    )
                )
                st.write("### Model Performance")
                st.write(f"R-squared: {model.rsquared:.4f}")
                st.write(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
                st.write(f"F-statistic: {model.fvalue:.2f}")
                st.write(f"F-statistic p-value: {model.f_pvalue:.4e}")
                
                # Variance Inflation Factor (VIF)
                st.write("### Multicollinearity Check (VIF)")
                vif_data = pd.DataFrame()
                vif_data["Feature"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                st.table(vif_data)
                
                # Residual Plot
                st.write("### Residual Plot")
                residuals = model.resid
                fig, ax = plt.subplots()
                sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, ax=ax, line_kws={"color": "red"})
                ax.set_xlabel("Fitted Values")
                ax.set_ylabel("Residuals")
                ax.set_title("Residuals vs Fitted")
                st.pyplot(fig)
                
                # Add AI Analysis
                add_ai_analysis(fig, title="AI Analysis for Residual Plot")

                    # Add a button to download the plot as JPG
            if st.button("Download Plot as JPG"):
                valid_feature_name = generate_valid_filename(feature)  # Ensure valid filename
                buf = save_plot_as_jpg(fig)
                st.download_button(
                    label="Download JPG",
                    data=buf,
                    file_name=f"{valid_feature_name}_plot.jpg",  # Use valid file name
                    mime="image/jpeg",
                    key=str(uuid.uuid4())  # Ensure the key is unique
                )
            
# Contact Us section
elif choice == "Contact Us":
    st.subheader("Contact Us")
    st.write("For inquiries, please email us at contact@example.com.")















