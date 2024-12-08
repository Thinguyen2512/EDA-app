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
            plot_combined_comparison(data, selected_columns, plot_type)

            # Add a button to download the plot as JPG
            if st.button("Download Plot as JPG"):
                valid_feature_name = generate_valid_filename('_'.join(selected_columns))  # Ensure valid filename
                buf = save_plot_as_jpg(plt.gcf())
                st.download_button(
                    label="Download JPG",
                    data=buf,
                    file_name=f"{valid_feature_name}_combined_plot.jpg",  # Use valid file name
                    mime="image/jpeg"
                )

    # Subgroup Analysis
        elif analysis_option == "Subgroup Analysis":
            st.sidebar.header("Subgroup Analysis Settings")
            subgroup_col = st.sidebar.selectbox("Select Subgroup Column", data.columns)
            metric_col = st.sidebar.selectbox("Select Metric Column", data.columns)
            data[metric_col] = pd.to_numeric(data[metric_col], errors='coerce')

            if subgroup_col and metric_col:
                # Drop NA values for selected columns
                data = data[[subgroup_col, metric_col]].dropna()
                if data.empty:
                    st.warning("No valid data available for the selected columns.")
                else:
                    subgroup_stats = data.groupby(subgroup_col)[metric_col].agg(['mean', 'sum', 'std']).reset_index()
                    st.write("### Subgroup Statistics")
                    st.dataframe(subgroup_stats)

                st.write(f"Data types: {data.dtypes}")
                st.write(f"Data sample: {data[[subgroup_col, metric_col]].head()}")

    # Chart type selection
                chart_type = st.sidebar.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart"])

                # Bar chart for mean, total, and standard deviation
            if chart_type == "Bar Chart":
                metric = st.sidebar.selectbox("Select Metric", ['mean', 'sum', 'std'])
                
                # Check if the metric column exists
                if metric in subgroup_stats.columns:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=subgroup_col, y=metric, data=subgroup_stats)
                    plt.title(f"Bar Chart of {metric.capitalize()} by {subgroup_col}")
                    plt.ylabel(metric.capitalize())
                    plt.xlabel(subgroup_col)
                    st.pyplot(plt)
                else:
                    st.error(f"Metric '{metric}' does not exist in the data.")

                # Pie chart for total values
            elif chart_type == "Pie Chart":
                if 'sum' in subgroup_stats.columns and subgroup_col in subgroup_stats.columns:
                    # Ensure that 'sum' is numeric
                    subgroup_stats['sum'] = pd.to_numeric(subgroup_stats['sum'], errors='coerce')
                    # Drop any rows with NaN values (if any) in case conversion failed
                    subgroup_stats = subgroup_stats.dropna(subset=['sum'])

                    # Create the pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust the figure size if needed
                    ax.pie(subgroup_stats['sum'], labels=subgroup_stats[subgroup_col], autopct='%1.1f%%', startangle=140)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)
                else:
                    st.error(f"The necessary columns ('sum' and '{subgroup_col}') are not found in the data.")

            else:
                st.warning("Please select both a subgroup column and a metric column.")

# Linear Regression Section
        elif analysis_option == "Linear Regression":
            st.subheader("Linear Regression Analysis")

            # Select Regression Type using selectbox
            regression_type = st.selectbox("Choose Regression Type:", ["Simple Regression", "Multiple Regression"])

            # Get numeric columns from the data
            num_list = data.select_dtypes(include=["number"]).columns.tolist()

            if regression_type == "Simple Regression":
                st.markdown("### Simple Linear Regression")
                x_col = st.selectbox("Select Independent Variable (X):", num_list)
                y_col = st.selectbox("Select Dependent Variable (Y):", num_list)

                if x_col and y_col:
                    # Prepare data by aligning both variables' indexes
                    X = data[[x_col]].dropna()  # Independent variable (X)
                    y = data[y_col].dropna()  # Dependent variable (Y)
                    common_index = X.index.intersection(y.index)
                    X = X.loc[common_index]
                    y = y.loc[common_index]

                    # Fit the linear regression model
                    model = LinearRegression()
                    model.fit(X, y)

                    # Make predictions and calculate performance metrics
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    coef = model.coef_[0]
                    intercept = model.intercept_

                    # Display regression equation and R-squared value
                    st.write(f"**Regression Equation:** Y = {intercept:.2f} + {coef:.2f} * X")
                    st.write(f"**R-squared:** {r2:.2f}")

                    # Plot the regression line along with data points
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=X[x_col], y=y, ax=ax, label="Data")
                    sns.lineplot(x=X[x_col], y=y_pred, color="red", label="Regression Line", ax=ax)
                    ax.set_title(f"Simple Linear Regression: {y_col} vs {x_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)

            elif regression_type == "Multiple Regression":
                st.markdown("### Multiple Linear Regression")
                x_cols = st.multiselect("Select Independent Variables (X):", num_list)
                y_col = st.selectbox("Select Dependent Variable (Y):", num_list)

                if x_cols and y_col:
                    # Prepare data by ensuring matching indexes
                    X = data[x_cols].dropna()  # Independent variables (X)
                    y = data[y_col].dropna()  # Dependent variable (Y)
                    common_index = X.index.intersection(y.index)
                    X = X.loc[common_index]
                    y = y.loc[common_index]

                    # Fit the multiple regression model
                    model = LinearRegression()
                    model.fit(X, y)

                    # Make predictions and calculate model metrics
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)

                    # Display R-squared
                    st.write(f"**R-squared:** {r2:.2f}")

                    # Display model coefficients
                    st.markdown("### Model Coefficients")
                    coef_df = pd.DataFrame({
                        "Variable": ["Intercept"] + x_cols,
                        "Coefficient": [model.intercept_] + list(model.coef_)
                    })
                    st.table(coef_df)

                    # Plot the regression in a 3D plot if there are exactly two independent variables
                    if len(x_cols) == 2:
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(X[x_cols[0]], X[x_cols[1]], y, color="blue", label="Data")
                        
                        # Create meshgrid for plotting regression surface
                        x1_range = X[x_cols[0]].values
                        x2_range = X[x_cols[1]].values
                        X1, X2 = np.meshgrid(np.linspace(min(x1_range), max(x1_range), 30),
                                             np.linspace(min(x2_range), max(x2_range), 30))
                        X_flat = np.c_[X1.ravel(), X2.ravel()]
                        y_pred_flat = model.predict(X_flat).reshape(X1.shape)
                        
                        # Plot the regression surface
                        ax.plot_surface(X1, X2, y_pred_flat, color='red', alpha=0.5)
                        ax.set_xlabel(x_cols[0])
                        ax.set_ylabel(x_cols[1])
                        ax.set_zlabel(y_col)
                        ax.set_title(f"Multiple Linear Regression: {y_col} vs {x_cols[0]} and {x_cols[1]}")
                        st.pyplot(fig)
                    else:
                        st.write("The plot can only be shown for exactly two independent variables.")
            
# Contact Us section
elif choice == "Contact Us":
    st.subheader("Contact Us")
    st.write("For inquiries, please email us at contact@example.com.")















