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
            "Hypothesis Testing",
            "AI Analysis"
        ])

        # General filter
        filter_col = st.sidebar.selectbox("Filter by Column", ["None"] + data.columns.tolist())
        if filter_col != "None":
            unique_values = data[filter_col].dropna().unique()
            filter_value = st.sidebar.selectbox(f"Filter {filter_col} by:", unique_values)
            data = data[data[filter_col] == filter_value]
            
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

             # Clean the selected features for numerical types
            data[x_axis] = clean_numeric_column(data, x_axis)
            data[y_axis] = clean_numeric_column(data, y_axis)

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
                        x = data[x_var].dropna().values
                        y = data[y_var].dropna().values
                        z = data[z_var].dropna().values

                        # Tạo lưới (grid) cho biểu đồ contour
                        X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))

                        # Dự đoán các giá trị Z cho lưới X, Y (sử dụng interpolation)
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
                        
            # Hypothesis Testing
        if analysis_option == "Hypothesis Testing":
            st.subheader("Hypothesis Testing")

            test_type = st.selectbox("Select Test Type:", ["t-test", "ANOVA", "Chi-Squared Test", "Linear Regression"])

            if test_type == "t-test":
                st.write("### Independent t-test")
                num_cols = data.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) >= 2:
                    col1 = st.selectbox("Select first numerical column:", num_cols)
                    col2 = st.selectbox("Select second numerical column:", num_cols, index=1)
                    t_stat, p_value = ttest_ind(data[col1].dropna(), data[col2].dropna())
                    st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
                    if p_value < 0.05:
                        st.write("Result: Reject null hypothesis (significant difference).")
                    else:
                        st.write("Result: Fail to reject null hypothesis (no significant difference).")
                else:
                    st.write("Not enough numerical columns for t-test.")

            elif test_type == "ANOVA":
                st.write("### One-way ANOVA")
                cat_cols = data.select_dtypes(include='object').columns.tolist()
                num_cols = data.select_dtypes(include=np.number).columns.tolist()
                if cat_cols and num_cols:
                    cat_col = st.selectbox("Select categorical column:", cat_cols)
                    num_col = st.selectbox("Select numerical column:", num_cols)
                    groups = [group[num_col].dropna() for _, group in data.groupby(cat_col)]
                    f_stat, p_value = f_oneway(*groups)
                    st.write(f"F-statistic: {f_stat:.4f}, P-value: {p_value:.4f}")
                    if p_value < 0.05:
                        st.write("Result: Reject null hypothesis (significant difference among groups).")
                    else:
                        st.write("Result: Fail to reject null hypothesis (no significant difference among groups).")
                else:
                    st.write("Not enough data for ANOVA.")

            elif test_type == "Chi-Squared Test":
                st.write("### Chi-Squared Test")
                cat_cols = data.select_dtypes(include='object').columns.tolist()
                if len(cat_cols) >= 2:
                    col1 = st.selectbox("Select first categorical column:", cat_cols)
                    col2 = st.selectbox("Select second categorical column:", cat_cols, index=1)
                    contingency_table = pd.crosstab(data[col1], data[col2])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    st.write(f"Chi-squared statistic: {chi2:.4f}, P-value: {p_value:.4f}")
                    if p_value < 0.05:
                        st.write("Result: Reject null hypothesis (variables are dependent).")
                    else:
                        st.write("Result: Fail to reject null hypothesis (variables are independent).")
                else:
                    st.write("Not enough categorical columns for Chi-Squared Test.")

            elif test_type == "Linear Regression":
                st.write("### Linear Regression")
                num_cols = data.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) >= 2:
                    x_col = st.selectbox("Select independent variable (X):", num_cols)
                    y_col = st.selectbox("Select dependent variable (Y):", num_cols, index=1)
                    X = data[[x_col]].dropna()
                    y = data[y_col].dropna()
                    if len(X) == len(y):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write(f"R-squared: {r2_score(y_test, y_pred):.4f}")
                        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
                    else:
                        st.write("Mismatch in data length between X and Y.")
                else:
                    st.write("Not enough numerical columns for regression.")
                    
        # AI Analysis Placeholder (Optional)
        elif analysis_option == "AI Analysis":
            st.subheader("AI-based Analysis: Predictive Analysis")
    st.write("Select variables for predictive analysis.")

    # Select input and output features
    features = st.multiselect("Select input features (X):", data.columns)
    target = st.selectbox("Select target variable (y):", data.columns)

    if features and target:
        # Clean numeric columns for modeling
        X = data[features].apply(pd.to_numeric, errors='coerce').dropna()
        y = pd.to_numeric(data[target], errors='coerce').dropna()

        # Align the indices
        X = X.loc[y.index]
        y = y.loc[X.index]

        # Train-Test Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model Selection
        model_choice = st.selectbox("Select model:", ["Linear Regression", "Decision Tree", "Random Forest"])
        if model_choice == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor()
        elif model_choice == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()

        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Metrics
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared (R²):** {r2:.2f}")

        # Visualization
        st.subheader("Prediction vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        plt.title("Prediction vs Actual")
        st.pyplot(fig)

# Contact Us section
elif choice == "Contact Us":
    st.subheader("Contact Us")
    st.write("For inquiries, please email us at contact@example.com.")















