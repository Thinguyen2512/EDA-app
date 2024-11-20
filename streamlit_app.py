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

# Plot One Variable
        elif analysis_option == "Plot One Variable":
            st.subheader("Plot One Variable")
            st.write("Select a variable to visualize. Choose a numerical or categorical variable.")
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
                    plt.title(f'Box Plot of {feature}')
                st.pyplot()

            else:
                st.write("### Categorical Variable Options")
                plot_type = st.selectbox("Select plot type:", ["Bar Chart", "Pie Chart"])

                plt.figure(figsize=(10, 6))
                if plot_type == "Bar Chart":
                    data[feature].value_counts().plot(kind='bar', color='skyblue')
                    plt.title(f'Bar Chart of {feature}')
                    plt.xlabel(feature)
                    plt.ylabel('Count')
                elif plot_type == "Pie Chart":
                    data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
                    plt.title(f'Pie Chart of {feature}')
                    plt.ylabel('')  # Hide y-label for better appearance
                st.pyplot()

        # Plot Two Variables
        elif analysis_option == "Plot Two Variables":
            st.subheader("Plot Two Variables")
            st.write("Select two variables to visualize their relationship.")

            # Dropdown for selecting two variables
            x_axis = st.selectbox("Select X variable:", data.columns)
            y_axis = st.selectbox("Select Y variable:", data.columns)

            # Determine data types
            x_is_numeric = np.issubdtype(data[x_axis].dtype, np.number)
            y_is_numeric = np.issubdtype(data[y_axis].dtype, np.number)

            if x_is_numeric and y_is_numeric:
                st.write("### Two Numerical Variables Options")
                plot_type = st.selectbox("Select plot type:", ["Scatter Plot", "Line Graph", "Area Chart"])

                plt.figure(figsize=(10, 6))
                if plot_type == "Scatter Plot":
                    sns.scatterplot(data=data, x=x_axis, y=y_axis)
                    plt.title(f'Scatter Plot of {y_axis} vs {x_axis}')
                elif plot_type == "Line Graph":
                    sns.lineplot(data=data, x=x_axis, y=y_axis)
                    plt.title(f'Line Graph of {y_axis} vs {x_axis}')
                elif plot_type == "Area Chart":
                    plt.fill_between(data[x_axis], data[y_axis], alpha=0.5)
                    plt.title(f'Area Chart of {y_axis} vs {x_axis}')
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                st.pyplot()

            elif (x_is_numeric and not y_is_numeric) or (not x_is_numeric and y_is_numeric):
                st.write("### Numerical and Categorical Variables")
                plt.figure(figsize=(10, 6))
                sns.barplot(x=x_axis if not x_is_numeric else y_axis,
                            y=y_axis if not x_is_numeric else x_axis,
                            data=data, ci=None)
                plt.title(f'Bar Chart of {y_axis} vs {x_axis}')
                plt.xlabel(x_axis if not x_is_numeric else y_axis)
                plt.ylabel(y_axis if not x_is_numeric else x_axis)
                st.pyplot()

            elif not x_is_numeric and not y_is_numeric:
                st.write("### Two Categorical Variables")
                plot_type = st.selectbox("Select plot type:", ["Grouped Bar Chart", "Mosaic Plot"])

                plt.figure(figsize=(10, 6))
                if plot_type == "Grouped Bar Chart":
                    data_grouped = data.groupby([x_axis, y_axis]).size().unstack(fill_value=0)
                    data_grouped.plot(kind='bar', stacked=True, figsize=(10, 6), color=sns.color_palette("pastel"))
                    plt.title(f'Grouped Bar Chart of {y_axis} by {x_axis}')
                    plt.xlabel(x_axis)
                    plt.ylabel('Count')
                elif plot_type == "Mosaic Plot":
                    mosaic(data, [x_axis, y_axis])
                    plt.title(f'Mosaic Plot of {x_axis} and {y_axis}')
                st.pyplot()

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
        elif analysis_option == "Hypothesis Testing":
            st.subheader("Hypothesis Testing")

            test_type = st.selectbox("Select Test Type:", ["t-test", "ANOVA", "Chi-Squared Test", "Linear Regression"])

            if test_type == "t-test":
                st.write("### Independent t-test")
                num_cols = data.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) >= 2:
                    col1 = st.selectbox("Select first numerical column:", num_cols)
                    col2 = st.selectbox("Select second numerical column:", num_cols, index=1)
                    # Perform t-test
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
                    
                    # Group by categorical column and filter out groups with fewer than 2 data points
                    groups = [group[num_col].dropna() for _, group in data.groupby(cat_col)]
                    
                    # Filter groups that have at least 2 data points
                    groups = [group for group in groups if len(group) > 1]
                    
                    if len(groups) > 1:  # Ensure there are enough groups for ANOVA
                        f_stat, p_value = f_oneway(*groups)
                        st.write(f"F-statistic: {f_stat:.4f}, P-value: {p_value:.4f}")
                        if p_value < 0.05:
                            st.write("Result: Reject null hypothesis (significant difference among groups).")
                        else:
                            st.write("Result: Fail to reject null hypothesis (no significant difference among groups).")
                    else:
                        st.write("Not enough valid groups for ANOVA.")
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
                features = st.multiselect("Select input features (X):", data.columns)
                target = st.selectbox("Select target variable (y):", data.columns)

                if features and target:
                    st.write(f"Selected input features: {features}")
                    st.write(f"Selected target variable: {target}")
                    
                    # Clean numeric columns for modeling
                    X = data[features].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
                    y = pd.to_numeric(data[target], errors='coerce')  # Convert target to numeric

                    # Drop rows with NaN values in either X or y
                    X = X.dropna()
                    y = y.dropna()

                    # Ensure that X and y have the same length (drop rows where either X or y has NaN)
                    X = X.loc[y.index]
                    y = y.loc[X.index]

                    # Now you can proceed with train_test_split safely
                    if len(X) > 0 and len(y) > 0:  # Check if data is available
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                        # Select model type
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
                    else:
                        st.write("There is not enough data to train the model.")

        # AI Analysis Placeholder (Optional)
        elif analysis_option == "AI Analysis":
            st.subheader("AI-based Analysis Placeholder")
            st.write("AI analysis options will be available soon.")


# Contact Us section
elif choice == "Contact Us":
    st.subheader("Contact Us")
    st.write("For inquiries, please email us at contact@example.com.")















