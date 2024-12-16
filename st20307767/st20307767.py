import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define paths to all datasets
data_paths = [
       r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    r"C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Changping_20130301-20170228.csv",
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Dingling_20130301-20170228.csv',
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Dongsi_20130301-20170228.csv',
   
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Guanyuan_20130301-20170228.csv',
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Gucheng_20130301-20170228.csv',
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Huairou_20130301-20170228.csv',
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Nongzhanguan_20130301-20170228.csv',
    
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Shunyi_20130301-20170228.csv',
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Tiantan_20130301-20170228.csv',
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Wanliu_20130301-20170228.csv',
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Wanshouxigong_20130301-20170228.csv'

 ]

# Read and merge datasets
def load_and_merge_data(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df['Station'] = path.split("\\")[-1].split("_")[2]  # Extract station name from filename
        dfs.append(df)
    merged_data = pd.concat(dfs, ignore_index=True)
    
    # EDA Preprocessing steps
    # 1. Handle missing values
    merged_data.fillna(method='ffill', inplace=True)  # Forward fill missing values
    
    # 2. Remove duplicate entries
    merged_data.drop_duplicates(inplace=True)
    
    # 3. Feature engineering (example: create a new 'Month' column from the 'year' column)
    merged_data['year'] = pd.to_datetime(merged_data['year'], errors='coerce')
    merged_data['Month'] = merged_data['year'].dt.month  # Extract month from 'year' as a new feature
    
    # Drop rows where 'year' is NaT after conversion
    merged_data.dropna(subset=['year'], inplace=True)
    
    return merged_data

data = load_and_merge_data(data_paths)

# Streamlit App
def main():
    st.title("Beijing Air Quality Data Analysis")

    st.sidebar.title("Options")
    show_data = st.sidebar.checkbox("Data Handling")
    summary_stats = st.sidebar.checkbox("Exploratory Data Analysis (EDA)")
    model_building = st.sidebar.checkbox("Machine Learning Model Building")
    model_evaluation = st.sidebar.checkbox("Model Evaluation")

    # Show basic insights into the dataset
    st.subheader("Dataset Insights")
    rows, columns = data.shape
    st.write(f"The dataset contains **{rows} rows** and **{columns} columns**.")
    
    st.write("The columns in the dataset are:")
    st.write(data.columns.tolist())

    # Show data types and check for missing values
    st.write("\n**Data Types of Each Column:**")
    st.write(data.dtypes)

    missing_values = data.isnull().sum()
    st.write("\n**Missing Values:**")
    st.write(missing_values)

    # Show raw data
    if show_data:
        st.subheader("Raw Merged Data")
        st.dataframe(data.head())

    # Show summary statistics
    if summary_stats:
        st.subheader("Summary Statistics")
        st.write(data.describe())  # Displays summary statistics (mean, std, min, max, etc.)

        # Display graphs for each column
        st.subheader("Visualizations of Features")

        # List of columns to graph (select numeric columns for visualization)
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        for column in numeric_columns:
            st.subheader(f"Visualization for {column}")
            # Plot the distribution of each column
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], kde=True, color='skyblue', bins=30)
            st.pyplot(plt)

            # Box Plot for detecting outliers
            st.subheader(f"Box Plot for {column}")
            plt.figure(figsize=(10, 6))
            sns.boxplot(data[column], color='lightgreen')
            st.pyplot(plt)

    # Map for highest pollution levels
    st.subheader("Map: Highest Pollution Locations")
    
    # Check if latitude and longitude columns exist in the dataset
    if 'latitude' in data.columns and 'longitude' in data.columns:
        # Find the station with the highest pollution (based on PM2.5 or another metric)
        highest_pollution_station = data.loc[data['PM2.5'].idxmax()]

        # Create a map centered on Beijing
        m = folium.Map(location=[39.9042, 116.4074], zoom_start=10)
        marker_cluster = MarkerCluster().add_to(m)

        # Add markers for the station with the highest pollution
        folium.Marker(
            location=[highest_pollution_station['latitude'], highest_pollution_station['longitude']],
            popup=f"Station: {highest_pollution_station['Station']}<br>PM2.5: {highest_pollution_station['PM2.5']}",
            icon=folium.Icon(color='red')
        ).add_to(marker_cluster)

        st.write("The station with the highest pollution is displayed on the map.")
        st.components.v1.html(m._repr_html_(), height=500)
    else:
        st.write("Latitude and Longitude data are missing. Please check the dataset.")

    # Model Building
    if model_building:
        st.subheader("Machine Learning Model Building")

        # Selecting target and features
        target_column = st.selectbox("Select Target Variable:", data.select_dtypes(include=['float64', 'int64']).columns.tolist())

        features = data.drop(columns=['year', 'Station', target_column])  # Drop 'year' and 'Station' columns for features
        target = data[target_column]

        # Encoding categorical features (if any)
        features = pd.get_dummies(features, drop_first=True)

        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

        # Choose model
        model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest", "Other"])

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        else:
            st.warning("Other models can be implemented here.")

        # Fit model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluation
        if model_evaluation:
            st.subheader("Model Evaluation")
            st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
            st.write("R-Squared:", r2_score(y_test, y_pred))

            # Hyperparameter tuning (optional)
            if model_type == "Random Forest":
                st.subheader("Hyperparameter Tuning")
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                }
                grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
                grid_search.fit(X_train, y_train)
                st.write("Best parameters:", grid_search.best_params_)

if __name__ == '__main__':
    main()
