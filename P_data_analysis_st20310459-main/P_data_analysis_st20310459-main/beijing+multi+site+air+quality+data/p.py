import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue

# Data Handling Functions
# make predidication  make the condition good,excellent
# 
def load_and_merge_data():
    file_paths = [
        r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Dingling_20130301-20170228.csv'
    ]
    dataframes = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def handle_missing_and_duplicates(df):
    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def feature_engineering(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Fall' if 9 <= x <= 11 else 'Winter')
    return df

def data_cleaning(df):
    if 'No' in df.columns:
        df.drop('No', axis=1, inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

def perform_analysis_and_visualization(df):
    pm25_clean = df['PM2.5'].dropna()
    plt.figure(figsize=(10, 6))
    sns.histplot(pm25_clean, bins=30)
    plt.title('Distribution of PM2.5 Concentration')
    plt.xlabel('PM2.5')
    plt.ylabel('Frequency')
    plt.xlim([0, pm25_clean.max()])
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PM2.5', y='PM10', data=df.dropna(subset=['PM2.5', 'PM10']))
    plt.title('Scatter plot of PM2.5 vs. PM10')
    plt.xlabel('PM2.5')
    plt.ylabel('PM10')
    plt.show()

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def model_training_and_prediction(df):
    df = df.dropna(subset=['PM2.5'])
    X = df.drop(columns=['PM2.5'])
    y = df['PM2.5']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, model

# GUI Application Setup
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Air Quality Data Analysis")
        self.geometry("800x600")

        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=20, expand=True)

        # Create tabs
        self.data_overview_tab = ttk.Frame(self.notebook)
        self.eda_tab = ttk.Frame(self.notebook)
        self.modeling_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.data_overview_tab, text="Data Overview")
        self.notebook.add(self.eda_tab, text="EDA")
        self.notebook.add(self.modeling_tab, text="Modeling")

        # Initialize data variable
        self.data = None  
        self.queue = queue.Queue()
        self.create_data_overview_tab()
        self.create_eda_tab()
        self.create_modeling_tab()
        self.after(100, self.check_queue)

    def create_data_overview_tab(self):
        self.load_data_button = ttk.Button(self.data_overview_tab, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10)

        self.data_display = scrolledtext.ScrolledText(self.data_overview_tab, width=80, height=15)
        self.data_display.pack(pady=10)

    def create_eda_tab(self):
        self.eda_button = ttk.Button(self.eda_tab, text="Perform EDA", command=self.perform_eda)
        self.eda_button.pack(pady=10)

        self.eda_display = scrolledtext.ScrolledText(self.eda_tab, width=80, height=15)
        self.eda_display.pack(pady=10)

    def create_modeling_tab(self):
        self.train_button = ttk.Button(self.modeling_tab, text="Train Model", command=self.start_training_model)
        self.train_button.pack(pady=10)

        self.model_results = tk.StringVar()
        self.results_label = ttk.Label(self.modeling_tab, textvariable=self.model_results, justify=tk.LEFT)
        self.results_label.pack(pady=10)

    def load_data(self):
        try:
            self.data = pd.read_csv(r'C:\Users\Acer\Desktop\st20310459\P_data_analysis_st20310459\beijing+multi+site+air+quality+data\data\data_air_control\PRSA_Data_Dingling_20130301-20170228.csv')
            self.data_display.insert(tk.END, "Data Loaded Successfully.\n")
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found. Please check the file path.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def perform_eda(self):
        if self.data is not None:
            self.eda_display.insert(tk.END, "Performing EDA...\n")
            self.eda_display.insert(tk.END, f"Data Overview:\n{self.data.info()}\n")
            handle_missing_and_duplicates(self.data)
            feature_engineering(self.data)
            data_cleaning(self.data)
            perform_analysis_and_visualization(self.data)
        else:
            messagebox.showwarning("Warning", "Please load data first.")

    def start_training_model(self):
        if self.data is not None:
            threading.Thread(target=self.train_model).start()
        else:
            messagebox.showwarning("Warning", "Please load data first.")

    def train_model(self):
        try:
            mse, r2, best_model = model_training_and_prediction(self.data)
            result_message = f"Model Training Completed:\nMSE: {mse:.2f}\nR²: {r2:.2f}"
            self.queue.put(result_message)
            self.model_results.set(result_message)
        except Exception as e:
            error_message = f"An error occurred during model training:\n{str(e)}"
            self.queue.put(error_message)
            self.model_results.set(error_message)

    def check_queue(self):
        try:
            message = self.queue.get_nowait()
        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_queue)

def build_and_evaluate_model(df):
    # Define the features and target variable
    X = df.drop('PM2.5', axis=1)  # Use all columns except the target
    y = df['PM2.5']  # Target variable

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['category']).columns.tolist()

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Feature scaling
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # Hyperparameter tuning
    param_distributions = {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__max_depth': [10, 20, 30],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }

    search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='r2', verbose=2, random_state=42)
    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")

    # Evaluate the model
    y_pred = search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

# Run the application
if __name__ == "__main__":
    app = Application()
    app.mainloop()
