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


