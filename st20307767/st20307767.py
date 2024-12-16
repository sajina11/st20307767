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

