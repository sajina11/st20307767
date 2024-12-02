import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load and concatenate data from multiple CSV files
air_quality_data_set_file_paths = [
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Aotizhongxin_20130301-20170228.csv',
    r"C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Changping_20130301-20170228.csv",
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Dingling_20130301-20170228.csv',
  
    r'C:\Users\Dell\OneDrive\Desktop\programmingdata\Dataset Folder for your Final Assessment-20241125\PRSA_Data_Dongsi_20130301-20170228.csv',
  ]

data_set_frames = [pd.read_csv(file) for file in air_quality_data_set_file_paths]
df = pd.concat(data_set_frames)

