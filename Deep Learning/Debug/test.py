import os
import zipfile
import pandas as pd
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset identifier and the output directory
dataset_identifier = 'jeffheaton/count-the-paperclips'
output_dir = 'data'


# Ensure the Kaggle API key is set up correctly
api_key_path = os.path.expanduser('~/.kaggle/kaggle.json')
if not os.path.exists(api_key_path):
    print(f"API key not found at {api_key_path}. Make sure to place your kaggle.json file there.")
    exit()

# Set the Kaggle API environment variable
os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(api_key_path)

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download and unzip the dataset
api.dataset_download_files(dataset_identifier, path=output_dir, unzip=True)


# Print the files in the output directory to check the structure
for root, dirs, files in os.walk(output_dir):
    for file in files:
        print(os.path.join(root, file))

# Verify downloaded files
expected_files = ['train.csv', 'test.csv', 'clips-dada-2020/clips']
missing_files = []
for file in expected_files:
    file_path = os.path.join(output_dir, file)
    if not os.path.exists(file_path):
        missing_files.append(file)

if missing_files:
    print(f"Missing files: {missing_files}. Please check the dataset structure.")
    exit()

print("Files downloaded and extracted successfully.")

# Load CSV files
train_csv_path = os.path.join(output_dir, 'train.csv')
test_csv_path = os.path.join(output_dir, 'test.csv')
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print("CSV files loaded successfully")

# Example: Display the first few rows of the training dataframe
print(train_df.head())
print(test_df.head())
