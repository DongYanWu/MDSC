import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Define the path to the training data folder
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace this with your actual folder path

# Create an empty list to store all the dataframes
all_data = []

# Loop through each dielectric material (1 to 13)
for material in range(1, 14):
    material_path = os.path.join(base_path, str(material))
    
    # Loop through each voltage condition (a, b, c, d)
    for voltage in ['a', 'b', 'c', 'd']:
        file_path = os.path.join(material_path, f'{voltage}.csv')
        
        # Load the CSV file into a dataframe
        df = pd.read_csv(file_path)
        
        # Add metadata to the dataframe: material and voltage condition
        df['material'] = material
        df['voltage'] = voltage
        
        # Append the dataframe to the list
        all_data.append(df)

# Concatenate all the dataframes into one big dataframe
training_data = pd.concat(all_data, ignore_index=True)

# Now 'training_data' contains all the CSV files from the folder structure
# You can inspect the first few rows
# print(training_data.head())

#Linear Regression 
# Step 1: Prepare the training data
# Assuming 'training_data' is loaded using the previous code with material and voltage conditions
# Let's prepare the data by flattening the 4000 rows (time series) into features
X_train = []
y_train = []

for material in training_data['material'].unique():
    for voltage in training_data['voltage'].unique():
        # Filter data for this specific material and voltage condition
        data = training_data[(training_data['material'] == material) & 
                             (training_data['voltage'] == voltage)]
        
        # Use the first 50 rows as features (input)
        X = data.iloc[:50, 1:11].values.flatten()
        
        # Use the remaining 3950 rows as the target (output to predict)
        y = data.iloc[50:, 1:11].values.flatten()
        
        # Append to training lists
        X_train.append(X)
        y_train.append(y)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Step 2: Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Step 4: Make predictions on the validation set
y_pred = model.predict(X_val_split)

# Step 5: Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_val_split, y_pred)
print(f'Mean Squared Error on validation set: {mse}')