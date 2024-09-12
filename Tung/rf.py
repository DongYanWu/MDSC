import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Define the path to the training data folder
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path

# Create an empty list to store all the dataframes
all_data = []

# Loop through each dielectric material (1 to 13)
for material in range(1, 14):
    material_path = os.path.join(base_path, str(material))
    
    # Loop through each voltage condition (a, b, c, d)
    for voltage in ['a', 'b', 'c', 'd']:
        file_path = os.path.join(material_path, f'{voltage}.csv')
        print(f"Processing file: {file_path}")
        
        # Step 1: Load the CSV file
        data = pd.read_csv(file_path)

        # Step 2: Create time-based features (lag features)
        for current_column in [f'y{i:02d}' for i in range(1, 11)]:
            print(f"Modeling Random Forest for: Material {material}, Voltage {voltage}, Column {current_column}")
            
            # Extract the current values from the column
            current_values = data[current_column].dropna()

            # Create lag features (e.g., use previous 3 time steps as features)
            lagged_data = pd.DataFrame({
                f'{current_column}_lag_1': current_values.shift(1),
                f'{current_column}_lag_2': current_values.shift(2),
                f'{current_column}_lag_3': current_values.shift(3)
            })

            # Combine with the original current column (target)
            df_lagged = pd.concat([current_values, lagged_data], axis=1).dropna()

            # Define the feature matrix (X) and target (y)
            X = df_lagged.drop(columns=[current_column])
            y = df_lagged[current_column]

            # Split the data into training and testing sets (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 3: Train the Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Step 4: Make predictions and calculate the Mean Squared Error (MSE)
            y_pred = rf_model.predict(X_test)
            mse_rf = mean_squared_error(y_test, y_pred)
            print(f'Mean Squared Error (Random Forest) for {current_column}: {mse_rf}')

print("Random Forest modeling completed for all materials and voltage conditions.")