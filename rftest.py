import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pickle  # To save the models

# Define the path to the training data folder
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path

# Define the path where the models will be saved
model_save_path = '/Users/wudongyan/Desktop/rf_cross_saved_models_wo13/'  # Replace with your actual save path
os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Number of splits for cross-validation
n_splits = 5

# Create a TimeSeriesSplit object for cross-validation
tscv = TimeSeriesSplit(n_splits=n_splits)

# Loop through each dielectric material (1 to 12)
for material in range(1, 13):
    material_path = os.path.join(base_path, str(material))
    
    # Only loop through voltage conditions `b` and `c` (exclude `a` and `d`)
    for voltage in ['b', 'c']:
        file_path = os.path.join(material_path, f'{voltage}.csv')
        print(f"Processing file: {file_path}")
        
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Perform cross-validation on each current column (y01 to y10)
        for current_column in [f'y{i:02d}' for i in range(1, 11)]:
            print(f"Performing Cross-Validation for: Material {material}, Voltage {voltage}, Column {current_column}")
            
            # Extract the current values from the column
            current_values = data[current_column].dropna()

            # Create lag features (use previous 3 time steps as features)
            lagged_data = pd.DataFrame({
                f'{current_column}_lag_1': current_values.shift(1),
                f'{current_column}_lag_2': current_values.shift(2),
                f'{current_column}_lag_3': current_values.shift(3)
            })

            # Combine with the original current column (target)
            df_lagged = pd.concat([current_values, lagged_data], axis=1).dropna()

            # Define feature matrix (X) and target (y)
            X = df_lagged.drop(columns=[current_column])
            y = df_lagged[current_column]

            # Initialize lists to store results
            mse_scores = []

            # Perform cross-validation
            fold = 1
            for train_index, test_index in tscv.split(X):
                # Split data into train and validation sets for this fold
                X_train, X_val = X.iloc[train_index], X.iloc[test_index]
                y_train, y_val = y.iloc[train_index], y.iloc[test_index]

                # Train the Random Forest model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)

                # Make predictions and calculate MSE for this fold
                y_pred = rf_model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                mse_scores.append(mse)
                print(f"Fold {fold} - MSE: {mse}")
                fold += 1

            # Calculate average MSE across all folds
            avg_mse = np.mean(mse_scores)
            print(f'Average MSE (Random Forest) for {current_column}: {avg_mse}')
            
            # Save the trained model to disk
            model_filename = f'random_forest_model_material_{material}_voltage_{voltage}_{current_column}.pkl'
            model_full_path = os.path.join(model_save_path, model_filename)
            with open(model_full_path, 'wb') as model_file:
                pickle.dump(rf_model, model_file)
            print(f"Model saved as: {model_full_path}")