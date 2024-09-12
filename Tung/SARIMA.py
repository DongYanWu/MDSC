import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import pickle

# Define the path to the training data folder
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path

# Loop through each dielectric material (1 to 13)
for material in range(1, 14):
    material_path = os.path.join(base_path, str(material))
    
    # Loop through each voltage condition (a, b, c, d)
    for voltage in ['a', 'b', 'c', 'd']:
        file_path = os.path.join(material_path, f'{voltage}.csv')
        print(f"Processing file: {file_path}")
        
        # Step 1: Load the CSV file
        data = pd.read_csv(file_path)

        # Step 2: Iterate through each current measurement column (y01 to y10)
        for current_column in [f'y{i:02d}' for i in range(1, 11)]:
            print(f"Training SARIMA for: Material {material}, Voltage {voltage}, Column {current_column}")
            
            # Extract the current values from the column (drop any NaNs if present)
            current_values = data[current_column].dropna().values

            # Split the data into training and validation sets (80% train, 20% validation)
            train_size = int(len(current_values) * 0.8)
            train, val = current_values[:train_size], current_values[train_size:]

            # Step 3: Train the SARIMA model using auto_arima
            auto_arima_model = auto_arima(train, 
                                          seasonal=True, 
                                          m=12,  # Adjust based on the data (seasonality if any)
                                          stepwise=True, 
                                          suppress_warnings=True,
                                          max_p=3, max_q=3, max_P=2, max_Q=2, max_order=10)

            # Step 4: Forecast on the validation set
            val_pred = auto_arima_model.predict(n_periods=len(val))

            # Step 5: Calculate MSE on the validation set
            mse_sarima = mean_squared_error(val, val_pred)
            print(f'Mean Squared Error (SARIMA) for {current_column}: {mse_sarima}')

            # Step 6: Save the trained model for future use
            model_save_path = f"sarima_model_material_{material}_voltage_{voltage}_{current_column}.pkl"
            with open(model_save_path, 'wb') as model_file:
                pickle.dump(auto_arima_model, model_file)
            print(f"Model saved: {model_save_path}")

print("Training completed for all materials and voltage conditions.")