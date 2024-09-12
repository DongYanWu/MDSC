import os
import pandas as pd
import numpy as np
import pickle

# Path to the test data
test_data_path = '/Users/wudongyan/Desktop/c_testing.csv'
test_data = pd.read_csv(test_data_path)

# Path to the saved models
model_base_path = '/Users/wudongyan/Desktop/rf_cross_saved_models_wo13/'  # Folder where models are saved

# List of current columns (y01 to y10)
current_columns = [f'y{i:02d}' for i in range(1, 11)]

# Step 1: Load the test data (first 50 rows)
X_test = test_data.iloc[:50, 1:11].values  # First 50 rows, columns y01 to y10

# Step 2: Prepare DataFrame for final predictions (for remaining 3950 rows)
predictions_df = pd.DataFrame(columns=current_columns)

# Loop through each material and voltage condition to predict for each column
for material in range(1, 13):  # For materials 1 to 12
    for voltage in ['c']:  # Excluding voltage conditions `a`, `b` and `d`
        
        print(f"Predicting for Material {material}, Voltage {voltage}")
        
        for current_column in current_columns:
            print(f"Predicting for column: {current_column}")
            
            # Step 3: Load the trained Random Forest model for this column
            model_filename = f'random_forest_model_material_{material}_voltage_{voltage}_{current_column}.pkl'
            model_path = os.path.join(model_base_path, model_filename)
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as model_file:
                    rf_model = pickle.load(model_file)
                
                # Step 4: Create lag features for the test data
                test_series = test_data[current_column].iloc[:50].dropna().values
                
                # Start with the last row (which will be used to predict the next value)
                X_pred = np.array([
                    test_series[-1],  # lag_1
                    test_series[-2],  # lag_2
                    test_series[-3]   # lag_3
                ]).reshape(1, -1)
                
                future_preds = []
                
                for _ in range(3950):
                    # Predict the next step
                    next_pred = rf_model.predict(X_pred)[0]
                    future_preds.append(next_pred)
                    
                    # Shift the input for the next prediction (rolling the window)
                    X_pred = np.roll(X_pred, -1)  # Shift values left
                    X_pred[0, -1] = next_pred  # Append the new predicted value at the end

                # Step 5: Store predictions in the DataFrame
                predictions_df[current_column] = future_preds
            else:
                print(f"Model not found: {model_path}")

# Step 6: Save the final predictions to 'answer.csv'
predictions_df.to_csv('answerrf_wo13.csv', index=False)
print("Predictions saved to 'answerrf_wo13.csv'.")