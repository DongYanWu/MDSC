import os
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

# Path to the saved LSTM model
model_path = '/Users/wudongyan/Desktop/lstm_saved_models/lstm_model_b_c.h5'

# Path to the testing data
test_data_path = '/Users/wudongyan/Desktop/testing.csv'
test_data = pd.read_csv(test_data_path)

# **Fix**: Define the custom loss function (MSE) during model loading
mse = MeanSquaredError()

# Load the saved LSTM model with the custom loss function
lstm_model = load_model(model_path, custom_objects={'mse': mse})

# Preprocessing the testing data: first 50 rows (which are provided in testing.csv)
X_test = test_data.iloc[:50, 1:11].values  # First 50 rows, columns y01 to y10

# Initialize MinMaxScaler for the test data (based on the same scaling as training)
scaler_X_test = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler on the first 50 rows of test data and transform it
X_test_scaled = scaler_X_test.fit_transform(X_test)

# Reshape the test data for LSTM input (samples, time steps, features)
X_test_lstm = X_test_scaled.reshape((1, X_test_scaled.shape[0], X_test_scaled.shape[1]))  # Shape [1, 50, 10]

# Step 1: Make predictions for the next 3950 rows using the trained LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Step 2: Reshape the predicted data back to 2D to apply inverse transform
y_pred_lstm_reshaped = y_pred_lstm.reshape(-1, 10)  # Flatten for inverse scaling

# Step 3: Inverse transform the predicted values to original scale
y_pred_lstm_original = scaler_X_test.inverse_transform(y_pred_lstm_reshaped)

# Step 4: Create a DataFrame to store the predicted values for the next 3950 rows
columns = [f'y{i:02d}' for i in range(1, 11)]
predictions_df = pd.DataFrame(y_pred_lstm_original, columns=columns)

# Step 5: Save the final predictions to 'answer_lstm.csv'
predictions_df.to_csv('answer_lstm.csv', index=False)
print("Predictions saved to 'answer_lstm.csv'.")