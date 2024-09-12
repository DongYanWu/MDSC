import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
import pickle

# Define the path to the training data folder
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path

# Define the path where the model will be saved
model_save_path = '/Users/wudongyan/Desktop/lstm_saved_models/'  # Replace with your actual save path
os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Create an empty list to store all the dataframes
all_data = []

# Loop through each dielectric material (1 to 13)
for material in range(1, 14):
    material_path = os.path.join(base_path, str(material))
    
    # Loop through voltage conditions `b` and `c` (exclude `a` and `d`)
    for voltage in ['b', 'c']:
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

# Add a time indicator for each column (representing the phase in the measurement)
training_data['time_indicator'] = training_data.index % 4000 + 1  # Adds time step from 1 to 4000

X_train = []
y_train = []

# Extract features (first 50 rows) and targets (remaining 3950 rows)
for material in training_data['material'].unique():
    for voltage in training_data['voltage'].unique():
        # Filter data for this specific material and voltage condition
        data = training_data[(training_data['material'] == material) & 
                             (training_data['voltage'] == voltage)]
        
        # Step 1: Create lag features
        lagged_data = []
        for lag in range(1, 10):  # Create 10 lagged features
            lagged_data.append(data.shift(lag))
        
        # Combine original and lagged data
        lagged_features = pd.concat(lagged_data, axis=1).dropna()

        # Use the first 50 rows as features (input)
        X = lagged_features.iloc[:50, 1:11].values  # First 50 rows of lagged data

        # Use the remaining 3950 rows as the target (output to predict)
        y = data.iloc[50:, 1:11].values  # Next 3950 rows × 10 columns
        
        # Append to training lists
        X_train.append(X)
        y_train.append(y)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Step 1: Normalize the data
# Reshape to 2D (samples, features) for scaling
X_train_2d = X_train.reshape(X_train.shape[0], -1)
y_train_2d = y_train.reshape(y_train.shape[0], -1)

scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_scaled_2d = scaler_X.fit_transform(X_train_2d)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled_2d = scaler_y.fit_transform(y_train_2d)

# Step 2: Reshape scaled data back into 3D for LSTM
X_train_scaled = X_train_scaled_2d.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
y_train_scaled = y_train_scaled_2d.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2])

# Step 3: Split into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
)

# **Key Fix**: Use the original feature dimension from the input data for reshaping
n_features = X_train.shape[2]  # Number of features before scaling

# Step 4: Reshape the input for LSTM [samples, time steps, features]
X_train_lstm = X_train_split.reshape((X_train_split.shape[0], 50, n_features))  # 50 rows as input
X_val_lstm = X_val_split.reshape((X_val_split.shape[0], 50, n_features))  # Same reshape for validation set

# Reshape y_train to match LSTM output (samples, time steps, features)
y_train_lstm = y_train_split.reshape((y_train_split.shape[0], 3950, 10))  # 3950 rows, 10 features
y_val_lstm = y_val_split.reshape((y_val_split.shape[0], 3950, 10))

# Step 5: Build the Seq2Seq LSTM model
lstm_model = Sequential()

# Encoder: LSTM processes the input sequence
lstm_model.add(LSTM(50, activation='relu', input_shape=(50, n_features)))

# RepeatVector: Repeat the context vector to match the output sequence length (3950)
lstm_model.add(RepeatVector(3950))

# Decoder: LSTM processes the repeated context and returns a sequence
lstm_model.add(LSTM(50, activation='relu', return_sequences=True))

# TimeDistributed: Apply Dense layer to each time step to output 10 features per time step
lstm_model.add(TimeDistributed(Dense(10)))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse')

# Step 6: Train the model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, verbose=1)

# Step 7: Predict on the validation set
y_pred_lstm = lstm_model.predict(X_val_lstm)

# Step 8: Reshape the prediction and apply inverse transform
# Flatten y_pred_lstm to 2D to match the shape expected by inverse_transform
y_pred_lstm_reshaped = y_pred_lstm.reshape(y_pred_lstm.shape[0], -1)  # Flatten to 2D for scaling
y_pred_lstm_original = scaler_y.inverse_transform(y_pred_lstm_reshaped)  # Inverse scaling

# Step 9: Calculate MSE
y_val_split_reshaped = y_val_split.reshape(y_val_split.shape[0], -1)
mse_lstm = mean_squared_error(y_val_split_reshaped, y_pred_lstm_original)
print(f'Mean Squared Error (LSTM): {mse_lstm}')

# Step 10: Save the trained model for later use
model_save_filename = 'lstm_model_b_c.h5'
lstm_model.save(os.path.join(model_save_path, model_save_filename))
print(f"LSTM model saved as: {model_save_filename}")