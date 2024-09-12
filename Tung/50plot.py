import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the training data folder and the predicted answer file
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path
answer_path = '/Users/wudongyan/Desktop/answer.csv'  # Path to the predicted answer.csv

# Load the predicted answer data
predicted_data = pd.read_csv(answer_path)

# Voltage condition a and current column y01
voltage = 'a'
current_column = 'y01'

# Create a plot for voltage condition a and current column y01 (first 50 rows only)
plt.figure(figsize=(10, 6))
plt.title(f"Voltage Condition: {voltage}, Current Column: {current_column} (First 50 Rows)")
plt.xlabel('ID (Time)')
plt.ylabel('Current')

# Loop through each dielectric material (1 to 13)
for material in range(1, 14):
    material_path = os.path.join(base_path, str(material))
    file_path = os.path.join(material_path, f'{voltage}.csv')
    
    if os.path.exists(file_path):
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Plot the first 50 rows of the current column (y01) against the id (which represents time)
        plt.plot(data['id'][:50], data[current_column][:50], label=f'Material {material} - Training', alpha=0.7)

# Plot the first 50 rows of the predicted test data (to compare)
plt.plot(predicted_data.index[:50], predicted_data[current_column][:50], label='Predicted - Test Data', linestyle='--', color='red', alpha=0.7)

# Add a legend and show the plot
plt.legend()
plt.tight_layout()
plt.show()