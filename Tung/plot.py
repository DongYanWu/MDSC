# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Define the path to the training data folder and the predicted answer file
# base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path
# answer_path = '/Users/wudongyan/Desktop/answer.csv'  # Path to the predicted answer.csv

# # Load the predicted answer data
# predicted_data = pd.read_csv(answer_path)

# # Voltage conditions and current columns
# voltage_conditions = ['a', 'b', 'c', 'd']
# current_columns = [f'y{i:02d}' for i in range(1, 11)]

# # Create separate plots for each current column
# for current_column in current_columns:
#     print(f"Plotting for {current_column}...")
    
#     # Create a 4-figure plot: one figure per voltage condition (a, b, c, d)
#     fig, axs = plt.subplots(4, 4, figsize=(16, 16))
#     fig.suptitle(f'Current: {current_column}', fontsize=16)
    
#     # Loop through each voltage condition (a, b, c, d)
#     for voltage in voltage_conditions:
        
#         # Loop through each dielectric material (1 to 13)
#         for material in range(1, 14):
#             material_path = os.path.join(base_path, str(material))
#             file_path = os.path.join(material_path, f'{voltage}.csv')

#             if os.path.exists(file_path):
#                 # Load the CSV file
#                 data = pd.read_csv(file_path)
                
#                 # Plot the current column against the id (which represents time)
#                 ax = axs[(material - 1) // 4, (material - 1) % 4]
#                 ax.plot(data['id'], data[current_column], label=f'Voltage {voltage} - Training', alpha=0.7)
#                 ax.set_title(f'Material {material}')
#                 ax.set_xlabel('ID (Time)')
#                 ax.set_ylabel('Current')

#         # Add the predicted test data to each plot
#         for material in range(1, 14):
#             ax = axs[(material - 1) // 4, (material - 1) % 4]
#             # Plot predicted values on the same plot
#             ax.plot(predicted_data.index + 50, predicted_data[current_column], label=f'Voltage {voltage} - Predicted', linestyle='--', alpha=0.7)

#     # Add legends and make the layout tight
#     for ax in axs.flat:
#         ax.legend()
    
#     # Adjust layout and show the plot for this current column
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()
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

# Create a plot for voltage condition a and current column y01
plt.figure(figsize=(10, 6))
plt.title(f"Voltage Condition: {voltage}, Current Column: {current_column}")
plt.xlabel('ID (Time)')
plt.ylabel('Current')

# Loop through each dielectric material (1 to 13)
for material in range(1, 14):
    material_path = os.path.join(base_path, str(material))
    file_path = os.path.join(material_path, f'{voltage}.csv')
    
    if os.path.exists(file_path):
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Plot the current column (y01) against the id (which represents time)
        plt.plot(data['id'], data[current_column], label=f'Material {material} - Training', alpha=0.7)

# Plot the predicted test data as well (we don't know the actual condition, so we plot it here)
plt.plot(predicted_data.index + 50, predicted_data[current_column], label='Predicted - Test Data', linestyle='--', color='red', alpha=0.7)

# Add a legend and show the plot
plt.legend()
plt.tight_layout()
plt.show()