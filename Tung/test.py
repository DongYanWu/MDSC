import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the training data folder and the predicted answer file
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path
answer_path = '/Users/wudongyan/Desktop/answer.csv'  # Path to the predicted answer.csv

# Load the predicted answer data
predicted_data = pd.read_csv(answer_path)

# Voltage condition 'a'
voltage = 'd'

# Current columns (y01 to y10)
current_columns = [f'y{i:02d}' for i in range(1, 11)]

# Directory to save the plot images
output_dir = '/Users/wudongyan/Desktop/plots/actual/d'  # Replace with your actual path
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Create HTML content
html_content = """
<html>
<head>
    <title>Voltage Condition a: Plots for y01 to y10</title>
</head>
<body>
    <h1>Voltage Condition a: Plots for y01 to y10</h1>
"""

# Loop through each current column (y01 to y10) and create plots
for current_column in current_columns:
    # Create a plot for the current column
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
            
            # Plot the first 50 rows of the current column (y01 to y10) against the id (time)
            plt.plot(data['id'][:50], data[current_column][:50], label=f'Material {material} - Training', alpha=0.7)

    # Plot the first 50 rows of the predicted test data (to compare)
    plt.plot(predicted_data.index[:50], predicted_data[current_column][:50], label='Predicted - Test Data', linestyle='--', color='red', alpha=0.7)

    # Add a legend
    plt.legend()

    # Save the plot as a PNG file
    plot_filename = f'plot_{current_column}.png'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    # Add the image to the HTML content
    html_content += f'<h2>Plot for {current_column}</h2>\n'
    html_content += f'<img src="{plot_filename}" width="800" />\n'

# Finalize the HTML content
html_content += """
</body>
</html>
"""

# Save the HTML file
html_file_path = os.path.join(output_dir, 'plotsd.html')
with open(html_file_path, 'w') as html_file:
    html_file.write(html_content)

print(f"HTML file created: {html_file_path}")