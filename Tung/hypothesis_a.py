import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp

# Define the path to the training data folder and the predicted answer file
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path
answer_path = '/Users/wudongyan/Desktop/answer.csv'  # Path to the predicted answer.csv

# Load the predicted answer data (first 50 rows)
predicted_data = pd.read_csv(answer_path)

# Voltage condition 'a'
voltage = 'a'

# Current columns (y01 to y10)
current_columns = [f'y{i:02d}' for i in range(1, 11)]

# Directory to store hypothesis test results
results = {}

# Loop through each current column (y01 to y10) and perform hypothesis tests
for current_column in current_columns:
    # Collect training data for the first 50 rows under condition 'a'
    training_values = []
    
    # Loop through each dielectric material (1 to 13)
    for material in range(1, 14):
        material_path = os.path.join(base_path, str(material))
        file_path = os.path.join(material_path, f'{voltage}.csv')
        
        if os.path.exists(file_path):
            # Load the CSV file
            data = pd.read_csv(file_path)
            
            # Append the first 50 rows of the current column to the list
            training_values.extend(data[current_column][:50].values)

    # Convert training data to a numpy array for comparison
    training_values = np.array(training_values)
    
    # Get the first 50 rows of the predicted test data
    test_values = predicted_data[current_column][:50].values

    # Perform the two-sample t-test
    t_stat, t_p_value = ttest_ind(training_values, test_values, equal_var=False)
    
    # Perform the Kolmogorov-Smirnov test
    ks_stat, ks_p_value = ks_2samp(training_values, test_values)

    # Store results
    results[current_column] = {
        't-test': {
            't-statistic': t_stat,
            'p-value': t_p_value
        },
        'K-S test': {
            'K-S statistic': ks_stat,
            'p-value': ks_p_value
        }
    }

    # Output the results
    print(f"Hypothesis Test Results for {current_column}:")
    print(f"  t-test p-value: {t_p_value}")
    print(f"  K-S test p-value: {ks_p_value}")
    print("")

# Print the hypothesis test results for all columns
for current_column, test_results in results.items():
    print(f"{current_column}:")
    print(f"  t-test: t-statistic = {test_results['t-test']['t-statistic']}, p-value = {test_results['t-test']['p-value']}")
    print(f"  K-S test: K-S statistic = {test_results['K-S test']['K-S statistic']}, p-value = {test_results['K-S test']['p-value']}")
    print("")

# For every current column (y01 to y10), both the t-test and the K-S test indicate that the testing data does not come from the same distribution as the training data under voltage condition a.
# This suggests that the testing data is highly unlikely to be from voltage condition a.