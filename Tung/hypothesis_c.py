import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ks_2samp

# Define the path to the training data folder and the predicted answer file
base_path = '/Users/wudongyan/Downloads/pre-train/'  # Replace with actual folder path
answer_path = '/Users/wudongyan/Desktop/answer.csv'  # Path to the predicted answer.csv

# Load the predicted answer data (first 50 rows)
predicted_data = pd.read_csv(answer_path)

# Voltage condition 'c'
voltage = 'c'

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


# 	•	For most of the current columns (y01, y02, y03, y06, y07, y08, y09), both the t-test and K-S test show very small p-values, indicating that the testing data does not come from condition c.
# 	•	y04, y05, and y10:
# 	•	t-test:
# 	•	For y04 and y10, the p-values are large enough (0.63 for y04 and 0.79 for y10), suggesting that the means of the testing data and training data are not significantly different.
# 	•	For y05, the p-value is 0.01, which is slightly below 0.05, meaning there is a mild indication of mean differences.
# 	•	K-S test:
# 	•	Despite the t-test results, the K-S test for all these columns shows very small p-values, indicating that the distributions of the testing data and training data are significantly different.

# Conclusion:

# 	•	Overall, the hypothesis test results suggest that the testing data does not come from condition c for the majority of columns.
# 	•	Even though the t-test for y04, y05, and y10 fails to reject the null hypothesis (indicating similar means), the K-S test indicates that the distributions are different for all columns.