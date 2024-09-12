import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Load the two CSV files
file1_path = '/Users/wudongyan/Downloads/pre-train/13/c.csv'
file2_path = '/Users/wudongyan/Desktop/c_testing_rf_finsihed.csv'

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)



# Extract the target columns (y01 to y10) for comparison
cols = ['y01', 'y02', 'y03', 'y04', 'y05', 'y06', 'y07', 'y08', 'y09', 'y10']

# Calculate the MSE for each column (raw values)
mse_raw = {col: mean_squared_error(df1[col], df2[col]) for col in cols}

# Apply MinMax scaling to both datasets
scaler = MinMaxScaler()

df1_scaled = pd.DataFrame(scaler.fit_transform(df1[cols]), columns=cols)
df2_scaled = pd.DataFrame(scaler.transform(df2[cols]), columns=cols)

# Calculate the MSE for each column (min-max scaled values)
mse_scaled = {col: mean_squared_error(df1_scaled[col], df2_scaled[col]) for col in cols}

# Combine results into a DataFrame for easy comparison
mse_results = pd.DataFrame({
    'MSE (Raw Values)': mse_raw,
    'MSE (MinMax Scaled)': mse_scaled
})

combined_mse_raw = sum(mse_raw.values()) 
combined_mse_scaled = sum(mse_scaled.values()) 

# Create a new DataFrame to present the combined results
combined_mse_results = pd.DataFrame({
    'MSE (Raw Values)': [combined_mse_raw],
    'MSE (MinMax Scaled)': [combined_mse_scaled]
}, index=['Combined MSE'])
print(combined_mse_results)


