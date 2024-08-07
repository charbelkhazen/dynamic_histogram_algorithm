'''def create_histogram(data, bins=None):
    total_value = len(data)  # Total number of data points

    if bins is None:
        histogram = {}
        for value in data:
            histogram[value] = histogram.get(value, 0) + 1
        # Convert counts to percentages
        for key in histogram:
            histogram[key] = (histogram[key] / total_value)
        return histogram

    min_value, max_value = min(data), max(data)
    bin_width = (max_value - min_value) / bins
    histogram = {}

    for i in range(bins):
        bin_min = min_value + i * bin_width
        bin_max = bin_min + bin_width
        count = sum(bin_min <= x < bin_max for x in data)
        # Include the max value in the last bin
        if i == bins - 1:
            count += sum(x == max_value for x in data)
        histogram[(bin_min, bin_max)] = (count / total_value) # Convert to percentage
    
    return histogram

# Example usage:
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
histogram = create_histogram(data , bins = 4)
print(histogram)'''


'''
CHI HEK 
def create_histogram(df, bins=None, t2):
    total_value = df.shape[0]  # Total number of data points   ?????????????????????????

    if bins is None:
        histogram = {}
        for _, row in df.iterrows():
            value  , vicinity = row[self.var_col_name],   row['vicinity']
            histogram[value] = histogram.get(value, 0) + vicinity
        # Convert counts to percentages
        for key in histogram:
            histogram[key] = histogram[key] / total_value
        return histogram

    min_value, max_value = df[self.var_col_name].min(), df[self.var_col_name].max()
    bin_width = (max_value - min_value) / bins
    histogram = {}

    for i in range(bins):
        bin_min = min_value + i * bin_width
        bin_max = bin_min + bin_width
        count = sum(row['vicinity'] for _, row in df.iterrows() if bin_min <= row[self.var_col_name] < bin_max)
        # Include the max value in the last bin
        if i == bins - 1:
            count += sum(row['vicinity']  for _, row in df.iterrows() if row[self.var_col_name] == max_value)
        histogram[(bin_min, bin_max)] = count / total_value  # Convert to percentage

    return histogram



'''

'''
def create_histogram(df, bins=None, t2 = 12 ,value_col_name = 'quantity', time_col_name = 'date'):
    total_value = df.shape[0]  # Total number of data points

    if bins is None:
        histogram = {}
        for _, row in df.iterrows():    #Note : _ is row's index 
            value, t1 = row[value_col_name], row[time_col_name]
            histogram[value] = histogram.get(value, 0) + already_defined_vicinity_function(t1, t2)
        # Convert counts to percentages
        for key in histogram:
            histogram[key] = histogram[key] / total_value
        return histogram

    min_value, max_value = df['value'].min(), df['value'].max()
    bin_width = (max_value - min_value) / bins
    histogram = {}

    for i in range(bins):
        bin_min = min_value + i * bin_width
        bin_max = bin_min + bin_width
        count = sum(already_defined_vicinity_function(row['t1'], t2) for _, row in df.iterrows() if bin_min <= row['value'] < bin_max)
        # Include the max value in the last bin
        if i == bins - 1:
            count += sum(already_defined_vicinity_function(row['t1'], t2) for _, row in df.iterrows() if row['value'] == max_value)
        histogram[(bin_min, bin_max)] = count / total_value  # Convert to percentage

    return histogram

''' '''
def pick_horizon(self, percentage): # How how many points to keep behind the horizon , if percentage = .6 -> you choose to forecast at time index for which 60% of values are behind 
    self.horizon = self.df[self.date_column_name].quantile(percentage)
    return self.horizon

def sample_from_histogram(self):
    
    keys = list(hist_dict.keys())
    weights = list(hist_dict.values())

    # Sample 10 times from the dictionary according to the weights
    samples = random.choices(keys, weights=weights, k=10)
    sample = sum(samples) / len(samples)
    return sample

def validation(params: dict):
    best_oos_performance = None
    lst_params = list(ParameterGrid(params))
    for param in lst_params:
        if best_mse_oos == None:
            mod = model().set_params(**param).fit(X_trn, y_trn)
            best_mod = mod
            y_pred = mod.predict(X_vld)
            best_ros = R_oos(y_vld, y_pred)
            best_mse_oos = mean_squared_error(y_vld,y_pred)
            best_param = param
            if illustration:
                print(f'Model with params: {param} finished.')
                print(f'with out-of-sample MSE on validation set: {best_mse_oos:.5f}')
                print(f'with out-of-sample R-squared on validation set: {best_ros*100:.7f}%')
                print('*'*60)
        else:
            time.sleep(sleep)
            mod = model().set_params(**param).fit(X_trn, y_trn)
            y_pred = mod.predict(X_vld)
            ros = R_oos(y_vld, y_pred)
            mse_oos = mean_squared_error(y_vld,y_pred)
            if illustration:
                print(f'Model with params: {param} finished.')
                print(f'with out-of-sample MSE on validation set: {mse_oos:.5f}')
                print(f'with out-of-sample R-squared on validation set: {ros*100:.7f}%')
                print('*'*60)
            if mse_oos < best_mse_oos:
                best_mse_oos = mse_oos
                best_mod = mod
                best_param = param
    if illustration:
        print('\n'+'#'*60)
        print('Tuning process finished!!!')
        print(f'The best setting is: {best_param}')
        print(f'with MSE OOS {best_mse_oos:.5f} on validation set.')
        print('#'*60)
    return best_mod

'''

'''
import pandas as pd
import numpy as np

# Create date range
date_range = pd.date_range(start='2001-01-01', end='2001-01-31')

# Generate random quantities
np.random.seed(0)
quantities = np.random.randint(1, 11, size=len(date_range))

# Create DataFrame
df = pd.DataFrame({
    'date_column': date_range,
    'quantity': quantities
})

percentile_date = df['date_column'].quantile(.98)
print(percentile_date)
'''

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


# Set the seed for reproducibility
np.random.seed(42)

# Generate date range for 5 years
date_range = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')

# Create a DataFrame
df = pd.DataFrame(date_range, columns=['Date'])

# Base quantity: upward trend + random noise
df['Quantity'] = 100 + (df.index / 365) * 10 + np.random.normal(0, 5, len(df))

# Monthly patterns (e.g., higher in December)
monthly_pattern = df['Date'].dt.month.map({
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 20
})
df['Quantity'] += monthly_pattern

# Days of the month patterns (e.g., higher on 1st and 15th)
day_of_month_pattern = df['Date'].dt.day.map(lambda x: 5 if x in [1, 15] else 0)
df['Quantity'] += day_of_month_pattern

# Days of the week patterns (e.g., lower on weekends)
day_of_week_pattern = df['Date'].dt.dayofweek.map({0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: -10, 6: -10})
df['Quantity'] += day_of_week_pattern

# Ensure no negative quantities
df['Quantity'] = df['Quantity'].clip(lower=0)

print(df.head())



# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Quantity'], label='Quantity')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.show()
