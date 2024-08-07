import pandas as pd
import numpy as np
from datetime import datetime
import calendar 
import math 
import random 
from sklearn.model_selection import ParameterGrid

'''
# Define date range
#date_range = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
date_range = pd.date_range(start='2020-01-01', end='2020-01-30', freq='D')

# Generate random quantity values from a normal distribution
#quantities = np.random.normal(loc=150, scale=20, size=len(date_range))
quantities = np.random.randint(1, 11, size=len(date_range))
# Create the DataFrame
df = pd.DataFrame({'date': date_range, 'quantity': quantities})

'''
#CREATE DATASET WITH TIME DEPENDENCIES  - SEE BELOW FMI 
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

'''
MORE ON DATA:
Date Range: Generates a date range for 5 years.
Base Quantity: General upward trend with noise.
Monthly Patterns: Higher values in specific months. (December. Other months show a gradual increase from January to November. )
Days of the Month Patterns: Increase on the 1st and 15th.
Days of the Week Patterns: Higher on weekdays, lower on weekends.
'''


'''
MAIN CODE 
'''






# Read vicinity function corresponding to each of them in order to undestand  - DOES NOT MAKE SENSE TO READ THEM FIRST 
equally_spaced_minute_of_hour = np.linspace(0.1, 0.9, 31)  
equally_spaced_hour_of_day = np.linspace(0.1, 0.9, 13)     
equally_spaced_day_of_week = np.linspace(0.1, 0.9, 4)      
equally_spaced_week_of_month = np.linspace(0.1, 0.9, 3)    
equally_spaced_month_of_year = np.linspace(0.1, 0.9, 7)    

def minute_of_hour_vicinity(x, y):
    if not (0 <= x < 60 and 0 <= y < 60):
        raise ValueError("Minute variables should be between 0 and 59.")
    delta = abs(x - y)
    min_delta = min(delta, 60 - delta)
    return equally_spaced_minute_of_hour[30 - min_delta]

def hour_of_day_vicinity(x, y):
    if not (0 <= x < 24 and 0 <= y < 24):
        raise ValueError("Hour variables should be between 0 and 23.")
    delta = abs(x - y)
    min_delta = min(delta, 24 - delta)
    return equally_spaced_hour_of_day[12 - min_delta]
def day_of_week_vicinity(x, y):
    if not (1 <= x <= 7 and 1 <= y <=7):
        raise ValueError("Week variables should be between 1 and 5.")
    delta = abs(x - y)
    min_delta = min(delta, 7 - delta)
    return equally_spaced_day_of_week[3 - min_delta]
def week_of_month_vicinity(x, y):
    if not (1 <= x <= 5 and 1 <= y <= 5):
        raise ValueError("Week variables should be between 1 and 5.")
    delta = abs(x - y)
    min_delta = min(delta, 5 - delta)
    return equally_spaced_week_of_month[2 - min_delta]

def month_of_year_vicinity(x, y):
    
    if not (1 <= x <= 12 and 1 <= y <= 12):
        raise ValueError("Month variables should be between 1 and 12.")
    delta = abs(x - y)
    min_delta = min(delta, 12 - delta)
    return equally_spaced_month_of_year[6 - min_delta]

# There are still two other functions for computing vicinity 
# They are inside the class as they depend on the attribytes of the df - notably date 
# see days_between for vicinity measure that is non cyclical 
def number_of_equally_spaced_values_day_of_month_vic(nbr_days_in_month):
    if nbr_days_in_month % 2 == 0:
        return nbr_days_in_month // 2 + 1
    else:
        return math.ceil(nbr_days_in_month / 2)
    
class HistAlg:
    def __init__(self, data, var_column_name='quantity', date_granularity ='days', date_column_name='date', 
                 weight_month_of_year_vic=0.1, weight_day_of_week_vic=0, weight_week_of_month_vic=0, 
                 weight_hour_of_day_vic=0, weight_minute_of_hour=0,weight_day_of_month_vic=0,  bins=None , turn_off_vicinity = False):
        '''
        Here are all possible granularity values:
        .days
        .seconds: Total seconds
        .microseconds: Total microseconds
        .total_seconds(): Total duration in seconds as a float
        .weeks: Weeks (not a direct attribute, but can be calculated)
        .hours: Hours (calculated from seconds)
        .minutes: Minutes (calculated from seconds)
        '''
        self.data = data
        
        self.var_column_name = var_column_name
        self.date_column_name = date_column_name
        self.df = self.data[[self.var_column_name, self.date_column_name]]

        self.df[self.date_column_name] = pd.to_datetime(self.df[self.date_column_name])
        self.original_df = self.df.copy()
        # Hyperparameters
        self.weight_month_of_year_vic = weight_month_of_year_vic
        self.weight_day_of_week_vic = weight_day_of_week_vic
        self.weight_week_of_month_vic = weight_week_of_month_vic
        self.weight_hour_of_day_vic = weight_hour_of_day_vic
        self.weight_minute_of_hour = weight_minute_of_hour
        self.weight_day_of_month_vic = weight_day_of_month_vic 

        self.bins = bins 

        self.turn_off_vicinity = turn_off_vicinity
        self.data_max_date = data[self.date_column_name].max()
        self.data_min_date = data[self.date_column_name].min()
        self.date_granularity = date_granularity

        self.granularity_mapping = {
            'days': lambda delta: delta.days,
            'seconds': lambda delta: delta.total_seconds(),
            'microseconds': lambda delta: delta.days * 86400 * 1e6 + delta.seconds * 1e6 + delta.microseconds,
            'total_seconds': lambda delta: delta.total_seconds(),
            'weeks': lambda delta: delta.days / 7,
            'hours': lambda delta: delta.total_seconds() / 3600,
            'minutes': lambda delta: delta.total_seconds() / 60,
            'months': lambda delta: (delta.days // 30) + (delta.days % 30 > 0),
            'years': lambda delta: (delta.days // 365) + (delta.days % 365 > 0)
        }

    def set_params(self, **params):  
        for param in params.keys():
            setattr(self, param, params[param])
        return self
    

    def list_horizons(self, prct = 0.6):  #prct : percentage of values before the horizon 
        n = len(self.df)
        start_index = int(n * (prct))
        list_horizons = df[self.date_column_name][start_index:].tolist()
        return list_horizons


    def non_stationary_vicinity(self, x, y):  # write as 'yyyy-mm-dd'
        delta = self.granularity_mapping[self.date_granularity](abs(x - y))
        vicinity = self.equally_spaced[self.nbr_time_indeces - delta]  #121212121
        return vicinity
    


    def day_of_month_vicinity(self,x,y,nbr_of_days_in_month):
        if not (1 <= x <= 31 and 1 <= y <=31) :
            raise ValueError("Day of month variables should be between 1 and 31.")
        nbr_equally_spaced_values = number_of_equally_spaced_values_day_of_month_vic(nbr_of_days_in_month)
        equally_spaced_day_of_month = np.linspace(0.1, 0.9, nbr_equally_spaced_values)  
        delta = abs(x - y)
        min_delta = min(delta, nbr_of_days_in_month - delta)
        return  equally_spaced_day_of_month[nbr_of_days_in_month - min_delta]
 


    def find_vicinity(self, horizon='yyyy-dd-mm'): 
        self.horizon = horizon
        non_stationary_horizon = pd.to_datetime(self.horizon)
        if non_stationary_horizon < self.data_max_date :
            self.df = self.df[self.df[self.date_column_name] <= non_stationary_horizon]
        vicinity = 0 # this is the final
        if self.turn_off_vicinity == False: 
                    
            if self.weight_month_of_year_vic != 0:
                self.df['month'] = self.df[self.date_column_name].dt.month
                month_horizon =  pd.to_datetime(self.horizon).month
                self.df['month_vicinity'] = self.df['month'].apply(lambda x: month_of_year_vicinity(x, y=month_horizon))
                vicinity +=self.weight_month_of_year_vic * self.df['month_vicinity']  

            if self.weight_week_of_month_vic != 0:
                self.df['week'] = self.df[self.date_column_name].apply(lambda x: (x.day - 1) // 7 + 1)
                week_horizon =  pd.to_datetime(self.horizon).isocalendar().week
                self.df['week_vicinity'] = self.df['week'].apply(lambda x: week_of_month_vicinity(x, y=week_horizon))
                vicinity += self.weight_week_of_month_vic * self.df['week_vicinity'] 

                
            if self.weight_day_of_month_vic != 0:    
                self.df['day_of_month'] = self.df[self.date_column_name].dt.day
                self.df['month'] =  self.df[self.date_column_name].dt.month 
                self.df['year'] = self.df[self.date_column_name].dt.year 
                day_of_month_horizon =  pd.to_datetime(self.horizon).day + 1   
                df['day_of_month_vicinity'] = df.apply(lambda row: self.day_of_month_vicinity(row['day'], y=day_of_month_horizon, z=calendar.monthrange(row['year'], row['month'])[1]), axis=1)
                vicinity +=       self.weight_day_of_month_vic * self.df['day_of_month_vicinity'] 


            if self.weight_day_of_week_vic != 0:
                self.df['day_of_week'] = self.df[self.date_column_name].dt.dayofweek
                day_of_week_horizon =  pd.to_datetime(self.horizon).dayofweek + 1 
                self.df['day_of_week_vicinity'] = self.df['day_of_week'].apply(lambda x: day_of_week_vicinity(x, y=day_of_week_horizon))
                vicinity += self.weight_day_of_week_vic * self.df['day_of_week_vicinity']


            if self.weight_hour_of_day_vic != 0:
                self.df['hour_of_day'] = self.df[self.date_column_name].dt.hour
                hour_of_day_horizon = pd.to_datetime(self.horizon).hour
                self.df['hour_of_day_vicinity'] = self.df['hour_of_day'].apply(lambda x: hour_of_day_vicinity(x, y=hour_of_day_horizon))
                vicinity += self.weight_hour_of_day_vic   * self.df['hour_of_day_vicinity'] 


            if self.weight_minute_of_hour != 0:
                self.df['minute'] = self.df[self.date_column_name].dt.minute
                minute_horizon =  pd.to_datetime(self.horizon).minute
                self.df['minute_of_hour_vicinity'] = self.df['minute'].apply(lambda x: minute_of_hour_vicinity(x, y=minute_horizon))
                vicinity += self.weight_minute_of_hour  * self.df['minute_of_hour_vicinity']

            self.nbr_time_indeces = self.granularity_mapping[self.date_granularity](abs( non_stationary_horizon - self.data_min_date ))
            self.equally_spaced = np.linspace(0.1, 0.9, self.nbr_time_indeces + 1)
            self.df['non_stationary_vicinity'] = self.df[self.date_column_name].apply(lambda x: self.non_stationary_vicinity( x, y=non_stationary_horizon))   
            vicinity += (1-self.weight_month_of_year_vic - self.weight_week_of_month_vic - self.weight_day_of_week_vic - self.weight_day_of_month_vic - self.weight_hour_of_day_vic - self.weight_minute_of_hour) *self.df['non_stationary_vicinity']
        else:


            print('ZABBET')                                  
      


        self.df['vicinity'] = vicinity
        self.df =self.df[[self.date_column_name,self.var_column_name ,'vicinity']]

        return self 
    def get_df(self):
        return self.df 

        

    def create_histogram(self, bins=None):
            
        total_value = self.df['vicinity'].sum()   

        if bins is None:
            self.histogram = {}
            for _, row in self.df.iterrows():
                value, vicinity = row[self.var_column_name], row['vicinity']
                self.histogram[value] = self.histogram.get(value, 0) + vicinity
            # Convert counts to percentages
            for key in self.histogram:
                self.histogram[key] = self.histogram[key] / total_value
            return self


        min_value, max_value = self.df[self.var_column_name].min(), self.df[self.var_column_name].max()
        bin_width = (max_value - min_value) / bins
        self.histogram = {}


        for i in range(bins):
            bin_min = min_value + i * bin_width
            bin_max = bin_min + bin_width
            count = sum(row['vicinity'] for _, row in self.df.iterrows() if bin_min <= row[self.var_column_name] < bin_max)
            # Include the max value in the last bin
            if i == bins - 1:
                count += sum(row['vicinity'] for _, row in self.df.iterrows() if row[self.var_column_name] == max_value)
            self.histogram[(bin_min, bin_max)] = count / total_value  # Convert to percentage

        return self
    
    def get_histogram(self):
        return self.histogram


    def sample_from_histogram(self, number_of_samples = 100):
        
        keys = list(self.histogram.keys())
        weights = list(self.histogram.values())

        # Sample 10 times from the dictionary according to the weights
        samples = random.choices(keys, weights=weights, k=number_of_samples)
        self.sample = sum(samples) / len(samples)
        return self
    
    def get_sample(self):
        return self.sample 
    def compute_absolute_deviation(self):
        print("\n!!!!self horizon is  :", self.horizon)
        actual = self.df.loc[self.df[self.date_column_name] == self.horizon, self.var_column_name ].values[0]
        #print('\n !!!!!!!!! self df is ',self.df)        
        print('\n !!!!!!!!!!!!!!!!!!!!!!!actual is ', actual)
        forecast = self.sample 
        print('!!!!!!!!!!!!!!forecast is: ',forecast)
        abs_dev = abs(actual - forecast) 
        return abs_dev
    
    def reset_to_original_df(self):
        self.df = self.original_df.copy()
        return self


print(df)

params = {
    'weight_month_of_year_vic': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'weight_day_of_week_vic': [0],
    'weight_week_of_month_vic': [0],
    'weight_hour_of_day_vic': [0],
    'weight_minute_of_hour': [0],
    'weight_day_of_month_vic': [0]
}



lst_params = list(ParameterGrid(params))
histogram = HistAlg(data=df, var_column_name='quantity', date_granularity='days', date_column_name='date')   # this is the complete df of a single time series 

lst_horizons = histogram.list_horizons(prct = 0.9)

best_model = None 
count_param , count_horizon = 1 , 1 
for param in lst_params:
    abs_dev_for_param = 0 
    print('\n\n\n\n\n\n\nPARAMETER USED: ' , param)
    for horizon in lst_horizons:
      print('\nHORIZON NUMBER ', count_horizon)
      count_horizon = count_horizon % 3 + 1
      abs_dev = histogram.set_params(**param).find_vicinity(horizon=horizon).create_histogram().sample_from_histogram().compute_absolute_deviation()
      print("absolute deviation is:" , abs_dev)
      abs_dev_for_param += abs_dev 
      histogram.reset_to_original_df()
    print('and absolute total deviation for model is ', abs_dev_for_param)
    if best_model == None : 
        best_model = param
        best_abs_dev_for_param = abs_dev_for_param
    else:
        if abs_dev_for_param < best_abs_dev_for_param :
            best_abs_dev_for_param = abs_dev_for_param
            best_model = param 
    count_param +=1 


print('\n\n\n\n\n\n\n\n!!!!!!!! AND THE BEST MODEL IS  ', best_model, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
