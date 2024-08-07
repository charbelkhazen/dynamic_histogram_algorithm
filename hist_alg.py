import numpy as np
import pandas as pd 




def get_object_df(df , lst_vars = ['col1', 'col2', 'col3'] , values= ['item1', 'dubai', 'big'] ):  #input col names of object and values that define the object
    
    # Construct the condition dynamically
    condition = pd.Series([True] * len(df))
    for col, val in zip(lst_vars, values):
        condition &= (df[col] == val)
    
    return df[condition]
    
class HistAlg:
    '''
    refer to vicinity functions/methods below ( and to manual ) to understand the equally_spaced_ vars 
    placed them here as it emproves comp efficiency 
    '''
    equally_spaced_month = np.linspace(0.1, 0.9, 7)  # same logic as below ...
    equally_spaced_week = np.linspace(0.1, 0.9, 3)  # 3 values because min(delta, 4 - delta) gives 0, 1, 2 (3 values) - delta is difference between two weeks
    equally_spaced_day = np.linspace(0.1, 0.9, 4)   # 4 values because min(delta, 7 - delta) gives 0, 1, 2, 3 (4 values) delta is difference between two days
    equally_spaced_hour = np.linspace(0.1, 0.9, 6)  # 6 values because min(delta, 12 - delta) gives 0, 1, 2, 3, 4, 5 (6 values) ....
    equally_spaced_minute = np.linspace(0.1, 0.9, 7) # 7 values because min(delta, 30 - delta) gives 0, 1, 2, 3, 4, 5, 6 (7 values) ....

    def __init__(self, data , var_column_name = 'quantity', date_column_name = 'date' , horizon = 'mm-dd-yyyy', weight_month_vic = 0.1, weight_day_of_week_vic = 0  ,weight_week_of_month_vic = 0  , weight_hour_of_day_vic = 0 , weight_minute_of_hour = 0 , bins = [1] ) :   #hyperpar are set to default values  ,#there shoudl be 'date' and 'month' columns in df
        
        '''
        
        Make sure df has no column 'month' or 'day' , No need.  
        data should be a pandas df, it should at least contain a date column and a variable of interest column
        The date can be in any of these formats : 'yyyy-mm-dd', 'dd/mm/yyyy', 'mm-dd-yyyy', etc
        Forecasting horizon will have the same format 
        No need to create a particular month , day , week column ... Code handles this
        weight_... are hyperparameters of the histogram , its the contribution of vicinity metrics in overall vicinity measure . E.G. weight_month_vic is the contribution ( in % ) of month_vicinity in overall vicinity metric 
        bins is a hyperparameter of the histogram
     
        

        Granualirity of viccinity metric reaches 'minutes of hours' 
        For seconds and smaller granularity , code should be modified
        
        '''

        self.data = data
        self.horizon = horizon   # forecasting time 
        self.var_column_name = var_column_name
        self.date_column_name = date_column_name


        self.df = self.data[[self.var_column_name , self.date_column_name]]

        # Hyperparameters 
        self.weight_month_vic = weight_month_vic   
        self.weight_day_of_week_vic = weight_day_of_week_vic
        self.weight_week_of_month_vic = weight_week_of_month_vic
        self.weight_hour_of_day_vic = weight_hour_of_day_vic
        self.weight_minute_of_hour = weight_minute_of_hour


        self.bins = bins 
       


        self.horizon_date = datetime.strptime(date_str, '%m-%d-%Y')
        self.horizon_month = horizon_date.month 
        


        #Ideally I should have added this in header (global var) , but couldnt as it is dependent on some specific feature of the obj - min/max date 
        data_max_date = data['date'].max()  # not inside function for computing efficiency
        data_min_date = data['date'].min()   # not inside function for computing efficiency
        nbr_days = abs((data_max_date - data_min_date).days)    # not inside function for computing efficiency
        self.equally_spaced = np.linspace(0.1, 0.9, nbr_days +1 )  # not inside function for computing efficiency

        
    def set_params(self, **params):  
        for param in params.keys():
            setattr(self, param, params[param])
        return self
    

    # df specific to each TS
    

    def get_df_for_hist(object_df, time_column='date', variable='sales'):
        return object_df[[time_column, variable]]




    def month_vicinity(self , x, y):
        if not (1 <= x <= 12 and 1 <= y <= 12):
            raise ValueError(" Month variables should be between 1 and 12.")
        delta = abs(x - y)
        min_delta = min(delta, 12 - delta)
        vicinity = self.equally_spaced_month[6 - min_delta]
        return vicinity
    
    def week_of_month_vicinity(self, x, y):    # Note : In february we have exactly 4 weeks ( No extra days ) , but this does not affect our code. For both 5 days and 4 days we have the same number of equally spaced values in the metric ( 3 ) 
        if not (1 <= x <= 5 and 1 <= y <= 5):
            raise ValueError("Week variables should be between 1 and 5.")
        delta = abs(x - y)
        min_delta = min(delta, 5 - delta)
        vicinity = self.equally_spaced_week[2 - min_delta]
        return vicinity
    
    def day_of_week_vicinity(self, x, y):     
        if not (1 <= x <= 7 and 1 <= y <= 7):
            raise ValueError("Day variables should be between 1 and 7.")
        delta = abs(x - y)
        min_delta = min(delta, 7 - delta)
        vicinity = self.equally_spaced_day[3 - min_delta]
        return vicinity
    
    def hour_of_day_vicinity(self, x, y):
        if not (0 <= x < 24 and 0 <= y < 24):
            raise ValueError("Hour variables should be between 0 and 23.")
        delta = abs(x - y)
        min_delta = min(delta, 24 - delta)
        vicinity = self.equally_spaced_hour[5 - min_delta]
        return vicinity
    
    def minute_of_hour_vicinity(self, x, y):
        if not (0 <= x < 60 and 0 <= y < 60):
            raise ValueError("Minute variables should be between 0 and 59.")
        delta = abs(x - y)
        min_delta = min(delta, 60 - delta)
        vicinity = self.equally_spaced_minute[6 - min_delta]
        return vicinity
    
    def days_between(self,date1, date2):
        delta =  abs((date2 - date1).days) 
        vicinity = self.equally_spaced[2860 - delta]
        return vicinity 
    



    def find_vicinity(object_df):    
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.day_name()
        df['week_of_month'] = df['date'].apply(lambda x: (x.day - 1) // 7 + 1)

        # Apply the month_vicinity function to each row
        object_df['month_vicinity'] = object_df['month'].apply(lambda x: month_vicinity(x, y = horizon_month , equally_spaced_month))
        object_df['date_vicinity'] = object_df['date'].apply(lambda x: days_between(x, y = horizon_date , equally_spaced ))
        object_df['vicinity_weight'] = pct_param * object_df['month_vicinity'] + (1-pct_param) object_df['date_vicinity']
        
        return data  

    
    #honeh

    def find_vicinity(self, day=False, month=False, day_of_week=False, week_of_month=False, horizon_month=None, horizon_date=None, equally_spaced_month=None, equally_spaced=None, pct_param , pct_param2 = 0 , pct_param3 = 0 , pct_param4 =0 ): 
        object_df['date'] = pd.to_datetime(object_df['date'])
        
        if month:
            object_df['month'] = object_df['date'].dt.month
            object_df['month_vicinity'] = object_df['month'].apply(lambda x: month_vicinity(x, y=horizon_month, equally_spaced_month=equally_spaced_month))
        
        if day:
            object_df['day'] = object_df['date'].dt.day
            object_df['day_vicinity'] = object_df['day'].apply(lambda x: day_vicinity(x, y=horizon_month, equally_spaced_month=equally_spaced_month))
        
        if day_of_week:
            object_df['day_of_week'] = object_df['date'].dt.day_name()
            object_df['day_of_week_vicinity'] = object_df['day_of_week'].apply(lambda x: day_of_week_vicinity(x, y=horizon_month, equally_spaced_month=equally_spaced_month))
        
        if week_of_month:
            object_df['week_of_month'] = object_df['date'].apply(lambda x: (x.day - 1) // 7 + 1)
            object_df['week_of_month_vicinity'] = object_df['week_of_month'].apply(lambda x: week_vicinity(x, y=horizon_month, equally_spaced_month=equally_spaced_month))
        
        object_df['date_vicinity'] = object_df['date'].apply(lambda x: days_between(x, y=horizon_date, equally_spaced=equally_spaced))
        
        object_df['vicinity_weight'] = pct_param * object_df['month_vicinity'] +pct_param * object_df['month_vicinity'] +pct_param * object_df['month_vicinity'] + pct_param * object_df['month_vicinity'] +  (1-pct_param) object_df['date_vicinity']

        return object_df

    
    
    def ts_split(item_df,big_split = 0.8 ) 
        # Ensure the data is sorted by date
        df = df.sort_values('date')

        # Calculate the index to split the data
        split_index = int(len(df) * big_split)

        # Split the data into training and testing sets
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
        return train , test 
    
    def apply_histogram(item_df):    # Item specific 
        
        data.sort_values('quantity')
        hist_dict = data.groupby('quantity')['vicinity_weight'].sum().to_dict()

        return self.hist_dict 
    
    
    def sample_from_histogram(self):
        
        keys = list(hist_dict.keys())
        weights = list(hist_dict.values())

        # Sample 10 times from the dictionary according to the weights
        samples = random.choices(keys, weights=weights, k=10)
        sample = sum(samples) / len(samples)
        return sample
    
    def mean_absolute_percentage_error(y_true, y_pred):  #!!!!!!!!!!!!!!!!!!!!!  here herer 

        # Convert to numpy arrays for easier computation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Ensure no division by zero
        epsilon = np.finfo(np.float64).eps
        y_true = np.where(y_true == 0, epsilon, y_true)

        # Compute MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return mape
    


    def run(self,lst_vars):
        
        val_lst_vars = self.data[lst_vars].drop_duplicates().values.tolist()
        results = {}
        
        for values in val_lst_vars:
            print("Processing item_id ",', '.join(values) )
            
            object_df = get_object_df(lst_vars,values)
            
            train , test = self.ts_split(object_df ,0.8)
            
            hist_result = self.apply_histogram(train)
            scores = self.train_lightgbm(item_id)
            results[item_id] = {'histogram': hist_result, 'cv_scores': scores}
            
        return results

# Example usage
actual_values = [100, 200, 300, 400, 500]
predicted_values = [110, 190, 310, 420, 490]

mape = mean_absolute_percentage_error(actual_values, predicted_values)
print("MAPE:", mape)
