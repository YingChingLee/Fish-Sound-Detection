import numpy as np
import pandas as pd
import random

# data:file path of the label data
data = '/Data/FishSound/Green Island/label_lyu.csv'
# p:file path of the label data
# k:file path of the frame data
p = '/Data/FishSound/Green Island/label_lyu.csv'
k = '/Data/FishSound/Green Island/frame_lyu.csv'
#set the name of contents csv of dataset
csv_filename = 'ones_lyu.csv' 
#set output file's names
csv_filename0 = 'lyu_0.csv' #frames without fish sound
csv_filename1 = 'lyu_1.csv' #frames with fish sound

def create_continuous_ones_dataframe(input_dataframe, column_name):
    ones_series = []
    ones_count = 0
    start_index = None
    for index, row in input_dataframe.iterrows():
        if row[column_name] == 1:
            if ones_count == 0:
                start_index = index
            ones_count += 1
        else:
            if ones_count > 0:
                ones_series.append((start_index, index - 1, ones_count, [1] * ones_count))
                ones_count = 0
    if ones_count > 0:
        ones_series.append((start_index, index, ones_count, [1] * ones_count))
    
    df = pd.DataFrame(ones_series, columns=['Start_Index', 'End_Index', 'Count', 'Series'])
    #df['End_Index'] += 1
    df['range'] = df.apply(lambda row: (row['Start_Index'], row['End_Index']), axis=1)
    return df
    
# Function to generate 20 random numbers excluding specified ranges
def generate_random_numbers(ex_ranges,rownum,lent):
    random_numbers = []
    while len(random_numbers) < rownum:
        # Generate a random number between 0 and 900
        number = random.randint(0, lent)
        # Check if the number falls within any excluded range
        if not any(start <= number <= end for start, end in ex_ranges):
            random_numbers.append(number)
    return random_numbers

input_dataframe = pd.read_csv(data)
column_name = '0'
lenth = input_dataframe.shape[0]
ones_dataframe = create_continuous_ones_dataframe(input_dataframe, column_name)
ones_dataframe.to_csv(csv_filename, index=False)

count_sum = ones_dataframe['Count'].sum()
rangelist = ones_dataframe['range'].tolist()
ex_ranges = rangelist

zero_list = generate_random_numbers(ex_ranges,count_sum,lenth)
p = pd.read_csv(p, index_col=False)
k = pd.read_csv(k)
total = pd.concat([k, p], axis=1)

selected1 = pd.concat([total.iloc[start:end+1, :] for start, end in rangelist])
selected0 = pd.concat([total.loc[intd, :] for intd in zero_list])
selected0 = selected0.to_numpy().reshape(count_sum, 3201)
selected0 = pd.DataFrame(selected0)

selected0.to_csv(csv_filename0)
selected1.to_csv(csv_filename1)

