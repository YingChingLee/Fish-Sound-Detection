from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import torch
import librosa.feature
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
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

import random

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

# data:file path of the label data
data = '/Data/FishSound/Keelung Chaojing/label800chaotest.csv'
input_dataframe = pd.read_csv(data)
column_name = '0'
lenth = input_dataframe.shape[0]
ones_dataframe = create_continuous_ones_dataframe(input_dataframe, column_name)

csv_filename = 'oneschaonew.csv'
ones_dataframe.to_csv(csv_filename, index=False)

count_sum = ones_dataframe['Count'].sum()
rangelist = ones_dataframe['range'].tolist()
ex_ranges = rangelist

# p:file path of the label data
# k:file path of the frame data
p = '/Data/FishSound/Keelung Chaojing/label800chaotest.csv'
k = '/Data/FishSound/Keelung Chaojing/800framechaotest.csv'
zero_list = generate_random_numbers(ex_ranges,count_sum,lenth)
p = pd.read_csv(p, index_col=False)
k = pd.read_csv(k)
total = pd.concat([k, p], axis=1)

selected1 = pd.concat([total.iloc[start:end+1, :] for start, end in rangelist])
selected0 = pd.concat([total.loc[intd, :] for intd in zero_list])
selected0 = selected0.to_numpy().reshape(count_sum, 3201)
selected0 = pd.DataFrame(selected0)

#set output file's names
csv_filename0 = 'oneschaonew0.csv'
selected0.to_csv(csv_filename0)
csv_filename1 = 'oneschaonew1.csv'
selected1.to_csv(csv_filename1)

