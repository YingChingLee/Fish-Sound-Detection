import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import librosa.feature
from torch.utils.data import DataLoader, Dataset

def prepro(frame_0,frame_1):
    P = frame_0
    K = frame_1
    selected0 = pd.read_csv(P, index_col=0)
    selected0label = selected0.iloc[:,-1]
    selected1 = pd.read_csv(K, index_col=0)
    selected1label = selected1.iloc[:,-1]
    selectedlabel = pd.concat([selected0label, selected1label], axis=0,  ignore_index=True)
    selected0frame = selected0.iloc[:, :-1]
    selected1frame = selected1.iloc[:, :-1]
    selectedframe = pd.concat([selected0frame, selected1frame], axis=0,  ignore_index=True)
    selected = pd.concat([selectedframe, selectedlabel], axis=1)
    selected_label = selected.iloc[:, -1]
    selected_frame = selected.iloc[:, :-1]

    p = selected_label.values
    k = selected_frame.values
    
    train_data, test_data, train_labels, test_labels = train_test_split(k, p, test_size=0.2, random_state=42)
    X_train = torch.tensor(train_data)
    y_train = torch.tensor(train_labels)
    X_test = torch.tensor(test_data)
    y_test = torch.tensor(test_labels)
    return X_train,y_train,X_test,y_test

class HARData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


if __name__ == "__main__":
    AUDIO_0 = 'oneslyunew0.csv'
    AUDIO_1 = 'oneslyunew1.csv'
    X_train,y_train,X_test,y_test = prepro(AUDIO_0,AUDIO_1)
    train_data = HARData(X_train, y_train)
    test_data = HARData(X_test, y_test)
