from load import HARData
from load import prepro
import torch.nn as nn
from nn_lstm import CNN1D
from train import create_data_loader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import librosa.feature
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

AUDIO_0 = 'lyu_0.csv'
AUDIO_1 = 'lyu_1.csv'
checkpoint_path = '1dm.pth'
BATCH_SIZE = 5
def evaluate(testset,mod,parameter):
    model = mod
    checkpointpath = parameter
    checkpoint = torch.load(checkpointpath)
    model.load_state_dict(checkpoint)
    model.eval()
    for j in range(10):
        thre = 0.1*j
        correct = 0
        total = 0
        y_test = np.zeros(1)
        y_pred = np.zeros(1)
        with torch.no_grad():
            for inputs, labels in testset:
                inputs = inputs.unsqueeze(1)
                
                total += labels.size(0)
                labels = labels.squeeze()
                
                outputs = model(inputs.float())
                outputs = (outputs > thre).float()
                outputs = torch.transpose(outputs, 0, 1)
                
                y_test = np.append(y_test, labels.numpy())
                
                predicted = outputs.squeeze()
                y_pred = np.append(y_pred, predicted.numpy())
                correct += (predicted == labels).sum().item()
        print(j)
        print("Accuracy: {}%".format(100 * correct / total))
        y_test = y_test[1:]
        y_pred = y_pred[1:]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        acc = (tn+tp)/len(y_test)
        print(fnr,fpr)

if __name__ == "__main__":
    X_train,y_train,X_test,y_test = prepro(AUDIO_0,AUDIO_1)
    test_data = HARData(X_test, y_test)
    test_dataloader = create_data_loader(test_data, BATCH_SIZE)
    evaluate(test_dataloader,CNN1D(),checkpoint_path)
