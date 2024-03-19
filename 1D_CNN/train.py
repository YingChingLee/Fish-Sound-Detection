from load import HARData
from load import prepro
import torch.nn as nn
from nn import CNN1D
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import librosa.feature
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset


AUDIO_0 = 'oneslyunew0.csv'
AUDIO_1 = 'oneslyunew1.csv'

BATCH_SIZE = 5
EPOCH = 50
LEARNING_RATE = 0.0001

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train(model, data_loader, loss_fn, optimizer,epo):
    for epoch in range(epo):
        running_loss = 0.0
        correct = 0
        total = 0
        x_test = np.zeros(1)
        x_pred = np.zeros(1)
        for i, (inputs, labels) in enumerate(data_loader):
            #print("Input tensor shape:", inputs.shape)
            #print("Input tensor shape:", labels.size(0))
            total += labels.size(0)
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1)
            #print(inputs.shape)
            #print("Input tensor shape:", inputs.shape)
            outputs = model(inputs.float())
            #print(outputs)
            labels = labels.to(torch.float)
            #print(labels)
            labels = labels.unsqueeze(1)
            #print(labels)
            #print(labels.size())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(i)
            outputss = (outputs > 0.5).float()
            predicted = outputss
            x_test = np.append(x_test, labels.numpy())
            x_pred = np.append(x_pred, predicted.numpy())
            correct += (predicted == labels).sum().item()
            #print(correct)
        print("Epoch {} loss: {}".format(epoch + 1, running_loss / len(data_loader)))
        print("Accuracy: {}%".format(100 * correct / total))
        x_test = x_test[1:]
        x_pred = x_pred[1:]
        tn, fp, fn, tp = confusion_matrix(x_test, x_pred).ravel()
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        acc = (tn+tp)/len(x_test)
        print(fnr,fpr)
    print("Finished training")

if __name__ == "__main__":
    # instantiating our dataset object and create data loader
    X_train,y_train,X_test,y_test = prepro(AUDIO_0,AUDIO_1)
    train_data = HARData(X_train, y_train)
    test_data = HARData(X_test, y_test)
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)

    # construct model and assign it to device
    model = CNN1D()
    # initialise loss funtion + optimiser
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    train(model, train_dataloader, loss_fn, optimizer, EPOCH)

    # save model
    torch.save(model.state_dict(), "1d.pth")
    print("Trained feed forward net saved at 1d.pth")
