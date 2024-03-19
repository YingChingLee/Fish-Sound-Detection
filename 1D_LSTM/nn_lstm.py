import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=30, stride=15)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=20, stride=10)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.relu4 = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(2*128, 128)
        self.relu5 = nn.Sigmoid()  
        self.fc2 = nn.Linear(128, 64)
        self.relu6 = nn.Sigmoid()
        self.fc3 = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

        

    def forward(self, x):
        #print("Input tensor shape:", x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        #print("Input tensor shape:", x.shape)
        x = out.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x
if __name__ == "__main__":
    model = CNN1D()
