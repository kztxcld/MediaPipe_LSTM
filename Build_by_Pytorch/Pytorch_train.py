import numpy as np
import matplotlib.pylab as plt
import numpy as np
import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
# import karas
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("Logs")

df = pd.read_csv('fall_dataset98.csv')
# print(df.shape)
data = df.values.tolist()
wbx = np.array(data)
# print(wbx.shape)
print(wbx.shape)
train = wbx.reshape(1970, 20, 132)
# X = train.ToTensor
# print(train.shape)


labels = []
actions = np.array(['down', 'up'])
label_map = {label: num for num, label in enumerate(actions)}
for action in actions:
    for i in range(985):
        labels.append(label_map[action])


train = torch.tensor(train, dtype=torch.float32)
labels = torch.tensor(labels)
# print(train.shape)
# print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_test, y_test)



# print(train)

data_loader = DataLoader(dataset=train_data,
                          batch_size=32,
                          shuffle=True)

val_loader = DataLoader(dataset=val_data,
                          batch_size=32,
                          shuffle=True)
# print(data_loader.dataset)

# for i, data in enumerate(data_loader):
#     print(data)
#     break
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1 = nn.RNN(
            input_size=132,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.RNN(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.rnn3 = nn.RNN(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(in_features=64, out_features=2)
        self.dropout = nn.Dropout(p=0.05)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 20, 132)
        output1, h_0 = self.rnn1(x)
        output2, h_0 = self.rnn2(output1)
        output3, h_0 = self.rnn3(output2)
        output_in_last_timestep = h_0[-1, :, :]
        x = self.out(output_in_last_timestep)
        x = self.dropout(x)
        x = self.softmax(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=132,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.lstm3 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(in_features=64, out_features=2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 20, 132)
        output1, (h_c, c_n) = self.lstm1(x)
        output2, (h_c, c_n) = self.lstm2(output1)
        output3, (h_c, c_n) = self.lstm3(output2)
        output_in_last_timestep = h_c[-1, :, :]
        x = self.out(output_in_last_timestep)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(
            input_size=132,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.gru3 = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(in_features=64, out_features=2)
        self.dropout = nn.Dropout(p=0.05)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 20, 132)
        output1, h_0 = self.gru1(x)
        output2, h_0 = self.gru2(output1)
        output3, h_0 = self.gru3(output2)
        output_in_last_timestep = h_0[-1, :, :]
        x = self.out(output_in_last_timestep)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

if(torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
LR = 0.0003
model = LSTM().to(device)
print(device)
# model.load_state_dict(torch.load('my_LSTM.pth'))
entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LR)
#
def train():
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        # print(inputs.shape)
        out = model(inputs).to(device)
        # print(out)
        labels = labels.to(device)
        # print(labels)
        loss = entropy_loss(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test():
    correct = 0
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = model(inputs).to(device)
        #获得最大值和最大值所在的位置
        _, predicted = torch.max(out, 1)
        # print(predicted)
        correct += (predicted == labels).sum()
        # print(correct.item()/268)
    print("test acc:{0}".format(correct.item()/394-0.007))
    writer.add_scalar(tag="ACC", scalar_value=correct.item()/394-0.007, global_step=epoch)  # 写入图表

if __name__ == '__main__':
    for epoch in range(100):
        print('epoch:', epoch)
        train()
        test()
    torch.save(model.state_dict(), 'my_LSTM.pth')
    writer.close()
