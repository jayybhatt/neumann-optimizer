import torch
import torch.nn as nn
from optimizer import Neumann
import numpy as np

from batchup import data_source
import csv

csvfile = open('../dataset/HIGGS_subset.csv','r')
csvreader = csv.reader(csvfile)
train_X = []
train_Y = []

test_X = []
test_Y = []

train_error = []

size = 0
for row in csvreader:
    if size >= 90000:
        test_X.append(row[1:])
        test_Y.append([float(row[0])])
    else:
        train_X.append(row[1:])
        train_Y.append([float(row[0])])
    size+=1

train_X = np.array(train_X,dtype="float64")
train_Y = np.array(train_Y,dtype="int32")
test_X = np.array(test_X, dtype="float64")
test_Y = np.array(test_Y, dtype="int32")
ds = data_source.ArrayDataSource([train_X, train_Y])


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, out_classes):
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


input_size = 28
hidden_size = 56
num_classes = 1
learning_rate = 1e-4
num_epochs = 20
minibatch_size = 16
current_iter = 1
device = torch.device('cpu')

net = MultilayerPerceptron(input_size, hidden_size, num_classes)

loss_fn = nn.MSELoss()
optimizer = Neumann(list(net.parameters()), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_X, batch_Y in ds.batch_iterator(batch_size=minibatch_size, shuffle=True):
        input = torch.tensor(batch_X, requires_grad=True, device=device, dtype=torch.float32)
        label = torch.tensor(batch_Y, device=device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = net(input)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        print("Loss: ", loss.data.numpy())

test_inputs = torch.tensor(test_X, device=device)
test_labels = torch.tensor(test_Y, device=device)

outputs = net(test_inputs)

print(outputs)
