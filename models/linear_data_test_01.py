import numpy as np
from mlp import MLP

from batchup import data_source
import csv

csvfile = open('../../dataset/HIGGS_subset.csv','r')
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
        test_Y.append(int(float(row[0])))
    else:
        train_X.append(row[1:])
        train_Y.append(int(float(row[0])))
    size+=1
    
ds = data_source.ArrayDataSource([train_X, train_Y])

minibatch_size = 16

perceptron = MLP()

for i in range(10):
    for batch_X, batch_Y in ds.batch_iterator(batch_size=minibatch_size, shuffle=True):
        error = perceptron.train(batch_X, batch_Y)
        print(error)
    error = perceptron.test(batch_X, batch_Y)
    print("Test Error: "+str(error))

# for i in range(10000):
#     error = perceptron.train(train_X[0], train_Y[1])
#     print(error)
