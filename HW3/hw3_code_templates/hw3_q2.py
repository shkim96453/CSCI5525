################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyMLP import MyMLP

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
lr = [1e-5, 1e-4, 1e-3, 0.01, 0.1]
criterion = nn.CrossEntropyLoss()

# SGD
print('SGD')
for i in range(len(lr)):
    print(lr[i])
    mlp = MyMLP(input_size=28*28, hidden_size=128, output_size=10, learning_rate=lr[i], max_epochs=10)
    optim_sgd = torch.optim.SGD(mlp.parameters(), lr = lr[i])
    mlp_sgd = mlp.fit(train_loader = train_loader, criterion = criterion, optimizer = optim_sgd)
    mlp_sgd
    print("test")
    test_mlp_sgd = mlp.predict(test_loader = test_loader, criterion = criterion)
    test_mlp_sgd

# Adagrad
print('Adagrad')
for i in range(len(lr)):
    print(lr[i])
    mlp = MyMLP(input_size=28*28, hidden_size=128, output_size=10, learning_rate=lr[i], max_epochs=10)
    optim_sgd = torch.optim.Adagrad(mlp.parameters(), lr = lr[i])
    mlp_sgd = mlp.fit(train_loader = train_loader, criterion = criterion, optimizer = optim_sgd)
    mlp_sgd
    print("test")
    test_mlp_sgd = mlp.predict(test_loader = test_loader, criterion = criterion)
    test_mlp_sgd

# RMSprop
print('RMSprop')
for i in range(len(lr)):
    print("Training Loss and Error Rate for LR", lr[i])
    mlp = MyMLP(input_size=28*28, hidden_size=128, output_size=10, learning_rate=lr[i], max_epochs=10)
    optim_sgd = torch.optim.RMSprop(mlp.parameters(), lr = lr[i])
    mlp_sgd = mlp.fit(train_loader = train_loader, criterion = criterion, optimizer = optim_sgd)
    mlp_sgd
    print("test")
    test_mlp_sgd = mlp.predict(test_loader = test_loader, criterion = criterion)
    test_mlp_sgd

# Adam
print('Adam')
for i in range(len(lr)):
    print(lr[i])
    mlp = MyMLP(input_size=28*28, hidden_size=128, output_size=10, learning_rate=lr[i], max_epochs=10)
    optim_sgd = torch.optim.Adam(mlp.parameters(), lr = lr[i])
    mlp_sgd = mlp.fit(train_loader = train_loader, criterion = criterion, optimizer = optim_sgd)
    mlp_sgd
    print("test")
    test_mlp_sgd = mlp.predict(test_loader = test_loader, criterion = criterion)
    test_mlp_sgd

