################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyAutoencoder import MyAutoencoder

from hw4_utils import load_MNIST, plot_points, convert_data_to_numpy

np.random.seed(2023)

batch_size = 10

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
import random
ind = random.sample(range(60000), k = 1000)
X, y = convert_data_to_numpy(train_dataset)

criterion = nn.MSELoss()
lr = 0.001

ae = MyAutoencoder(input_size = 784, learning_rate = lr, threshold = lr/2, max_epochs = 50)
optimizer = torch.optim.Adam(ae.parameters(), lr = lr)
ae.fit(train_loader = train_loader, criterion = criterion, optimizer = optimizer)

res_enc = ae.project(sel_features = X[ind, :], criterion = criterion)
plot_points(res_enc[:, 0], res_enc[:, 1], y[ind], 'hw4_q2_plot')
