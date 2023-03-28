################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyCNN import MyCNN

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
import matplotlib.pyplot as plt
lr = [1e-5, 1e-4, 1e-3, 0.01, 0.1]
criterion = nn.CrossEntropyLoss()

# SGD
criterion = nn.CrossEntropyLoss()
print('SGD')
for i in range(len(lr)):
    print(lr[i])
    cnn = MyCNN(input_size=28*28, output_size=10, kernel_size=3, stride_size=2, max_pool_size=2, learning_rate=lr[i], max_epochs=10)
    optim_sgd = torch.optim.SGD(cnn.parameters(), lr = lr[i])
    cnn_sgd = cnn.fit(train_loader = train_loader, criterion = criterion, optimizer = optim_sgd)
    cnn_sgd
    print("test")
    test_cnn_sgd, missed_img = cnn.predict(test_loader = test_loader, criterion = criterion)
    test_cnn_sgd

# Best Performing model
print("Model with best performing lr")
best_lr = 0.1
best_cnn = MyCNN(input_size=28*28, output_size=10, kernel_size=3, stride_size=2, max_pool_size=2, learning_rate=best_lr, max_epochs=10)
best_optim_sgd = torch.optim.SGD(best_cnn.parameters(), lr = best_lr)
best_cnn_sgd = best_cnn.fit(train_loader = train_loader, criterion = criterion, optimizer = best_optim_sgd)
best_cnn_sgd
print("test")
best_test_cnn_sgd, best_missed_img = best_cnn.predict(test_loader = test_loader, criterion = criterion)
best_test_cnn_sgd


# Plot Image
# Use Caution on running it as it will produce multiple images. 
'''
for i in range(len(best_missed_img)):
    missed_np = best_missed_img[i].numpy()
    plt.imshow(missed_np, 'gray')
    plt.show()
'''

