import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from hw4_utils import convert_data_to_numpy

class MyDiscriminator(nn.Module):
    def __init__ (self, input_size):
        super(MyDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024, bias = True)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256, bias = True)
        self.fc4 = nn.Linear(256, 1, bias = True)
        self.drop = nn.Dropout(p = 0.3)
        self.flat = nn.Flatten()

    def forward(self, x):
        sigmoid = nn.Sigmoid()
        relu = nn.ReLU()
        x = self.flat(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        x = sigmoid(x)

        return x
    

