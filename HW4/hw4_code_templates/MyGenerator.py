import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class MyGenerator(nn.Module):
    def __init__ (self):
        super(MyGenerator, self).__init__()
        self.fc1 = nn.Linear(128, 256, bias = True)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024, bias = True)
        self.fc4 = nn.Linear(1024, 784, bias = True)

    def forward(self, x):
        tanh = nn.Tanh()
        relu = nn.ReLU()
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        x = relu(x)
        x = self.fc4(x)
        x = tanh(x)

        return x



