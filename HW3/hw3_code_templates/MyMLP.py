import numpy as np

import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        ### Construct your MLP Here (consider the recommmended functions in homework writeup)  
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        ### Use the layers you constructed in __init__ and pass x through the network
        ### and return the output
        relu = nn.ReLU()
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)

        return x

    def fit(self, train_loader, criterion, optimizer):
        '''
        Function used to train the MLP

        train_loader: includes the feature matrix and class labels corresponding to the training set,
        criterion: the loss function used,
        optimizer: which optimization method to train the model.
        '''
        total, err = 0, 0
        prediction = []
        # Epoch loop
        for i in range(self.max_epochs):

            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader, 0):
                images, labels = (images, labels)
                images = images.view(-1, 28*28)
                # Forward pass (consider the recommmended functions in homework writeup)
                outputs = self.forward(images)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                optimizer.zero_grad()
                # Track the loss and error rate
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, pred = torch.max(outputs.data, 1)
                prediction.append(outputs)
                total += labels.size(0)
                err += (pred != labels).sum().item()
            # Print/return training loss and error rate in each epoch
            print(i, ": ", {"loss": round(loss.item(), 4), "err_rate": round(err/total, 4)})

    def predict(self, test_loader, criterion):
        '''
        Function used to predict with the MLP

        test_loader: includes the feature matrix and classlabels corresponding to the test set,
        criterion: the loss function used.
        '''
        total, err = 0, 0
        prediction = []
        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader, 0):
                images, labels = (images, labels)
                images = images.view(-1, 28*28)

                # Compute prediction output and loss
                outputs = self.forward(images)
                # Measure loss and error rate and record
                loss = criterion(outputs, labels)
                _, pred = torch.max(outputs.data, 1)
                prediction.append(outputs)
                total += labels.size(0)
                err += (pred != labels).sum().item()
                
        # Print/return test loss and error rate
        print({"loss": round(loss.item(), 4), "err_rate": round(err/total, 4)})