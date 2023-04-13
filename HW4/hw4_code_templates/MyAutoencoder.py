import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class MyAutoencoder(nn.Module):
    def __init__ (self, input_size, learning_rate, threshold, max_epochs):
        super(MyAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 400, bias = True)
        self.fc2 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400, bias = True)
        self.fc4 = nn.Linear(400, input_size, bias = True)
        self.flat = nn.Flatten()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_epochs = max_epochs

    def forward(self, x):
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        enc_x = self.flat(x)
        enc_x = self.fc1(enc_x)
        enc_x = tanh(enc_x)
        enc_x = self.fc2(enc_x)
        enc_x = tanh(enc_x)
        dec_x = self.fc3(enc_x)
        dec_x = tanh(dec_x)
        dec_x = self.fc4(dec_x)
        dec_x = sigmoid(dec_x)

        return enc_x, dec_x

    def fit(self, train_loader, criterion, optimizer):
        loss_list = []
        epochs = []
        # Epoch loop
        for i in range(self.max_epochs):
            loss = 0
            # Mini batch loop
            for j,(features,_) in enumerate(train_loader, 0):
                features, _ = (features, _)
                features = features.view(-1, 28*28)
                # Forward pass (consider the recommmended functions in homework writeup)
                res_enc, res_dec = self.forward(features)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                optimizer.zero_grad()
                # Track the loss and error rate
                train_loss = criterion(res_dec, features)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()

            # Print/return training loss and error rate in each epoch
            loss = loss / len(train_loader)
            loss_list.append(loss)
            epochs.append(i + 1)
            print(i+1, ": ", {"loss": round(loss, 4)})

            if len(loss_list) > 10:
                last_five = loss_list[-5:]
                last_five_diff = [np.abs(t - s) for s, t in zip(last_five, last_five[1:])]
                if np.mean(last_five_diff) < self.threshold:
                    plot_df = np.column_stack((epochs, loss_list))
                    plt.plot(plot_df[:, 0], plot_df[:, 1])
                    plt.xticks(np.arange(min(plot_df[:, 0]), max(plot_df[:, 0])+1, 1.0))
                    plt.yticks(np.arange(min(plot_df[:, 1]), max(plot_df[:, 1])+0.01, 0.01))
                    plt.savefig('imgs/q2/hw4_q2_train_loss.png')
                    break

    def project(self, sel_features, criterion):
        with torch.no_grad(): # no backprop step so turn off gradients
            sel_features = torch.from_numpy(sel_features)
            res_enc, res_dec = self.forward(sel_features)
            # Measure loss and error rate and record
            loss = criterion(res_dec, sel_features)
            res_enc = res_enc.numpy()

        # Print/return test loss and error rate
        print({"loss": round(loss.item(), 4)})
        return res_enc
