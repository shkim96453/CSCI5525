################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyGenerator import MyGenerator
from MyDiscriminator import MyDiscriminator

from hw4_utils import load_MNIST

np.random.seed(2023)

batch_size = 128

normalize_vals = (0.5, 0.5)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################
import time
import random
startTime = time.time()
class MyGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(MyGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        z = torch.randn(128, 128)
        fake = self.generator.forward(z)

        real_output = self.discriminator.forward(x)
        fake_output = self.discriminator.forward(fake)
        return fake_output, real_output, fake

    def fit(self, train_loader, threshold , max_epochs):
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr = 0.0002)
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr = 0.0002)
        d_loss_list = []
        g_loss_list = []
        epochs = []
        fake_imgs = torch.empty((1, 784))
        selected_imgs = []
        for i in range(max_epochs):
            d_loss = 0
            g_loss = 0
            for j, (real_features, _) in enumerate(train_loader):
                real_features, _ = (real_features, _)
                real_features = real_features.view(-1, 784)
                fake_output, real_output, fake_images = self.forward(real_features) 

                self.discriminator.zero_grad()
                d_loss = nn.BCELoss()(real_output, torch.ones_like(real_output)) + nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
                d_loss.backward(retain_graph = True)
                disc_optim.step()
                d_loss += d_loss.item()

                self.generator.zero_grad()
                re_fake_output, _, __ = self.forward(real_features)
                g_loss = nn.BCELoss()(re_fake_output, torch.ones_like(fake_output))
                g_loss.backward()
                gen_optim.step()
                g_loss += g_loss.item()

                sel_img = fake_images
            fake_imgs = torch.vstack((fake_imgs, sel_img))
            rand_ind = random.sample(range(fake_imgs.shape[0]), k = 1)
            img_from_epoch = fake_imgs[rand_ind]
            selected_imgs.append(img_from_epoch)
            self.selected_imgs = selected_imgs

            d_loss = d_loss.item()
            g_loss = g_loss.item()
            d_loss_list.append(d_loss)
            g_loss_list.append(g_loss)
            epochs.append(i+1)
            print(i+1, ": ", {"d_loss": round(d_loss,4), "g_loss": round(g_loss,4)})

            if i > 10:
                d_last_ten = d_loss_list[-10:]
                d_last_ten_diff = [np.abs(t - s) for s, t in zip(d_last_ten, d_last_ten[1:])]
                g_last_ten = g_loss_list[-10:]
                g_last_ten_diff = [np.abs(t - s) for s, t in zip(g_last_ten, g_last_ten[1:])]
                
                if np.mean(d_last_ten_diff) < threshold and np.mean(g_last_ten_diff) < threshold and np.mean(d_last_ten) > np.mean(g_last_ten) or i == max_epochs-1:
                    plot_df = np.column_stack((epochs, d_loss_list, g_loss_list))
                    plt.plot(plot_df[:, 0], plot_df[:, 1], color = 'orange')
                    plt.plot(plot_df[:, 0], plot_df[:, 2], color = 'blue')
                    plt.xticks(np.arange(min(plot_df[:, 0]), max(plot_df[:, 0]), 5.0))
                    #plt.yticks(np.arange(min(plot_df[:, 1]), max(plot_df[:, 1])+0.01, 0.01))
                    plt.legend(['D', 'G'])
                    plt.savefig('imgs/q3/hw4_q3_train_loss')
                    break

    def img_sel(self):
        rand_five = random.sample(range(len(self.selected_imgs)), k = 5)
        epochs = [x+1 for x in rand_five]
        reshape_di = int(np.sqrt(self.selected_imgs[0].shape[1]))
        fig,ax = plt.subplots()
        for i in range(len(rand_five)):
            img_to_plot = self.selected_imgs[rand_five[i]]
            img_to_plot = img_to_plot.view(-1, reshape_di, reshape_di)
            img_to_plot = img_to_plot.detach().numpy()
            img_to_plot = np.reshape(img_to_plot, (reshape_di, reshape_di))
            ax.imshow(img_to_plot, 'gray')
            plt.savefig('imgs/q3/hw4_q3_fake_imgs_epoch ' + str(epochs[i]))
        



gen = MyGenerator()
disc = MyDiscriminator(input_size = 784)
gan = MyGAN(generator=gen, discriminator=disc)

gan.fit(train_loader=train_loader, threshold=0.1, max_epochs=70)
gan.img_sel()

executionTime = (time.time() - startTime)
print('Execution time in seconds: ', str(executionTime))