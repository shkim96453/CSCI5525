################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

from matplotlib import pyplot as plt

from MyPCA import MyPCA

from hw4_utils import load_MNIST, convert_data_to_numpy, plot_points

np.random.seed(2023)

normalize_vals = (0.1307, 0.3081)

batch_size = 100

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

# convert to numpy
X, y = convert_data_to_numpy(train_dataset)

#####################
# ADD YOUR CODE BELOW
#####################
pca = MyPCA(num_reduced_dims=2)
pca.fit(X = X)
new_coord = pca.project(x = X)

plot_points(new_coord[:, 1], new_coord[:, 0], y, 'imgs/q1/hw4_q1.png')

