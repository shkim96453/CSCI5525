################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MySVM import MySVM

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

# change labels from 0 and 1 to -1 and 1 for SVM
y[y == 0] = -1

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# append column of 1s to include intercept
X = np.hstack((X, np.ones((num_data, 1))))
num_data, num_features = X.shape

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

#####################
# ADD YOUR CODE BELOW
#####################

# Import your CV package here (either your my_cross_val or sci-kit learn )
from my_cross_val import my_cross_val

eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [0.01, 0.1, 1, 10, 100]
# Warning Switch to turn off verbose warning about runtime warning
np.seterr(all="ignore")
# SVM
for eta_val in eta_vals:
    for c_val in C_vals:
        print("eta: ", eta_val, "c: ", c_val)
        # instantiate svm object
        svm = MySVM(d = 10**-6, max_iters = 500, eta = eta_val, c = c_val)
        # call to CV function to compute error rates for each fold
        svm_err_rate = my_cross_val(svm, loss_func = 'err_rate', X = X, y = y)
        # print error rates from CV
        print(svm_err_rate)
# instantiate svm object for best value of eta and C
best_eta = 1e-05
best_c = 1
best_svm = MySVM(d = 10**-6, max_iters = 1000, eta = best_eta, c = best_c)
# fit model using all training data
best_svm.fit(X_train, y_train)
# predict on test data
best_svm.predict(X_test)
# compute error rate on test data
best_svm_err_rate = my_cross_val(best_svm, loss_func = 'err_rate',X = X_test, y = y_test)
# print error rate on test data
print("Error Rate with Best Parameters, eta = 1e-5, c = 1.", best_svm_err_rate)