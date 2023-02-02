################################
# DO NOT EDIT THE FOLLOWING CODE
################################
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np

from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val

# load dataset
X, y = fetch_california_housing(return_X_y=True)

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

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 1, 10, 100]

#####################
# ADD YOUR CODE BELOW
#####################

for lambda_val in lambda_vals:
    print(lambda_val)
    # instantiate ridge regression object
    rr_model = MyRidgeRegression(lambda_val)
    # call to your CV function to compute mse for each fold
    rr_mse_vals = my_cross_val(rr_model, 'mse', X, y)
    # print mse from CV
    print("Ridge Regression MSE by Fold", rr_mse_vals)
    # instantiate lasso object
    ls_model = Lasso(lambda_val)
    # call to your CV function to compute mse for each fold
    ls_mse_vals = my_cross_val(rr_model, 'mse', X, y)
    # print mse from CV
    print("Lasso Regression MSE by Fold", ls_mse_vals)
# instantiate ridge regression and lasso objects for best values of lambda
rr_model_best = MyRidgeRegression(0.01)
# fit models using all training data
rr_model_best.fit(X_train, y_train)
# predict on test data
rr_model_best.predict(X_test)
# compute mse on test data
best_rr_mse_vals = my_cross_val(rr_model_best, 'mse', X_test, y_test)
# print mse on test data
print("Ridge Regression MSE by fold, Best Lambda on Test Set", best_rr_mse_vals)
