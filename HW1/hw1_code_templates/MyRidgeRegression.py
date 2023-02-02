import numpy as np

class MyRidgeRegression():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def fit(self, X, y):
        self.w = np.linalg.inv(X.transpose()@X + self.lambda_val*np.identity(X.shape[1]))@X.transpose()@y

    def predict(self, X):
        y_hat = X@self.w
        return y_hat

