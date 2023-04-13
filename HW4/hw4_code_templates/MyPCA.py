import numpy as np
from scipy.linalg import eigh

class MyPCA():
    
    def __init__(self, num_reduced_dims):
        self.num_reduced_dims = num_reduced_dims

    def fit(self, X):
        cov = np.matmul(X.T, X)
        eig_vals, eig_vecs = eigh(cov)
        eig_vecs = eig_vecs[:, -self.num_reduced_dims:]
        self.eig_vecs = eig_vecs

    def project(self, x):
        new_coord = np.matmul(self.eig_vecs.T, x.T)
        new_coord = new_coord.T
        return new_coord