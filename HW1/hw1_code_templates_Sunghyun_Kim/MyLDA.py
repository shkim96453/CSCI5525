import numpy as np

class MyLDA():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def fit(self, X, y):
        self.mean_vectors = []
        for cl in range(0,2):
            self.mean_vectors.append(np.mean(X[y==cl], axis=0))
        self.Sw = np.zeros((2,2))
        for cl,mv in zip(range(0,1), self.mean_vectors):
            class_sc_mat = np.zeros((2,2))
            for row in X[y == cl]:
                row = np.asarray(row)
                row, mv = row.reshape(2,1), mv.reshape(2,1) 
                class_sc_mat += (row-mv).dot((row-mv).T)
        self.Sw += class_sc_mat  
        self.w = np.linalg.inv(self.Sw)@(self.mean_vectors[1] - self.mean_vectors[0]).transpose()

    def predict(self, X):
        Fx = X@self.w.T
        prediction = np.where(Fx >= self.lambda_val, int(1), int(0))
        return prediction, self.w