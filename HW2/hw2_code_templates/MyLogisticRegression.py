import numpy as np

class MyLogisticRegression:

    def __init__(self, d, max_iters, eta):
        self.eta = eta
        self.d = d
        self.max_iters = max_iters

    def fit(self, X, y):
        cost = []
        self.intercept = np.zeros(1)
        self.w_t = np.random.uniform(-0.01, 0.01, size = X.shape[1]-1)
        for i in range(self.max_iters):
            self.X_f = X[:, :-1]
            sigma_wx = sigmoid(self.X_f, self.w_t, self.intercept)
            error = sigma_wx - y
            grad = self.X_f.T.dot(error)
            self.intercept = self.intercept - self.eta*error.sum()
            self.w_t[0:] = self.w_t[0:] - self.eta*grad
            cost.append(lrCost(y, sigma_wx, self.d))
            if len(cost) >= 2:
                if np.abs(cost[len(cost) - 1] - cost[len(cost) - 2]) <= self.d:
                    break

    def predict(self, X):
        X_f = X[:, :-1]
        X_i = X[:, -1]
        p_1 = sigmoid(X_f, self.w_t, self.intercept*X_i)
        p_0 = 1 - sigmoid(X_f, self.w_t, self.intercept*X_i)
        prob = np.column_stack([p_0, p_1])
        prediction = np.where(prob[:, 1] >= prob[:, 0], int(1), int(0))
        return prediction

def sigmoid(X, w_t, intercept):
    z = np.dot(X, w_t.T) + intercept
    return 1.0 / ( 1.0 + np.exp(-z))

def lrCost(y, sigma_wx, d):
    j = -y@(np.log(np.clip(sigma_wx, d, None))) - ((1-y)@np.log(np.clip(1-sigma_wx, d, None)))
    return j