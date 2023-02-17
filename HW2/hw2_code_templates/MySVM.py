import numpy as np

class MySVM:

    def __init__(self, d, max_iters, eta, c):
        self.d = d
        self.max_iters = max_iters
        self.eta = eta
        self.c = c

    def fit(self, X, y):
        self.intercept = np.zeros(1)
        self.w_t = np.random.uniform(-0.01, 0.01, size = X.shape[1]-1)
        hinge_loss = []
        cost = []
        X_f = X[:, :-1] # Only features
        X_i = X[:, -1] # Only intercept
        for i in range(self.max_iters):
            hl_comp = y@(X_f@self.w_t + self.intercept*X_i)
            hinge_loss.append(max(0, 1 - hl_comp))
            error = X_f@self.w_t - y
            self.intercept = self.intercept - self.eta*error.sum()
            grad = X_f.T.dot(error)
            if hinge_loss[len(hinge_loss) - 1] == 0:
                self.w_t[0:] = self.w_t[0:] - self.eta*grad
                cost.append(svmCost(self.w_t[0:], self.c, hinge_loss[len(hinge_loss) - 1]))
            else:
                self.w_t[0:] = self.w_t[0:] - self.eta*grad
                cost.append(svmCost(self.w_t[0:], self.c, hinge_loss[len(hinge_loss) - 1]))
            if len(cost) > 2:
                if np.abs(cost[len(cost) - 1] - cost[len(cost) - 2]) <= self.d:
                    break

    def predict(self, X):
        X_f = X[:, :-1] # Only features
        X_i = X[:, -1] 
        prediction = X_f@self.w_t + self.intercept*X_i
        return np.sign(prediction)

def svmCost(w_t, c, hinge_loss):
    j = 0.5*np.linalg.norm(w_t, ord=2) + c*np.sum(hinge_loss)
    return j

