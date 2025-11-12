import numpy as np

class LinearRegression:
    def __init__(self,lr=0.001,n_iters=1000,penalty = None,alpha=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.penalty = penalty
        self.bias = None
        self.alpha = alpha
        self.weights = None

    def fit(self,X,y):
        if self.penalty not in [None, 'l1', 'l2']:
            raise ValueError("penalty must be None, 'l1', or 'l2'")

        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        tol = 1e-5
        prev_mse = float('inf')
        for i in range(self.n_iters):
            y_pred = np.dot(X,self.weights) + self.bias
            mse = (1/n_samples) * (np.sum(np.square(y_pred - y)))
            gw = (2/n_samples) * np.dot(X.T,(y_pred-y))
            if self.penalty == 'l1':
                gw += self.alpha * np.sign(self.weights)
            if self.penalty == 'l2':
                gw += self.alpha * 2 * self.weights
            gb = (2/n_samples) * np.sum(y_pred-y)
            self.weights = self.weights - self.lr * gw
            self.bias = self.bias - self.lr * gb

            if abs(prev_mse - mse) < tol:
                break

            prev_mse = mse

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
    
    def score(self,X,y):
        y_pred = self.predict(X)
        ss_res = np.sum((y-y_pred) ** 2)
        ss_tot = np.sum((y-np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)