import numpy as np
from data import sample_gmm_2d, graph_surface, graph_data
import matplotlib.pyplot as plt

class fcann2:
    def train(self, X, Y_, n_hidden=5, lr=0.05, reg_coe=0.001, steps=10000):
        """
        Parameters:
            X - data
            Y_ - true classes
            n_hidden - # of neurons in the hidden layer
            lr - learning rate
            steps - iterations

        Returns:
            W - weights
            b - biases
        """
        
        n_classes = np.max(Y_) + 1
        n_samples = X.shape[0]
        n_features = X.shape[1]

        W1 = 0.01 * np.random.randn(n_features, n_hidden)
        b1 = np.zeros(n_hidden)

        W2 = 0.01 * np.random.randn(n_hidden, n_classes)
        b2 = np.zeros(n_classes)

        for _ in range(steps):
            s1 = X @ W1 + b1
            h1 = np.maximum(0, s1)
            s2 = h1 @ W2 + b2
            exps = np.exp(s2)
            pp = exps / np.sum(exps, axis=1, keepdims=True)

            # loss = -np.mean(np.log(pp[np.arange(n_samples), Y_])) + reg_coe * (np.sum(W1**2) + np.sum(W2**2))

            Gs2 = pp
            Gs2[np.arange(n_samples), Y_] -= 1
            Gs2 /= n_samples
            GW2 = h1.T @ Gs2 + 2 * reg_coe * W2
            Gb2 = np.sum(Gs2, axis=0)
            Gh1 = Gs2 @ W2.T
            Gh1[s1 <= 0] = 0
            GW1 = X.T @ Gh1 + 2 * reg_coe * W1
            Gb1 = np.sum(Gh1, axis=0)

            W1 -= lr * GW1
            b1 -= lr * Gb1
            W2 -= lr * GW2
            b2 -= lr * Gb2

        W = [W1, W2]
        b = [b1, b2]

        self.W = W
        self.b = b

    def classify(self, X):
        """
        Parameters:
            X - data
            W - weights
            b - biases

        Returns:
            class_probs - each row contains the probabilities of classifying a point into each class
        """

        s1 = X @ self.W[0] + self.b[0]
        h1 = np.maximum(0, s1)
        s2 = h1 @ self.W[1] + self.b[1]
        return np.argmax(s2, axis=1)

if __name__=="__main__":
    np.random.seed(100)

    X,Y_ = sample_gmm_2d(6, 2, 10)

    f = fcann2()
    f.train(X, Y_)
    Y = f.classify(X)

    rect=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(f.classify, rect, 0.5)

    graph_data(X, Y_, Y, special=[])

    plt.show()