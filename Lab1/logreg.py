import numpy as np
import matplotlib.pyplot as plt
import data

class Logreg:
    def train(self, X, Y_, lr=0.01, steps=10000, print_step=500):
        """
        Parameters:
            X - data
            Y_ - true classes
            lr - learning rate
            steps - iterations
            print_step - print loss every # of steps; 0 for no print
        """
        n_samples, n_features = X.shape
        n_classes = np.max(Y_) + 1

        W = 0.01 * np.random.randn(n_features, n_classes)
        b = np.zeros((1, n_classes))

        for i in range(steps):
            scores = X @ W + b
            scores -= np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
            probs = exp_scores / sum_exp
            logprobs = np.log(probs)

            if print_step != 0 and i % print_step == 0:
                loss = -np.sum(logprobs[np.arange(n_samples), Y_]) / n_samples
                print(f"step: {i}, loss: {loss}")

            der = probs.copy()
            der[np.arange(n_samples), Y_] -= 1
            der /= n_samples
            grad_W = X.T @ der
            grad_b = np.sum(der, axis=0, keepdims=True)

            W -= lr * grad_W
            b -= lr * grad_b

        if print_step != 0:
            loss = -np.sum(logprobs[np.arange(n_samples), Y_]) / n_samples
            print(f"FINAL: loss: {loss}")

        self.W = W
        self.b = b

    def classify(self, X):
        """
        Parameters:
            X - data
        
        Returns: class probabilities matrix - each row contains the probabilities of classifying a point into each class
        """

        scores = X @ self.W + self.b
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

def classify_wrapper(model):
    def predict(X):
        probs = model.classify(X)
        return np.argmax(probs, axis=1)
    return predict

if __name__=="__main__":
    np.random.seed(100)

    X,Y_ = data.sample_gmm_2d(3, 3, 50)

    model = Logreg()
    model.train(X, Y_)
    probs = model.classify(X)
    Y = np.argmax(probs, axis=1)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(classify_wrapper(model), rect, 0.5)

    data.graph_data(X, Y_, Y, special=[])

    plt.show()