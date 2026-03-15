import numpy as np
import data
import matplotlib.pyplot as plt

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

def eval_perf_multi(Y, Y_):
    """
    Parameters:
        Y - predicted classes
        Y_ - true classes

    Returns:
        accuracy
        confusion matrix
        precisions
        recalls
    """

    n_classes = max(Y.max(), Y_.max()) + 1
    conf = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(Y_, Y):
        conf[t, p] += 1

    accuracy = np.trace(conf) / np.sum(conf)

    precision = np.zeros(n_classes)
    for i in range(n_classes):
        sum = np.sum(conf[:, i])
        if sum > 0:
            precision[i] = conf[i, i] / np.sum(conf[:, i])

    recall = np.zeros(n_classes)
    for i in range(n_classes):
        sum = np.sum(conf[i, :])
        if sum > 0:
            recall[i] = conf[i, i] / np.sum(conf[i, :])

    return accuracy, conf, precision, recall

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