from sklearn.svm import SVC
from pt_logreg import one_hot
import numpy as np
import data
import matplotlib.pyplot as plt
import pt_deep as deep
from pt_logreg import train, eval
import torch

class KSVMWrap:
    def __init__(self, X, Y_, C=1, gamma='auto'):
        """
        Parameters:
            X - data
            Y_ - true classes
            C - svm regularization
            gamma - RBF kernel width
        """
        
        self.model = SVC(C=C, gamma=gamma, kernel='rbf', probability=True)
        self.model.fit(X, Y_)
        self.support_ = self.model.support_
        self.Yoh = one_hot(Y_)

    def predict(self, X):
        """
        Parameters:
            X - data
            
        Returns: predictions
        """

        return self.model.predict(X)

    def get_scores(self, X):
        """
        Parameters:
            X - data

        Returns: predicted probabilities
        """

        return self.model.predict_proba(X)

    @property
    def support(self):
        """
        Returns: indices of support vectors
        """
        return self.support_

if __name__=="__main__":
    np.random.seed(100)

    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    model = KSVMWrap(X, Y_)
    Y = model.predict(X)
    
    accuracy, conf, precisions, recalls = data.eval_perf_multi(Y, Y_)
    print(f"accuracy: {accuracy}")
    print(f"confusion matrix:\n{conf}")
    print(f"precisions: {precisions}")
    print(f"recalls: {recalls}")

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(model.predict, rect)
    data.graph_data(X, Y_, Y, model.support)
    plt.show()

    data_params = [
        (2, 2, 10),
        (2, 2, 50),
        (6, 2, 10),
        (6, 2, 50),
        (4, 3, 10),
        (4, 3, 50)
    ]

    seeds = [
        44,
        100,
        100,
        100,
        100,
        100
    ]

    fig, axs = plt.subplots(3, 4, figsize=(15, 15))
    axs = axs.flatten()
    index = 0

    print()
    for data_param, seed in zip(data_params, seeds):
        np.random.seed(seed)
        X, Y_ = data.sample_gmm_2d(*data_param)
        Yoh_ = one_hot(Y_)
        X_torch = torch.tensor(X, dtype=torch.float32)
        Yoh_torch = torch.tensor(Yoh_, dtype=torch.float32)
        rect = (np.min(X, axis=0), np.max(X, axis=0))
        
        model_deep = deep.PTDeep([2, 10, 10, data_param[1]])
        train(model_deep, X_torch, Yoh_torch, 0.0001, 0.01, 5000, 0)
        probs = eval(model_deep, X)
        Y = np.argmax(probs, axis=1)
        def predict(X):
            return np.argmax(eval(model_deep, X), axis=1)

        plt.sca(axs[index])
        index += 1
        data.graph_surface(predict, rect, 0.5)
        data.graph_data(X, Y_, Y)
        plt.title(f"PTDeep {data_param}")

        accuracy, _, _, _ = data.eval_perf_multi(Y, Y_)
        print(f"PTDeep {data_param} accuracy: {accuracy}")

        model_ksvm = KSVMWrap(X, Y_)
        Y = model_ksvm.predict(X)

        plt.sca(axs[index])
        index += 1
        data.graph_surface(model_ksvm.predict, rect)
        data.graph_data(X, Y_, Y, model_ksvm.support)
        plt.title(f"KSVMWrap {data_param}")

        accuracy, _, _, _ = data.eval_perf_multi(Y, Y_)
        print(f"KSVMWrap {data_param} accuracy: {accuracy}")

    plt.show()