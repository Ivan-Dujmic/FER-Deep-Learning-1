import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt
import logreg

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """
        Parameters:
            D - dimensions of each datapoint 
            C - number of classes
        """

        super().__init__()

        self.W = nn.Parameter(torch.randn(D, C) * 0.01)
        self.b = nn.Parameter(torch.zeros(C))

    def forward(self, X):
        """
        Parameters:
            X - data

        Returns: class probabilities
        """

        scores = torch.mm(X, self.W) + self.b
        probs = torch.softmax(scores, dim=1)
        return probs

    def get_loss(self, X, Yoh_, reg_coe=0.0):
        """
        Parameters:
            X - data
            Yoh_ - one-hot encoded true classes
            reg_coe - regularization coefficient

        Returns: loss
        """
        
        probs = self.forward(X)
        log_probs = torch.log(probs + 1e-12)
        cross_entropy = -torch.sum(Yoh_ * log_probs) / X.shape[0]
        reg = reg_coe * torch.sum(self.W ** 2)

        return cross_entropy + reg


def train(model, X, Yoh_, reg_coe=0.0, lr=0.01, steps=10000, print_step=500):
    """
    Parameters:
        model
        X - data
        Yoh_ - one-hot encoded true classes
        reg_coe - regularization coefficient
        steps - iterations
        lr - learning rate
        print_step - print loss every # of steps; 0 for no print
    """
    
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for i in range(steps):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_, reg_coe)
        loss.backward()
        optimizer.step()

        if print_step != 0 and i % print_step == 0:
            print(f"step: {i}, loss: {loss}")
        
    if print_step != 0:
        print(f"FINAL: loss: {loss}")


def eval(model, X, Y=None):
    """
    Parameters:
        model - type: PTLogreg
        X - data, type: np.array
        Y - optional to get loss print
        
    Returns: predicted class probabilites, type: np.array
    """

    X_torch = torch.tensor(X, dtype=torch.float32)
    probs = model.forward(X_torch)

    if Y is not None:
        Yoh_ = one_hot(Y)
        Yoh_torch = torch.tensor(Yoh_, dtype=torch.float32)
        log_probs = torch.log(probs + 1e-12)
        loss = -torch.sum(Yoh_torch * log_probs) / X_torch.shape[0]
        print(f"loss: {loss.item()}")

    return probs.detach().numpy()

def one_hot(Y):
    """
    Parameters:
        Y - classes
        
    Returns: one-hot encoded classes
    """
    n_classes = np.max(Y) + 1
    Yoh = np.zeros((len(Y), n_classes))
    Yoh[np.arange(len(Y)), Y] = 1
    return Yoh

if __name__=="__main__":
    np.random.seed(100)

    data_parts = data.sample_gmm_2d(3, 3, 100, 2)

    X, Y_ = data_parts[0]
    Yoh_ = one_hot(Y_)
    X_torch = torch.tensor(X, dtype=torch.float32)
    Yoh_torch = torch.tensor(Yoh_, dtype=torch.float32)
    rect = (np.min(X, axis=0), np.max(X, axis=0))

    X_test, Y_test = data_parts[1]
    Yoh_test = one_hot(Y_test)
    rect_test = (np.min(X_test, axis=0), np.max(X_test, axis=0))

    reg_coe = 0.00001

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    # Logreg train
    print("\nLogreg train:")
    logreg_m = logreg.Logreg()
    logreg_m.train(X, Y_, 0.01, 1000, 1000)
    probs = logreg_m.classify(X)
    Y = np.argmax(probs, axis=1)

    plt.sca(axs[0])
    data.graph_surface(logreg.classify_wrapper(logreg_m), rect, 0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.title("Logreg train")

    # PTLogreg train
    print("\nPTLogreg train:")
    ptlogreg_m = PTLogreg(X.shape[1], Yoh_.shape[1])
    train(ptlogreg_m, X_torch, Yoh_torch, 0, 0.01, 1000, 1000)
    probs = eval(ptlogreg_m, X)
    Y = np.argmax(probs, axis=1)

    def predict1(X):
        return np.argmax(eval(ptlogreg_m, X), axis=1)

    plt.sca(axs[1])
    data.graph_surface(predict1, rect, 0.5)
    data.graph_data(X, Y_, Y)
    plt.title("PTLogreg train")

    # PTLogreg reg train
    print("\nPTLogreg reg train:")
    ptlogreg_reg_m = PTLogreg(X.shape[1], Yoh_.shape[1])
    train(ptlogreg_reg_m, X_torch, Yoh_torch, reg_coe, 0.01, 1000, 1000)
    probs = eval(ptlogreg_reg_m, X)
    Y = np.argmax(probs, axis=1)

    def predict2(X):
        return np.argmax(eval(ptlogreg_reg_m, X), axis=1)

    plt.sca(axs[2])
    data.graph_surface(predict2, rect, 0.5)
    data.graph_data(X, Y_, Y)
    plt.title("PTLogreg reg train")

    # PTLogreg test
    print("\nPTLogreg test:")
    probs = eval(ptlogreg_m, X_test, Y_test)
    Y_test_pred = np.argmax(probs, axis=1)

    def predict3(X_test):
        return np.argmax(eval(ptlogreg_m, X_test), axis=1)

    plt.sca(axs[3])
    data.graph_surface(predict3, rect_test, 0.5)
    data.graph_data(X_test, Y_test, Y_test_pred)
    plt.title("PTLogreg test")

    # PTLogreg reg test
    print("\nPTLogreg reg test:")
    probs = eval(ptlogreg_reg_m, X_test, Y_test)
    Y_test_pred = np.argmax(probs, axis=1)

    def predict4(X_test):
        return np.argmax(eval(ptlogreg_reg_m, X_test), axis=1)

    plt.sca(axs[4])
    data.graph_surface(predict4, rect_test, 0.5)
    data.graph_data(X_test, Y_test, Y_test_pred)
    plt.title("PTLogreg reg test")

    plt.tight_layout()
    plt.show()
