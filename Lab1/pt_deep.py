import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt
from pt_logreg import train
from pt_logreg import one_hot, eval

class PTDeep(nn.Module):
    def __init__(self, config, activation=torch.relu):
        """
        Parameters:
            config - [input_dim, hidden1, ..., hiddenN, num_classes]
            activation - activation function for hidden layers
        """

        super().__init__()

        self.n_layers = len(config) - 1
        self.activation = activation

        self.W = nn.ParameterList()
        self.b = nn.ParameterList()

        for i in range(self.n_layers):
            W = nn.Parameter(torch.empty(config[i], config[i+1]))
            nn.init.xavier_uniform_(W)
            b = nn.Parameter(torch.zeros(config[i+1]))
            self.W.append(W)
            self.b.append(b)

    def forward(self, X):
        """
        Parameters:
            X - data

        Returns: class probabilities
        """

        h = X
        for i in range(self.n_layers - 1):
            h = self.activation(torch.mm(h, self.W[i]) + self.b[i])
        scores = torch.mm(h, self.W[-1]) + self.b[-1]
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
        reg = reg_coe * sum(torch.sum(W ** 2) for W in self.W)

        return cross_entropy + reg
    
    def count_params(self):
        """
        Returns: # of parameters
        """
        count = 0
        for name, param in self.named_parameters():
            print(f"{name}: {tuple(param.shape)}")
            count += param.numel()
        print(f"count: {count}")
        return count
    
if __name__=="__main__":
    model0 = PTDeep([2, 5, 3])
    model0.count_params()

    data_params = [(4, 2, 40), (6, 2, 10)]
    model_params = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
    acts = [torch.relu, torch.sigmoid]

    fig, axs = plt.subplots(3, 4, figsize=(15, 15))
    axs = axs.flatten()
    index = 0

    np.random.seed(43)
    torch.manual_seed(43)

    for data_param in data_params:
        X, Y_ = data.sample_gmm_2d(*data_param)
        X_torch = torch.tensor(X, dtype=torch.float32)
        Yoh_ = one_hot(Y_)
        Yoh_torch = torch.tensor(Yoh_, dtype=torch.float32)
        rect = (np.min(X, axis=0), np.max(X, axis=0))

        for model_param in model_params:
            for act in acts:
                print(f"\ndata: {data_param}, layers: {model_param}, act: {act.__name__}")
                model = PTDeep(model_param, act)
                train(model, X_torch, Yoh_torch, 0.0001, 0.1, 10000, 10000)

                probs = eval(model, X)
                Y = np.argmax(probs, axis=1)
                def predict(X):
                    return np.argmax(eval(model, X), axis=1)
                
                accuracy, conf, precisions, recalls = data.eval_perf_multi(Y, Y_)
                print(f"accuracy: {accuracy}")
                print(f"confusion matrix:\n{conf}")
                print(f"precisions: {precisions}")
                print(f"recalls: {recalls}")
                
                plt.sca(axs[index])
                index += 1
                data.graph_surface(predict, rect, 0.5)
                data.graph_data(X, Y_, Y)
                plt.title(f"{data_param} - {model_param} - {act.__name__}")

    plt.tight_layout()
    plt.show()
        