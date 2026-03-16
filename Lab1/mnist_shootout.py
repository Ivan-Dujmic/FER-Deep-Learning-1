import copy
import torch
import torchvision
import matplotlib.pyplot as plt
import pt_deep
from pt_logreg import train
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

experiments = []
experiments.append("regularization")
experiments.append("config")
experiments.append("adam")
experiments.append("scheduler")
experiments.append("svm")

def train_mb(model, optimizer, x, y, x_val, y_val, reg_coe, epochs, batch_size, scheduler=None):
    n_samples = x.shape[0]

    best_loss = float('inf')
    best_model = None

    losses_train = []
    losses_val = []

    for epoch in range(epochs):
        shuffle = torch.randperm(n_samples)
        x = x[shuffle]
        y = y[shuffle]

        for i in range(0, n_samples, batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            optimizer.zero_grad()
            loss = model.get_loss(x_batch, y_batch, reg_coe)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_train = model.get_loss(x, y, reg_coe)
            loss_val = model.get_loss(x_val, y_val, reg_coe)

        losses_train.append(loss_train.item())
        losses_val.append(loss_val.item())

        print(f"epoch: {epoch}, loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}")

        if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)

        if scheduler:
            scheduler.step()

    return best_model, losses_train, losses_val
    
def eval(model, x, y):
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

    # y_true = torch.argmax(y, dim=1)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='macro', zero_division=0)
    rec = recall_score(y, preds, average='macro', zero_division=0)

    return acc, prec, rec

dataset_root = '/tmp/mnist'
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)
x_train, x_test = x_train.view(x_train.shape[0], -1), x_test.view(x_test.shape[0], -1)
yoh_train, yoh_test = F.one_hot(y_train), F.one_hot(y_test)

N = x_train.shape[0]
D = x_train.shape[1]
C = y_train.max().add_(1).item()

shuffle = torch.randperm(N)
split = int(0.8 * N)
train_idx = shuffle[:split]
val_idx = shuffle[split:]

x_tr = x_train[train_idx]
y_tr = yoh_train[train_idx]
x_val = x_train[val_idx]
y_val = yoh_train[val_idx]

configs = [
    [784, 10],
    [784, 100, 10],
    [784, 100, 100, 10],
    [784, 100, 100, 100, 10]
]

def test(optimizer_class, lr, reg_coe, epochs, batch_size, scheduler_class=None):
    results = {}

    print()
    for config in configs:
        model = pt_deep.PTDeep(config)
        optimizer = optimizer_class(model.parameters(), lr=lr)
        scheduler = None
        if scheduler_class:
            scheduler = scheduler_class(optimizer, gamma=1 - 1e-4)
        model, losses_train, losses_val = train_mb(model, optimizer, x_tr, y_tr, x_val, y_val, reg_coe, epochs, batch_size, scheduler)
        results[str(config)] = (losses_train, losses_val)

        acc, prec, rec = eval(model, x_test, y_test)
        print(f"config {config}:\nacc: {acc:.4f}\nprec: {prec:.4f}\nrec: {rec:.4f}\n")

    _, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes = axes.flatten()

    for i, config in enumerate(results):
        losses_train, losses_val = results[config]

        ax = axes[i]

        ax.plot(losses_train, label="train")
        ax.plot(losses_val, label="val")

        ax.set_title(config)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()

    plt.tight_layout()
    plt.show()

# EXPERIMENTS
if "regularization" in experiments:
    # reg_coefs = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    reg_coefs = [0.1]

    for reg_coe in reg_coefs:
        model = pt_deep.PTDeep([784, 10])
        train(model, x_train, yoh_train, reg_coe, lr=5, steps=50, print_step=0)
        
        W = model.W[0].detach().cpu()

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))

        for i in range(10):
            w_digit = W[:, i].reshape(28, 28)
            ax = axes[i // 5, i % 5]
            ax.imshow(w_digit, cmap=plt.get_cmap('gray'))
            ax.set_title(f"Digit {i}")
            ax.axis("off")

        plt.suptitle(f"reg_coe = {reg_coe}")
        plt.tight_layout()
        plt.show()

if "config" in experiments:
    test(torch.optim.SGD, 0.05, 0, 30, 256)

if "adam" in experiments:
    test(torch.optim.Adam, 1e-4, 0, 30, 256)

if "scheduler" in experiments:
    test(torch.optim.Adam, 1e-4, 0, 30, 256, torch.optim.lr_scheduler.ExponentialLR)

if "svm" in experiments:
    models = [SVC(kernel='linear'), SVC(kernel='rbf')]

    for model in models:
        model.fit(x_train.numpy(), y_train.numpy())
        pred = model.predict(x_test.numpy())
        print(accuracy_score(y_test.numpy(), pred))

        # 0.9404
        # 0.9792