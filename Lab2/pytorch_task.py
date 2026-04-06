from pathlib import Path
import os
from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split, DataLoader
import time
import torch.nn as nn
import math
import numpy as np
import skimage.io
import torch.optim as optim
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / 'datasets'
SAVE_DIR = Path(__file__).parent / 'out'
os.makedirs(name=SAVE_DIR, exist_ok=True)

config = {
    'max_epochs': 8,
    'batch_size': 50,
    'weight-decay': 1e-1,
    'lr_policy': {1: 1e-1, 3: 1e-2, 5: 1e-3, 7: 1e-4}
}

torch.manual_seed(int(time.time() * 1e6) % 2 ** 31)

# ds = datasets.MNIST(DATA_DIR, train=True, download=True)
# train_tensor = ds.data.float() / 255.0
# mean = train_tensor.mean()
# std = train_tensor.std()
# print(f"mean:{mean}, std:{std}")
ds_mean = 0.13066047430038452
ds_std = 0.30810782313346863
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(ds_mean,), std=(ds_std,))
])

ds_train_valid = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
ds_test = datasets.MNIST(root=DATA_DIR, train=False, transform=transform)
ds_train, ds_valid = random_split(dataset=ds_train_valid, lengths=[55000, 5000])

loader_train = DataLoader(dataset=ds_train, batch_size=config['batch_size'], shuffle=True)
loader_valid = DataLoader(dataset=ds_valid, batch_size=config['batch_size'], shuffle=False)
loader_test = DataLoader(dataset=ds_test, batch_size=config['batch_size'], shuffle=False)

class ConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=512, bias=True)
        self.logits = nn.Linear(in_features=512, out_features=10, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.logits:
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            self.logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = torch.relu(h)

        h = self.conv2(h)
        h = self.pool2(h)
        h = torch.relu(h)

        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        h = torch.relu(h)

        h = self.logits(h)
        return h
    
def draw_conv1_kernels(epoch, step, kernels, save_dir):
    kernels_ = kernels.copy()
    count = kernels_.shape[0]
    channels = kernels_.shape[1]
    dims = kernels_.shape[2]

    kernels_ -= kernels_.min()
    kernels_ /= kernels_.max()

    border = 1
    cols = 8
    rows = math.ceil(count / cols)
    width = cols * (dims + border) - border
    height = rows * (dims + border) - border

    img = np.zeros([height, width])
    for i in range(count):
        r = int(i / cols) * (dims + border)
        c = int(i % cols) * (dims + border)
        img[r:r+dims, c:c+dims] = kernels_[i, 0]

    img = (img * 255).astype(np.uint8)
    filename = f'conv1_epoch_{epoch:02d}_step_{step:06d}.png'
    skimage.io.imsave(os.path.join(save_dir, filename), img)

def evaluate(name, loader, model, loss_func):
    model.eval()
    correct = 0
    total = 0
    loss = 0

    print('\nRunning evaluation: ', name)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss += loss_func(logits, y).item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = (correct / total) * 100
    loss /= len(loader)
    print(f'{name} accuracy = {acc:.2f}')
    print(f'{name} avg loss = {loss:.2f}\n')

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = ConvolutionModel().to(device)
    loss_func = nn.CrossEntropyLoss()
    
    params_reg = [model.conv1.weight, model.conv2.weight, model.fc.weight]
    params_other = [p for p in model.parameters() if not any(p is reg_p for reg_p in params_reg)]

    optimizer = optim.SGD([
        {'params': params_reg, 'weight_decay': config['weight-decay']},
        {'params': params_other, 'weight_decay': 0.0}
    ], lr=config['lr_policy'][1])

    kernels = model.conv1.weight.detach().cpu().numpy()
    draw_conv1_kernels(1, 0, kernels, SAVE_DIR)

    epoch_losses = []
    for epoch in range(1, config['max_epochs'] + 1):
        if epoch in config['lr_policy']:
            lr = config['lr_policy'][epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        epoch_loss = 0.0

        for i, (x, y) in enumerate(loader_train):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss =  loss_func(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i * config['batch_size']}, batch loss = {loss.item():.4f}")

        epoch_loss /= len(loader_train)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch} finished | Avg loss: {epoch_loss:.4f}")
        kernels = model.conv1.weight.detach().cpu().numpy()
        draw_conv1_kernels(epoch, (i + 1) * config['batch_size'], kernels, SAVE_DIR)
        evaluate("Validation", loader_valid, model, loss_func)

    evaluate("Test", loader_test, model, loss_func)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, config['max_epochs'] + 1), epoch_losses, marker='o', color='b', label='Training Loss')
    plt.title('Loss through training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.xticks(range(1, config['max_epochs'] + 1))
    plt.show()