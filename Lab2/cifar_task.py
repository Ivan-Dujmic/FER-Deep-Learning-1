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
import torch.nn.functional as F

DATA_DIR = Path(__file__).parent / 'datasets'
SAVE_DIR = Path(__file__).parent / 'out'
os.makedirs(name=SAVE_DIR, exist_ok=True)

config = {
    'max_epochs': 8,
    'batch_size': 50,
    'weight-decay': 1e-1,
    'lr_start': 1e-1
}

torch.manual_seed(int(time.time() * 1e6) % 2 ** 31)

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
# train_tensor = torch.stack([img for img, _ in ds], dim=0)
# ds_mean = train_tensor.mean(dim=(0, 2, 3))
# ds_std = train_tensor.std(dim=(0, 2, 3))
# print(f"Mean: {ds_mean[0]:.27}, {ds_mean[1]:.27}, {ds_mean[2]:.27}")
# print(f"Std: {ds_std[0]:.27}, {ds_std[1]:.27}, {ds_std[2]:.27}")
ds_mean = (0.4914008080959320068359375, 0.4821589887142181396484375, 0.44653093814849853515625)
ds_std = (0.24703224003314971923828125, 0.24348513782024383544921875, 0.2615878582000732421875)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=ds_mean, std=ds_std)
])

ds_train_valid = datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
ds_test = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
ds_train, ds_valid = random_split(dataset=ds_train_valid, lengths=[45000, 5000])

loader_train = DataLoader(dataset=ds_train, batch_size=config['batch_size'], shuffle=True)
loader_valid = DataLoader(dataset=ds_valid, batch_size=config['batch_size'], shuffle=False)
loader_test = DataLoader(dataset=ds_test, batch_size=config['batch_size'], shuffle=False)

# conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)
class ConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.logits = nn.Linear(in_features=128, out_features=10, bias=True)

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
        h = torch.relu(h)
        h = self.pool1(h)
        h = self.conv2(h)
        h = torch.relu(h)
        h = self.pool2(h)
        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.logits(h)
        return h
    
def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    img = (img * 255).astype(np.uint8)
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    skimage.io.imsave(os.path.join(save_dir, filename), img)

def draw_image(img, mean, std):
    img = img.transpose(1, 2, 0)
    img *= std
    img += mean
    img = img * 255 
    img = np.clip(img, 0, 255)  
    img = img.astype(np.uint8)
    skimage.io.imshow(img)
    skimage.io.show()

def evaluate(name, loader, model, loss_func, optimizer, top20worst=False):
    model.eval()
    confusion = np.zeros((10, 10))
    loss = 0

    misses = []

    print('\nRunning evaluation: ', name)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss += loss_func(logits, y).item()
            pred = torch.argmax(logits, dim=1)
            for t, p in zip(y.view(-1), pred.view(-1)):
                confusion[t.long(), p.long()] += 1

            if top20worst:
                individual_losses = F.cross_entropy(logits, y, reduction='none')

                mask = pred != y

                for i in range(len(y)):
                    if mask[i]:
                        top3_probs, top3_idx = torch.topk(logits[i], 3)

                        misses.append({
                            'image': x[i].cpu().numpy(),
                            'true': classes[y[i].item()],
                            'loss': individual_losses[i].item(),
                            'top3_classes': [classes[j.item()] for j in top3_idx],
                            'top3_probs': top3_probs.cpu().numpy()
                        })

    correct = np.trace(confusion)
    total = confusion.sum()
    acc = (correct / total) * 100
    TP = np.diag(confusion)
    pred_pos = confusion.sum(axis=0)
    real_pos = confusion.sum(axis=1)
    prec = np.mean(np.nan_to_num(TP / pred_pos))
    rec = np.mean(np.nan_to_num(TP / real_pos))
    loss /= len(loader)
    lr = optimizer.param_groups[0]['lr']

    print(f'{name} accuracy = {acc:.2f}')
    print(f'{name} precision = {prec:.4f}')
    print(f'{name} recall = {rec:.4f}')
    print(f'{name} avg loss = {loss:.2f}')
    print(f'{name} learning rate = {lr:.4f}')
    print()

    if top20worst:
        misses.sort(key=lambda item: item['loss'], reverse=True)
        top20 = misses[:20]
        for image in top20:
            print(f"Image loss: {image['loss']:.4f}")
            print(f"True: {image['true']}")
            print(f"Top3: {image['top3_classes'][0]} {image['top3_classes'][1]} {image['top3_classes'][2]}")
            draw_image(image['image'], ds_mean, ds_std)

    return acc, prec, rec, loss, lr

def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['acc_train'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Accuracy')
    ax1.plot(x_data, data['acc_train'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['acc_valid'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='lower right', fontsize=legend_size)

    ax2.set_title('Precision')
    ax2.plot(x_data, data['prec_valid'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='lower right', fontsize=legend_size)

    ax3.set_title('Recall')
    ax3.plot(x_data, data['rec_valid'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax3.legend(loc='lower right', fontsize=legend_size)

    ax4.set_title('Loss')
    ax4.plot(x_data, data['loss_train'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax4.plot(x_data, data['loss_valid'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax4.legend(loc='upper right', fontsize=legend_size)

    ax5.set_title('Learning Rate')
    ax5.plot(x_data, data['lr'], marker='o', color='k', linewidth=linewidth, linestyle='-')

    save_path = os.path.join(save_dir, 'graph.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)

if __name__=="__main__":
    device = torch.device('cpu')
    print(f'Device: {device}')
    model = ConvolutionModel().to(device)
    loss_func = nn.CrossEntropyLoss()
    
    params_reg = [model.conv1.weight, model.conv2.weight, model.fc1.weight, model.fc2.weight]
    params_other = [p for p in model.parameters() if not any(p is reg_p for reg_p in params_reg)]

    optimizer = optim.SGD([
        {'params': params_reg, 'weight_decay': config['weight-decay']},
        {'params': params_other, 'weight_decay': 0.0}
    ], config['lr_start'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.7)

    draw_conv_filters(0, 0, model.conv1.weight.detach().numpy(), SAVE_DIR)

    results = {
        'acc_train': [],
        'acc_valid': [],
        'prec_valid': [],
        'rec_valid': [],
        'loss_train': [],
        'loss_valid': [],
        'lr': []
    }
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i, (x, y) in enumerate(loader_train):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i * config['batch_size']}, batch loss = {loss.item():.4f}")

        scheduler.step()

        epoch_loss /= len(loader_train)
        print(f"Epoch {epoch} finished | Avg loss: {epoch_loss:.4f}")
        
        draw_conv_filters(epoch, (i + 1) * config['batch_size'], model.conv1.weight.detach().numpy(), SAVE_DIR)
        acc, prec, rec, loss_avg, lr = evaluate("Validation", loader_valid, model, loss_func, optimizer)
        results['acc_train'].append((correct / total) * 100)
        results['acc_valid'].append(acc)
        results['prec_valid'].append(prec)
        results['rec_valid'].append(rec)
        results['loss_train'].append(epoch_loss)
        results['loss_valid'].append(loss_avg)
        results['lr'].append(lr)

    evaluate("Test", loader_test, model, loss_func, optimizer, True)

    plot_training_progress(SAVE_DIR, results)