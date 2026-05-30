import torch
from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train', remove_classes=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        self.remove_classes = remove_classes or []
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if self.remove_classes:
            mask = torch.ones_like(self.targets, dtype=torch.bool)
            for c in self.remove_classes:
                mask &= self.targets != c
            self.images = self.images[mask]
            self.targets = self.targets[mask]
            self.classes = [c for c in self.classes if c not in self.remove_classes]

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        anchor_class = self.targets[index].item()
        r = choice([c for c in self.classes if c != anchor_class])
        return choice(self.target2indices[r])

    def _sample_positive(self, index):
        anchor_class = self.targets[index].item()
        positive = choice(self.target2indices[anchor_class])

        while positive == index: # Until we choose something different from index
            positive = choice(self.target2indices[anchor_class])

        return positive

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)