import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, k, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size

        self.embedder = nn.Sequential(
            _BNReluConv(input_channels, emb_size),
            nn.MaxPool2d(3, 2),
            _BNReluConv(1 + torch.floor((emb_size - 3) / 2)),
            nn.MaxPool2d(3, 2),
            _BNReluConv(1 + torch.floor((1 + torch.floor((emb_size - 3) / 2) - 3) / 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # TODO: remove ?
        )

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        x = self.embedder(img)
        return x

    def loss(self, anchor, positive, negative, margin):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        
        loss = torch.relu(torch.maximum(torch.cdist(a_x, p_x), 0) - torch.maximum(torch.cdist(n_x, p_x), 0) + margin)
        return loss