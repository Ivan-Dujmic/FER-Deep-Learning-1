import torch
import torch.nn as nn


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.GroupNorm(num_groups=1, num_channels=num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(in_channels=num_maps_in, out_channels=num_maps_out, kernel_size=k, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32, margin=1.0):
        super().__init__()
        self.emb_size = emb_size
        self.margin = margin

        self.embedder = nn.Sequential(
            _BNReluConv(num_maps_in=input_channels, num_maps_out=emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=emb_size, num_maps_out=emb_size),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        x = self.embedder(img)
        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        d_ap = torch.norm(a_x - p_x, p=2, dim=1)
        d_np = torch.norm(a_x - n_x, p=2, dim=1)
        
        loss = torch.relu(d_ap - d_np + self.margin)
        return loss.mean()