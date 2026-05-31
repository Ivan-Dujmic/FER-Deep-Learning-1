import time
import torch
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from identity_model import IdentityModel
from utils2 import evaluate, compute_representations


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_test)} validation images!")

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 28 * 28
    model = IdentityModel().to(device)

    t0 = time.time_ns()

    representations = compute_representations(model, traineval_loader, num_classes, emb_size, device)
    print("Evaluating on test set...")
    acc = evaluate(model, representations, test_loader, device)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    t1 = time.time_ns()
    print(f"Time (sec): {(t1-t0)/10**9:.1f}")