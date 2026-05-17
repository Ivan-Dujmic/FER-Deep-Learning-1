import Loader
import torch
from sklearn.metrics import f1_score, confusion_matrix


class MeanPoolingModel(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.fc1 = torch.nn.Linear(300, 150)
        self.fc2 = torch.nn.Linear(150, 150)
        self.fc3 = torch.nn.Linear(150, 1)
    
    def forward(self, x, lengths):
        h = self.embedding(x)
        h = h.sum(dim=1) / lengths.unsqueeze(1).float()
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        h = torch.relu(h)
        h = self.fc3(h)
        return h.squeeze(1)


def train(model, data, optimizer, criterion, clip):
    model.train()

    loss_total = 0.0

    for batch in data:
        x, y, lengths = batch
        model.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss_total += loss.item()
    
    return loss_total / len(data)


def evaluate(model, data, criterion):
    model.eval()
    loss_total = 0.0

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in data:
            x, y, lengths = batch
            logits = model(x, lengths)
            loss = criterion(logits, y.float())
            loss_total += loss.item()
            preds = (logits > 0).long()
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(y.cpu().tolist())

    avg_loss = loss_total / len(data)
    acc = (torch.tensor(preds_all) == torch.tensor(labels_all)).float().mean().item()
    f1 = f1_score(labels_all, preds_all, zero_division=0)
    conf = confusion_matrix(labels_all, preds_all)
    return avg_loss, acc, f1, conf


if __name__ == "__main__":
    batch_sizes = {"train": 10, "valid": 32, "test": 32}
    lr = 1e-4
    epochs = 5
    clip = 1.0

    train_dataset = Loader.load_instances('datasets/sst_train_raw.csv')
    valid_dataset = Loader.load_instances('datasets/sst_valid_raw.csv')
    test_dataset = Loader.load_instances('datasets/sst_test_raw.csv')

    vocab_text = Loader.Vocab(token for dataset in [train_dataset] for instance in dataset for token in instance.text)
    vocab_label = Loader.Vocab([instance.label for dataset in [train_dataset] for instance in dataset], uses_specials=False)

    nlp_dataset_train = Loader.NLPDataset(train_dataset, vocab_text, vocab_label)
    nlp_dataset_valid = Loader.NLPDataset(valid_dataset, vocab_text, vocab_label)
    nlp_dataset_test = Loader.NLPDataset(test_dataset, vocab_text, vocab_label)

    vectors = Loader.load_vectors('datasets/sst_glove_6b_300d.txt')
    embeddings = Loader.build_embedding_matrix(vocab_text, vectors)

    pad_idx = vocab_text.stoi['<PAD>']

    train_loader = torch.utils.data.DataLoader(
        nlp_dataset_train,
        batch_size=batch_sizes['train'],
        shuffle=True,
        collate_fn=lambda batch: Loader.collate_fn(batch, pad_index=pad_idx)
    )

    valid_loader = torch.utils.data.DataLoader(
        nlp_dataset_valid,
        batch_size=batch_sizes['valid'],
        shuffle=False,
        collate_fn=lambda batch: Loader.collate_fn(batch, pad_index=pad_idx)
    )

    test_loader = torch.utils.data.DataLoader(
        nlp_dataset_test,
        batch_size=batch_sizes['test'],
        shuffle=False,
        collate_fn=lambda batch: Loader.collate_fn(batch, pad_index=pad_idx)
    )

    model = MeanPoolingModel(embeddings)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, valid_acc, valid_f1, valid_conf = evaluate(model, valid_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f} - Valid F1: {valid_f1:.4f}")

    test_loss, test_acc, test_f1, test_conf = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print(f"Test Confusion Matrix:\n{test_conf}")
