import torch
import Loader
import Trainer

class RNN(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.rnn = torch.nn.RNN(
            input_size=300,
            hidden_size=150,
            num_layers=2,
            batch_first=False,
            bidirectional=False
        )
        self.fc1 = torch.nn.Linear(150, 150)
        self.fc2 = torch.nn.Linear(150, 1)

    def forward(self, x, lengths):
        h = self.embedding(x)
        h = h.transpose(0, 1)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=False, enforce_sorted=False)
        out, h = self.rnn(h)
        hidden = h[-1]
        h = self.fc1(hidden)
        h = torch.relu(h)
        h = self.fc2(h)

        return h.squeeze(1)
    

if __name__ == "__main__":
    batch_sizes = {"train": 10, "valid": 32, "test": 32}
    lr = 1e-4
    epochs = 5
    clip = 0.25

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

    model = RNN(embeddings)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = Trainer.train(model, train_loader, optimizer, criterion, clip)
        valid_loss, valid_acc, valid_f1, valid_conf = Trainer.evaluate(model, valid_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f} - Valid F1: {valid_f1:.4f}")

    test_loss, test_acc, test_f1, test_conf = Trainer.evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print(f"Test Confusion Matrix:\n{test_conf}")