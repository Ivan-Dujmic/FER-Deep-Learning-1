import torch
import os
import Loader
import Trainer
import numpy as np

class MeanPoolingModelHyper(torch.nn.Module):
    def __init__(self, embedding, hidden_size=150, dropout=0.0, activation='relu'):
        super().__init__()
        self.embedding = embedding
        self.fc1 = torch.nn.Linear(300, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)
    
        self.dropout = torch.nn.Dropout(dropout)
        self.act = torch.relu if activation == 'relu' else torch.tanh

    def forward(self, x, lengths):
        h = self.embedding(x)
        h = h.sum(dim=1) / lengths.unsqueeze(1).float()
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.act(h)
        h = self.fc3(h)
        return h.squeeze(1)


class RNNHyper(torch.nn.Module):
    def __init__(self, embedding, rnn_type='GRU', hidden_size=150, num_layers=2, dropout=0.0, bidirectional=False):
        super().__init__()
        self.embedding = embedding
        rnn_type = rnn_type.upper()
        if rnn_type not in ['RNN', 'GRU', 'LSTM']:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        rnn_class = getattr(torch.nn, rnn_type)

        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = rnn_class(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=False,
            bidirectional=bidirectional,
        )

        self.directional_factor = 2 if bidirectional else 1

        self.fc1 = torch.nn.Linear(hidden_size * self.directional_factor, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, lengths):
        h = self.embedding(x)
        h = h.transpose(0, 1)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=False, enforce_sorted=False)
        _, h = self.rnn(h)
        if isinstance(h, tuple): # LSTM returns (h_n, c_n)
            h = h[0]

        if self.directional_factor == 2:
            hidden = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        else:
            hidden = h[-1]

        h = self.dropout(hidden)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)

        return h.squeeze(1)


def run_experiment(model_type, config, train_dataset, valid_dataset, test_dataset, vectors, epochs=5):
    vocab_text = Loader.Vocab(
        (token for instance in train_dataset for token in instance.text),
        min_freq=config['min_freq']
    )
    vocab_label = Loader.Vocab([instance.label for instance in train_dataset], uses_specials=False)

    nlp_train = Loader.NLPDataset(train_dataset, vocab_text, vocab_label)
    nlp_valid = Loader.NLPDataset(valid_dataset, vocab_text, vocab_label)
    nlp_test = Loader.NLPDataset(test_dataset, vocab_text, vocab_label)

    pad_idx = vocab_text.stoi['<PAD>']

    if config['use_pretrained']:
        embeddings = Loader.build_embedding_matrix(vocab_text, vectors)
    else:
        embeddings = Loader.build_embedding_matrix(vocab_text, None, 300)

    train_loader = torch.utils.data.DataLoader(
        nlp_train, batch_size=config['batch_size'], shuffle=True,
        collate_fn=lambda b: Loader.collate_fn(b, pad_index=pad_idx)
    )
    valid_loader = torch.utils.data.DataLoader(
        nlp_valid, batch_size=32, shuffle=False,
        collate_fn=lambda b: Loader.collate_fn(b, pad_index=pad_idx)
    )
    test_loader = torch.utils.data.DataLoader(
        nlp_test, batch_size=32, shuffle=False,
        collate_fn=lambda b: Loader.collate_fn(b, pad_index=pad_idx)
    )

    if model_type == 'baseline':
        model = MeanPoolingModelHyper(embeddings, hidden_size=config['hidden_size'], dropout=config['dropout'])
    else:
        model = RNNHyper(embeddings, rnn_type='GRU', hidden_size=config['hidden_size'], dropout=config['dropout'])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_valid_f1 = -1.0
    best_metrics = {}

    for epoch in range(epochs):
        _ = Trainer.train(model, train_loader, optimizer, criterion, config['clip'])
        _, _, valid_f1, _ = Trainer.evaluate(model, valid_loader, criterion)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            test_loss, test_acc, test_f1, test_conf = Trainer.evaluate(model, test_loader, criterion)
            best_metrics = {'loss': test_loss, 'accuracy': test_acc, 'f1': test_f1, 'conf': test_conf}

    return best_metrics


if __name__ == "__main__":
    out_path = os.path.join('results', 'rnn2.txt')

    train_dataset = Loader.load_instances('datasets/sst_train_raw.csv')
    valid_dataset = Loader.load_instances('datasets/sst_valid_raw.csv')
    test_dataset = Loader.load_instances('datasets/sst_test_raw.csv')
    vectors = Loader.load_vectors('datasets/sst_glove_6b_300d.txt')

    opt_min_freq = [1, 3, 5]
    opt_lr = [1e-5, 1e-4, 1e-3]
    opt_batch_size = [10, 32, 64]
    opt_dropout = [0.0, 0.3, 0.6]
    opt_hidden_size = [50, 150, 300]
    opt_clip = [0.1, 0.25, 1.0]
    # opt_use_pretained = [True, False]

    base_config = {
        'min_freq': 1,
        'lr': 1e-4,
        'batch_size': 32,
        'dropout': 0.3,
        'hidden_size': 300,
        'clip': 0.25,
        'use_pretrained': True
    }

    experiments = []
    
    def add_experiment(param_name, values):
        for val in values:
            cfg = base_config.copy()
            cfg[param_name] = val
            experiments.append((param_name, cfg))

    add_experiment('min_freq', opt_min_freq)
    add_experiment('lr', opt_lr)
    add_experiment('batch_size', opt_batch_size)
    add_experiment('dropout', opt_dropout)
    add_experiment('hidden_size', opt_hidden_size)
    add_experiment('clip', opt_clip)

    with open(out_path, 'a') as f:
        f.write("### PRE-TRAINED EMBEDDINGS VS RANDOM ###\n")
        f.flush()
        for m_type in ['baseline', 'gru']:
            for use_pre in [True, False]:
                cfg = base_config.copy()
                cfg['use_pretrained'] = use_pre
                res = run_experiment(m_type, cfg, train_dataset, valid_dataset, test_dataset, vectors)
                f.write(f"Model: {m_type} | Pretrained: {use_pre} --> Test Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}\n")
                f.flush()
        f.write("\n")
        f.flush()

        f.write("### HYPERPARAMETER SEARCH ###\n")
        f.flush()
        for m_type in ['baseline', 'gru']:
            f.write(f"\n# Model Type: {m_type}\n")
            f.flush()
            for param, cfg in experiments:
                res = run_experiment(m_type, cfg, train_dataset, valid_dataset, test_dataset, vectors)
                f.write(f"Varying {param}={cfg[param]} --> Loss: {res['loss']:.4f} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f}\n")
                f.flush()
        f.write("\n")
        f.flush()

        best_gru_config = {
            'min_freq': 3,
            'lr': 1e-3,
            'batch_size': 32,
            'dropout': 0.3,
            'hidden_size': 150,
            'clip': 0.1,
            'use_pretrained': True
        }
        best_base_config = {
            'min_freq': 1,
            'lr': 1e-3,
            'batch_size': 10,
            'dropout': 0.3,
            'hidden_size': 300,
            'clip': 1.0,
            'use_pretrained': True
        }

        f.write("### BEST CONFIGURATIONS RUNS ###\n")
        for m_type, best_cfg in [('baseline', best_base_config), ('gru', best_gru_config)]:
            accs, f1s = [], []
            for run in range(5):
                res = run_experiment(m_type, best_cfg, train_dataset, valid_dataset, test_dataset, vectors)
                accs.append(res['accuracy'])
                f1s.append(res['f1'])
            
            f.write(f"Model: {m_type} over 5 runs:\n")
            f.flush()
            f.write(f"  Accuracy: Mean = {np.mean(accs):.4f}, Std = {np.std(accs):.4f}\n")
            f.flush()
            f.write(f"  F1-Score: Mean = {np.mean(f1s):.4f}, Std = {np.std(f1s):.4f}\n")
            f.flush()