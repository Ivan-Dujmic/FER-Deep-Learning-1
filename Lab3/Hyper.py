import torch
import os
import Loader
import Trainer
import RNN

def run_experiment(config, train_loader, valid_loader, test_loader, embeddings, epochs, lr, clip):
    model = RNN(
        embedding=embeddings,
        rnn_type=config['rnn_type'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_f1 = 0.0
    test_metrics = {}

    for epoch in range(epochs):
        _ = Trainer.train(model, train_loader, optimizer, criterion, clip)
        _, _, valid_f1, _ = Trainer.evaluate(model, valid_loader, criterion)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            test_loss, test_acc, test_f1, test_conf = Trainer.evaluate(model, test_loader, criterion)
            test_metrics = {
                'loss': test_loss,
                'accuracy': test_acc,
                'f1': test_f1,
                'confusion_matrix': test_conf
            }

    return test_metrics


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
    embeddings_random = Loader.build_embedding_matrix(vocab_text, None, 300)

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

    options_rnn_type = ['RNN', 'GRU', 'LSTM']
    options_hidden_size = [50, 150, 300]
    options_num_layers = [1, 2, 3]
    options_dropout = [0.0, 0.25, 0.5]
    options_bidirectional = [True, False]

    experiments = []
    mid = lambda opts: opts[1]

    for rnn_type in options_rnn_type:
        experiments.append({
            'rnn_type': rnn_type,
            'hidden_size': mid(options_hidden_size),
            'num_layers': mid(options_num_layers),
            'dropout': mid(options_dropout),
            'bidirectional': mid(options_bidirectional),
        })

    for hidden_size in options_hidden_size:
        experiments.append({
            'rnn_type': mid(options_rnn_type),
            'hidden_size': hidden_size,
            'num_layers': mid(options_num_layers),
            'dropout': mid(options_dropout),
            'bidirectional': mid(options_bidirectional),
        })

    for num_layers in options_num_layers:
        experiments.append({
            'rnn_type': mid(options_rnn_type),
            'hidden_size': mid(options_hidden_size),
            'num_layers': num_layers,
            'dropout': mid(options_dropout),
            'bidirectional': mid(options_bidirectional),
        })

    for dropout in options_dropout:
        experiments.append({
            'rnn_type': mid(options_rnn_type),
            'hidden_size': mid(options_hidden_size),
            'num_layers': mid(options_num_layers),
            'dropout': dropout,
            'bidirectional': mid(options_bidirectional),
        })

    for bidirectional in options_bidirectional:
        experiments.append({
            'rnn_type': mid(options_rnn_type),
            'hidden_size': mid(options_hidden_size),
            'num_layers': mid(options_num_layers),
            'dropout': mid(options_dropout),
            'bidirectional': bidirectional,
        })

    out_path = os.path.join('results', 'rnn1.txt')
    with open(out_path, 'a') as f:
        for config in experiments:
            metrics = run_experiment(config, train_loader, valid_loader, test_loader, embeddings, epochs, lr, clip)
            header = f"{config['rnn_type']} ; {config['hidden_size']} ; {config['num_layers']} ; {config['dropout']} ; {config['bidirectional']}\n"
            loss = metrics['loss']
            acc = metrics['accuracy']
            f1 = metrics['f1']
            conf = metrics['confusion_matrix']
            f.write(header)
            f.write(f"{loss:.5f} ; {acc:.5f} ; {f1:.5f} ; [ {str(conf[0][0]).rjust(3)} {str(conf[0][1]).rjust(3)} {str(conf[1][0]).rjust(3)} {str(conf[1][1]).rjust(3)} ]\n\n")
            f.flush()