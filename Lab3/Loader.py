from torch.utils.data import DataLoader, Dataset
import torch
from dataclasses import dataclass


class Vocab:
    def __init__(self, tokens, min_freq=1, max_size=-1, uses_specials=True):
        if uses_specials:
            self.size = 2
            self.itos = {0: '<PAD>', 1: '<UNK>'}
            self.stoi = {'<PAD>': 0, '<UNK>': 1}
        else:
            self.size = 0
            self.itos = {}
            self.stoi = {}
        self.freqs = {}

        self.min_freq = min_freq
        self.max_size = max_size
        if self.max_size != -1 and self.max_size < self.size:
            self.max_size = self.size

        for token in tokens:
            if token in self.freqs:
                self.freqs[token] += 1
            else:
                self.freqs[token] = 1

        sorted_tokens = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        for token, freq in sorted_tokens:
            if freq < self.min_freq or (self.max_size != -1 and self.size >= self.max_size):
                break
            self.itos[self.size] = token
            self.stoi[token] = self.size
            self.size += 1
    
    def encode(self, tokens):
        # Can't use .get(token, '<UNK>') because uses_special=False does not have <UNK> token and would raise KeyError
        return [self.stoi[token] if token in self.stoi else self.stoi.get('<UNK>', -1) for token in tokens]
    

def load_vectors(file_path):
    vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            token = parts[0]
            vector = list(map(float, parts[1:]))
            vectors[token] = vector
    return vectors


def build_embedding_matrix(vocab, vectors=None, embedding_dim=None):
    if vectors is not None:
        embedding_dim = len(next(iter(vectors.values())))
    elif embedding_dim is None:
        raise ValueError("Either vectors or embedding_dim must be provided")

    embedding_matrix = torch.randn(vocab.size, embedding_dim)
    embedding_matrix[vocab.stoi['<PAD>']] = torch.zeros(embedding_dim)

    if vectors is not None:
        for idx, token in vocab.itos.items():
            if token in vectors:
                embedding_matrix[idx] = torch.tensor(vectors[token])
    
    return torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=(vectors is not None), padding_idx=vocab.stoi['<PAD>'])


@dataclass
class Instance:
    text: list[str]
    label: str


def load_instances(file_path, separator=', '):
    instances = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(separator)
            if len(parts) != 2:
                continue
            text, label = parts
            instance = Instance(text.split(), label)
            instances.append(instance)    
    return instances


class NLPDataset(Dataset):
    def __init__(self, instances, vocab_text, vocab_label):
        self.instances = instances
        self.vocab_text = vocab_text
        self.vocab_label = vocab_label
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        encoded_text = self.vocab_text.encode(instance.text)
        encoded_label = self.vocab_label.encode([instance.label])[0]
        return torch.tensor(encoded_text, dtype=torch.long), torch.tensor(encoded_label, dtype=torch.long)


def collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index)
    return padded_texts, torch.tensor(labels), torch.tensor(lengths, dtype=torch.long)


if __name__ == "__main__":
    instances = load_instances("datasets/sst_train_raw.csv")
    texts = [instance.text for instance in instances]
    labels = [instance.label for instance in instances]
    text_tokens = [
        token
        for instance in instances
        for token in instance.text
    ]
    vocab_text = Vocab(text_tokens)
    vocab_label = Vocab(labels, uses_specials=False)

    print("freqs:")
    for token in ['the', 'a', 'and', 'of', 'to']:
        print(f"{token}: {vocab_text.freqs.get(token, 0)}")

    print("\nvocab_text stoi:")
    for token in ['<PAD>', '<UNK>', 'the', 'a', 'and', 'my', 'twists', 'lets', 'sports', 'amateurishly']:
        print(f"{token}: {vocab_text.stoi.get(token, vocab_text.stoi.get('<UNK>', -1))}")

    print("\nvocab_label stoi:")
    for token, index in vocab_label.stoi.items():
        print(f"{token}: {index}")
    
    print("\nencode:")
    print(texts[3])
    print(vocab_text.encode(texts[3]))
    print(labels[3])
    print(vocab_label.encode([labels[3]]))

    print("\nsize of vocab_text:", vocab_text.size)

    print("\nencode through NLPDataset:")
    dataset = NLPDataset(instances, vocab_text, vocab_label)
    instance_text, instance_label = dataset.instances[3].text, dataset.instances[3].label
    print(instance_text)
    print(instance_label)
    encoded_text, encoded_label = dataset[3]
    print(encoded_text)
    print(encoded_label)

    train_dataset = NLPDataset(instances, vocab_text, vocab_label)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_index=vocab_text.stoi['<PAD>'])
    )
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")