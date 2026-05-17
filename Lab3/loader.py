from torch.utils.data import DataLoader, Dataset

# Instance - wrapper for data (@dataclass)
# NLPDataset - inherits Dataset, implement __getitem__ - save and fetch data and operations such as vocab building
# Vocab - text to index (numericalization)


# Vocab
# Size? itos (index to string), stoi (string to index)
# Each input field should get a vocab
# Implement vocab based on a dictionary of woprd frequencies (key = token, value = frequency)
# Convention is to give lower indexes to more frequent tokens

# Special tokens:
# <PAD> - padding - index 0 -used to make all sequences in a batch the same length
# <UNK> - unknown - index 1

# Vocab should have encode (tokens -> numbers)
# Vocab should have max_size - max tokens in vocab (includes specials) where -1 is no limit
# Vocab should have min_freq - min freq for a token to be included in vocab where specials don't need to meet this requirement


# NLPDataset
# __getitem__ - enables class indexing - should return numerized text and label of referenced instance - on the fly is ok, no need for cacheing


# collate_fn(batch)
# Used to build a batch tensor for each input field in instance list
# Expects elements to be of same size so we'll have pad to size of longest instance, but we return the original size
    # Look at torhc.nn.utils.rnn.pad_sequence for padding sequences of different lengths