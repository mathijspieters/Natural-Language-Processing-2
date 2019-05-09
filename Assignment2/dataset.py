import torch
import os
import numpy as np

from torch.utils import data
from nltk.tree import Tree


class Dataset(data.Dataset):
    def __init__(self, path):
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.UNK= '<UNK>'
        self._words = set([self.SOS, self.EOS, self.PAD, self.UNK])
        self.data = open(os.path.join(path, "02-21.10way.clean"), "r")
        self.data = self.process(self.data)
        self._word_2_idx = {}
        self._idx_2_word = {}
        self.create_vocabulary()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def process(self, data):
        tmp = []
        for i, d in enumerate(data.readlines()):
            t = Tree.fromstring(d)
            sentence = t.leaves()
            self._words |= set(sentence)
            tmp.append(sentence)
            if i > 100:
                break
        return tmp

    def create_vocabulary(self):
        self._word_2_idx = dict(zip(list(self._words), range(len(list(self._words)))))
        self._idx_2_word = {self._word_2_idx[word]:word for word in self._word_2_idx.keys()}

    def word_2_idx(self, word):
        return self._word_2_idx[word] if word in self._word_2_idx else self._word_2_idx[self.UNK]

    def idx_2_word(self, idx):
        return self._idx_2_word[idx] if idx in self._idx_2_word else self.UNK


class DataLoader:
    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self._data_size = len(self.dataset)
        self.permute()
        self.index = 0
        self.batch_size = batch_size

    def __getitem__(self, item):
        SOS = self.dataset.word_2_idx(self.dataset.SOS)
        EOS = self.dataset.word_2_idx(self.dataset.EOS)
        PAD = self.dataset.word_2_idx(self.dataset.PAD)
        data = [self.dataset[idx] for idx in self._indices[self.index:self.index+self.batch_size]]
        self.index = self.index + self.batch_size

        if self.index + self.batch_size > self._data_size:
            self.index = 0
            self.permute()

        tokens = [[self.dataset.word_2_idx(w) for w in sentence] for sentence in data]

        lengths = [len(sentence)+1 for sentence in tokens]
        max_length = max(lengths)

        inputs = [[SOS]+sentence for sentence in tokens]
        inputs = [sentence+[PAD]*(max_length-len(sentence)) for sentence in inputs]

        outputs = [sentence+[EOS] for sentence in tokens]
        outputs = [sentence+[PAD]*(max_length-len(sentence)) for sentence in outputs]

        inputs = torch.tensor(inputs)
        outputs = torch.tensor(outputs)
        seq_mask = (inputs != PAD)
        seq_length = torch.tensor(lengths)

        return inputs, outputs, seq_mask, seq_length

    def permute(self):
        self._indices = np.random.permutation(self._data_size)

    def print_batch(self, batch):
        for sentence in batch:
            print(" ".join([self.dataset.idx_2_word(w) for w in sentence.tolist()]))




"""

class Dataset(data.Dataset):
    def __init__(self, path, seq_length):
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.UNK= '<UNK>'
        self._words = set([self.SOS, self.EOS, self.PAD, self.UNK])

        self._seq_length = seq_length

        self._data = open(os.path.join(path, "02-21.10way.clean"), "r")
        self._data = self.process(self._data)
        self._words = sorted(list(self._words))

        self._data_size, self._vocab_size = len(self._data), len(self._words)

        self._word_to_ix = { ch:i for i,ch in enumerate(self._words)  }
        self._ix_to_word = { i:ch for i,ch in enumerate(self._words)  }

        self._max_length = max([len(sample) for sample in self._data]) + 1

        self._offset = 0
        self.permute()
        self._index = 0

    def permute(self):
        self._indices = np.random.permutation(self._data_size)

    def __getitem__(self, item):
        # offset = np.random.randint(0, len(self._data)-self._seq_length-2)
        inputs =  [self._word_to_ix[ch] for ch in self._data[self._indices[self._index]]]#[offset:offset+self._seq_length]]
        targets = [self._word_to_ix[ch] for ch in self._data[self._indices[self._index]]]#[offset+1:offset+self._seq_length+1]]
        inputs = [self._word_to_ix[self.SOS]] + inputs
        targets = targets + [self._word_to_ix[self.EOS]]
        lengths = len(inputs)

        inputs = inputs + [self._word_to_ix[self.PAD]] * (self._max_length - lengths)
        targets = targets + [self._word_to_ix[self.PAD]] * (self._max_length - lengths)

        assert len(inputs) == self._max_length

        self._index += 1
        if self._index == self._data_size:
            self.permute()
            self._index = 0

        return inputs, targets

    def convert_to_string(self, word_ix):
        return ' '.join(self._ix_to_word[ix] for ix in word_ix)

    def __len__(self):
        return self._data_size

    @property
    def vocab_size(self):
        return self._vocab_size

    
"""