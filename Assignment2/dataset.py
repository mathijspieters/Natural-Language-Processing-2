<<<<<<< HEAD
import os
import numpy as np
import torch
from nltk.tree import Tree
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, path, seq_length):
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.UNK= '<UNK>'

        self._seq_length = seq_length

        self._data = open(os.path.join(path, "02-21.10way.clean"), "r")
        self._data = self.process(self._data)
        self._words = list(set(self._data))

        self._words = sorted(self._words)

        self._data_size, self._vocab_size = len(self._data), len(self._words)

        self._word_to_ix = { ch:i for i,ch in enumerate(self._words)  }
        self._ix_to_word = { i:ch for i,ch in enumerate(self._words)  }

        self._offset = 0

    def __getitem__(self, item):
        offset = np.random.randint(0, len(self._data)-self._seq_length-2)
        inputs =  [self._char_to_ix[ch] for ch in self._data[offset:offset+self._seq_length]]
        targets = [self._char_to_ix[ch] for ch in self._data[offset+1:offset+self._seq_length+1]]
        return inputs, targets

    def __len__(self):
        return self._data_size

    def process(self, data, train=False):
        tmp = []
        for i, d in enumerate(data.readlines()):
            t = Tree.fromstring(d)
            sentence = t.leaves()
            tmp.extend(sentence)
        return tmp
