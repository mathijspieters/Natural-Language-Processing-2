import torch
import os
import numpy as np

from torch.utils import data
from nltk.tree import Tree


class PadSequence:
    def __call__(self, batch):
		# Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        inputs, targets = batch
        print(inputs)
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
		# Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences)
		# Also need to store the length of each sequence
		# This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
		# Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor(map(lambda x: x[1], sorted_batch))
        return sequences_padded, labels, lengths


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

    def process(self, data, train=False):
        tmp = []
        for i, d in enumerate(data.readlines()):
            t = Tree.fromstring(d)
            sentence = t.leaves()
            self._words |= set(sentence)
            tmp.append(sentence)

        return tmp
