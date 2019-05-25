import torch
import os
import numpy as np
from collections import Counter
import string
import random

from torch.utils import data
from nltk.tree import Tree
from torch.autograd import Variable


class Dataset(data.Dataset):
    def __init__(self, path, file_="02-21.10way.clean", sorted_words=None):
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.PAD = '<PAD>'
        self.UNK= '<UNK>'
        self.count = Counter()
        self._words = set([self.SOS, self.EOS, self.PAD, self.UNK])
        self.data = open(os.path.join(path, file_), "r")
        self.data = self.process(self.data)
        self._words |= set([word for (word, count) in self.count.most_common(100000)])

        if sorted_words == None:
            self.sorted_words = sorted(list(self._words))
        else:
            self.sorted_words = sorted_words
        self._words = self.sorted_words
        self._word_2_idx = {}
        self._idx_2_word = {}
        self.create_vocabulary()
        self.vocab_size = len(list(self._words))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def process(self, data):
        tmp = []
        #punctuation = string.punctuation
        for i, d in enumerate(data.readlines()):
            t = Tree.fromstring(d)
            sentence = t.leaves()
            #sentence = [w.lower() for w in sentence]
            #sentence = [w for w in sentence if w not in punctuation]
            self.count += Counter(sentence)
            tmp.append(sentence)
        return tmp

    def create_vocabulary(self):
        self._word_2_idx = dict(zip(list(self._words), range(len(list(self._words)))))
        self._idx_2_word = {self._word_2_idx[word]:word for word in self._word_2_idx.keys()}

    def word_2_idx(self, word):
        return self._word_2_idx[word] if word in self._word_2_idx else self._word_2_idx[self.UNK]

    def idx_2_word(self, idx):
        return self._idx_2_word[idx] if idx in self._idx_2_word else self.UNK


class DataLoader:
    def __init__(self, dataset, batch_size=8, word_dropout=0.):
        self.dataset = dataset
        self._data_size = len(self.dataset)
        self.permute()
        self.index = 0
        self.epoch = 0
        self.batch_size = batch_size
        self.word_dropout = word_dropout

    def __getitem__(self, item):
        SOS = self.dataset.word_2_idx(self.dataset.SOS)
        EOS = self.dataset.word_2_idx(self.dataset.EOS)
        PAD = self.dataset.word_2_idx(self.dataset.PAD)
        UNK = self.dataset.word_2_idx(self.dataset.UNK)

        data = [self.dataset[idx] for idx in self._indices[self.index:self.index+self.batch_size]]
        self.index = self.index + self.batch_size

        if self.index + self.batch_size > self._data_size:
            self.index = 0
            self.epoch += 1
            self.permute()

        tokens = [[self.dataset.word_2_idx(w) for w in sentence] for sentence in data]

        lengths = np.array([len(sentence)+1 for sentence in tokens])
        max_length = max(lengths)

        idx_sorted = np.argsort(lengths)[::-1]

        lengths = lengths[idx_sorted]
        tokens = [tokens[idx] for idx in idx_sorted]

        if self.word_dropout > 0:
            inputs = [[SOS]+ [i if random.uniform(0,1) > self.word_dropout else UNK for i in inp] for inp in tokens]
        else:
            inputs = [[SOS]+inp for inp in tokens]

        inputs = [sentence+[PAD]*(max_length-len(sentence)) for sentence in inputs]

        outputs = [sentence+[EOS] for sentence in tokens]
        outputs = [sentence+[PAD]*(max_length-len(sentence)) for sentence in outputs]

        inputs = torch.tensor(inputs)
        outputs = torch.tensor(outputs)
        seq_mask = (inputs != PAD).long()
        seq_length = torch.tensor(lengths)

        return inputs, outputs, seq_mask, seq_length

    def permute(self):
        self._indices = np.random.permutation(self._data_size)

    def print_batch(self, batch, stop_after_EOS=False):
        s = []
        for sentence in batch:
            sentence = sentence.tolist()
            if stop_after_EOS:
                sentence_ = []
                for w in sentence:
                    if w != self.dataset.word_2_idx(self.dataset.EOS):
                        sentence_.append(w)
                    else:
                        sentence = sentence_
                        break
            sent = " ".join([self.dataset.idx_2_word(w) for w in sentence])
            s.append(sent)
        return s


def load_dataset(config, type_='train', sorted_words=None, dropout_rate=0.2):
    assert type_ in ['train', 'test', 'train_eval', 'validation'], 'Type must be train/test/train_eval/validation'

    if type_ == 'train':
        dataset = Dataset('data')
        data_loader = DataLoader(dataset, batch_size=config.batch_size, word_dropout=dropout_rate)
    elif type_ == 'test':
        dataset = Dataset('data', file_='23.auto.clean', sorted_words=sorted_words)
        data_loader = DataLoader(dataset, batch_size=config.batch_size)
    elif type_ == 'validation':
        dataset = Dataset('data', file_='22.auto.clean', sorted_words=sorted_words)
        data_loader = DataLoader(dataset, batch_size=config.batch_size)
    elif type_ == 'train_eval':
        dataset = Dataset('data', sorted_words=sorted_words)
        data_loader = DataLoader(dataset, batch_size=config.batch_size)

    return dataset, data_loader
