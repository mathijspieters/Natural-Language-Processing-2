import os
import torch
from nltk.tree import Tree


class Dataloader():
    def __init__(self, path):
        self.vocab = set()

        self.train = open(os.path.join(path, "02-21.10way.clean"), "r")
        self.val = open(os.path.join(path, "22.auto.clean"), "r")
        self.test = open(os.path.join(path, "23.auto.clean"), "r")

        self.train = self.process(self.train)
        self.val = self.process(self.val)
        self.test = self.process(self.test)


    def process(self, data):
        for d in data.readlines():
            t = Tree.fromstring(d)
            sentence = t.leaves()
            self.vocab |= set(sentence)
            print(sentence)
            break
        return data
