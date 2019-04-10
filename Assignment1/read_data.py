from collections import Counter

class Sentence():
    """
    Stores the sentence, makes it iterable and initialises a counter object.
    """
    def __init__(self, s, pad=False):
        self.s = s.split(" ")
        if pad:
            self.s = ["NULL"] + self.s
        self.c = Counter(self.s)
        self.i = 0

    def __str__(self):
        return str(self.s)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == len(self.s):
            self.i = 0
            raise StopIteration
        else:
            self.i += 1
            return self.s[self.i - 1]

    def count(self, token):
        return self.c[token]

def read_data(e_path, f_path):
    """
    Combine two halves of a parallel corpus into one.

    :param e_path: path to language 1 file.
    :param f_path: path to language 2 file.
    :return: a list of tuples with parallel sentences. Sentences consist of
        a list of tokens.
    """
    fe = open(e_path, "r").read()
    ff = open(f_path, "r").read()

    corpus = []

    for e, f in zip(fe.split(" \n"), ff.split(" \n")):
        corpus.append((Sentence(e, pad=True),Sentence(f)))
    return corpus
