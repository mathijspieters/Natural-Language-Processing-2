from collections import Counter

class Sentence():
    """
    Stores the sentence, makes it iterable and initialises a counter object.
    """
    def __init__(self, s, pad=False):
        self.s = s
        if pad:
            self.s = ["NULL"] + self.s
        self.c = Counter(self.s)
        self.i = 0

    def __str__(self):
        return str(self.s)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.s)

    def __next__(self):
        if self.i == len(self.s):
            self.i = 0
            raise StopIteration
        else:
            self.i += 1
            return self.s[self.i - 1]

    def count(self, token):
        return self.c[token]
