class Corpus():
    def __init__(self):
        self.corpus = []
        self.foreign_words = set()

    def add_foreign(self, words):
        self.foreign_words |= words

    def get_L(self):
        max_L = 0
        for E, F in self.corpus:
            max_L = max(len(E), max_L)
        return max_L
