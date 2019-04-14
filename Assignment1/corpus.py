class Corpus():

    def __init__(self):
        self.corpus = []
        self.foreign_words = set()

    def add_foreign(self, words):
        self.foreign_words |= words
