from read_data import read_data
from collections import Counter

class IBM1():
    def __init__(self, iterations=10):
        self.corpus = None

    def get_corpus(self, e_path, f_path, l=5000):
        self.corpus = read_data(e_path, f_path)[:l]

    def fit(self):
        if self.corpus is None:
            print("Hey you forgot to give me something to work with")
            return

        tmp = set()
        for _, F in self.corpus:
            tmp = tmp | set(F.c.keys())

        total_french_words = len(tmp)
        print(total_french_words)
        thetha = 1/total_french_words
        # for thing in something:
        #     for something in something else:
        #         train the model

        # return a good model
