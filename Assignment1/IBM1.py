from read_data import read_data

class IBM1():
    def __init__(self, something, who_cares):
        self.corpus = None

    def get_corpus(self, path1, path2):
        self.corpus = read_data(path1, path2)

    def fit(self):
        if self.corpus is None:
            print("Hey you forgot to give me something to work with")
            return

        for thing in something:
            for something in something else:
                train the model

        return a good model
