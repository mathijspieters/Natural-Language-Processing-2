from collections import Counter
from sentence import Sentence
from corpus import Corpus

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

    corpus = Corpus()

    for e, f in zip(fe.split(" \n"), ff.split(" \n")):
        e = e.split(' ')
        f = f.split(' ')
        corpus.add_foreign(set(f))
        corpus.corpus.append((Sentence(e, pad=True), Sentence(f)))
    return corpus
