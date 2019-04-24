import pickle
import os
import collections

def nested_dict():
        return collections.defaultdict(nested_dict)


def save(doc, name):
    path = os.getcwd()
    with open(os.path.join(path, name), "wb") as output:
        pickle.dump(doc, output)

def load(name):
    path = os.getcwd()
    with open(os.path.join(path, name), "rb") as input:
        doc = pickle.load(input)
    return doc
