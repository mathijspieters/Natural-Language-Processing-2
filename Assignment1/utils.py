import pickle
import os

def save(object, name):
    path = os.getcwd()
    with open(os.path.join(path, name), "wb") as output:
        pickle.dump(object, output)

def load(name):
    path = os.getcwd()
    with open(os.path.join(path, name), "rb") as input:
        object = pickle.load(input)
    return object
