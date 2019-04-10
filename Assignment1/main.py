from read_data import read_data
from ibm_model_1 import ibm_model_1


if __name__ == '__main__':
    corpus = read_data('data/training/hansards.36.2.e', 'data/training/hansards.36.2.f')
    print(len(corpus))
    # r = ibm_model_1(corpus)

