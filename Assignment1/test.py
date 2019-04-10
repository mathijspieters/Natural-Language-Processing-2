from read_data import *

corpus = read_data("data/training/hansards.36.2.e", "data/training/hansards.36.2.f")

for i in range(10):
    print(corpus[i])
