from read_data import *

corpus = read_data("data/training/hansards.36.2.e", "data/training/hansards.36.2.f")

E, F = corpus[10]
print(E, F)

for e in E:
    print(e)
    print(E.count)
for f in F:
    print(f)
