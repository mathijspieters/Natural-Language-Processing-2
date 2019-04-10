from read_data import read_data
from IBM1 import IBM1
from utils import save, load


ENGLISH = 'data/training/hansards.36.2.e'
FRENCH = 'data/training/hansards.36.2.f'


if __name__ == '__main__':
    iterations = 10
    ibm1 = IBM1(iterations)
    ibm1.get_corpus(ENGLISH, FRENCH)

    ibm1.fit()

    print(ibm1.translate("The black cat"))

    save(ibm1, "Henk")
