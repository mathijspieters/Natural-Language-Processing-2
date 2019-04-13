from read_data import read_data
from IBM1 import IBM1
from utils import save, load


ENGLISH = 'data/training/hansards.36.2.e'
FRENCH = 'data/training/hansards.36.2.f'


if __name__ == '__main__':
    iterations = 10
    ibm1 = IBM1()
    ibm1.get_corpus(ENGLISH, FRENCH)

    ibm1.fit(iterations=iterations, save=True)

    #ibm1.load('IBM-9')

    print(ibm1.translate("The black cat"))

    alignments_matrix, alignment = ibm1.viterbi_alignment('the black cat', 'le chat noir')
    print(alignment)
    print(alignments_matrix)