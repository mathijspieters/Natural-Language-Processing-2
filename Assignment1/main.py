from read_data import read_data
from IBM1 import IBM1
from utils import save, load
import aer


ENGLISH = 'data/training/hansards.36.2.e'
FRENCH = 'data/training/hansards.36.2.f'

ENGLISH_VAL = 'data/validation/dev.e'
FRENCH_VAL = 'data/validation/dev.f'


if __name__ == '__main__':
    iterations = 10
    ibm1 = IBM1()
    ibm1.get_corpus(ENGLISH, FRENCH)

    ibm1.fit(iterations=iterations, save=True)

    ibm1.load('IBM-9')

    print(ibm1.translate("the old cat"))

    alignments_matrix, alignment = ibm1.viterbi_alignment('the black cat', 'le chat noir')

    ibm1.validation(ENGLISH_VAL, FRENCH_VAL)

    #ibm1.plot_alignments(ENGLISH_VAL, FRENCH_VAL)

    aer.test_model(path_model='results/results.out')
