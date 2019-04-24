from read_data import read_data
from IBM1 import IBM1
from IBM2 import IBM2
from utils import save, load
import aer

ENGLISH = 'data/training/hansards.36.2.e'
FRENCH = 'data/training/hansards.36.2.f'

ENGLISH_VAL = 'data/validation/dev.e'
FRENCH_VAL = 'data/validation/dev.f'

model_2 = True
use_jump = True

initialization = 'uniform' # 'uniform' 'random' 'IBM1'

if __name__ == '__main__':
    iterations = 10

    if model_2:
        ibm = IBM2(use_jump=use_jump, init=initialization)
    else:
        ibm = IBM1()

    ibm.get_corpus(ENGLISH, FRENCH, l=100)

    ibm.initialize_parameters()

    ibm.fit(iterations=iterations, save=True)

    #print(ibm.Likelihood())

    #ibm.load()

    #print(ibm.translate("the old cat"))

    #alignments_matrix, alignment = ibm.viterbi_alignment('the black cat', 'le chat noir')

    ibm.plot_alignments(ENGLISH_VAL, FRENCH_VAL)
