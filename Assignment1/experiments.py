from read_data import read_data
from IBM1 import IBM1
from IBM2 import IBM2
from utils import save, load
import aer

ENGLISH = 'data/training/hansards.36.2.e'
FRENCH = 'data/training/hansards.36.2.f'

ENGLISH_VAL = 'data/validation/dev.e'
FRENCH_VAL = 'data/validation/dev.f'

def perform_experiment(initialization, use_jump, iterations=15):

    ibm = IBM2(use_jump=use_jump, init=initialization)
    ibm.get_corpus(ENGLISH, FRENCH)
    ibm.initialize_parameters()
    ibm.fit(iterations=iterations, save=True)
    ibm.plot_alignments(ENGLISH_VAL, FRENCH_VAL)

if __name__ == "__main__":
    perform_experiment('IBM1', True)
    perform_experiment('uniform', True)
    perform_experiment('random', True)
    perform_experiment('random', True)
    perform_experiment('random', True)
    
    perform_experiment('uniform', False)
    perform_experiment('IBM1', False)
    perform_experiment('random', False)
    perform_experiment('random', False)
    perform_experiment('random', False)
