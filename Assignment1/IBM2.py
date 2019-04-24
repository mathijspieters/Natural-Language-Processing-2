from read_data import read_data
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import random
import utils
import os
import plot_alignment
from IBM1 import IBM1

class IBM2(IBM1):
    def __init__(self, load_IBM1=True, use_jump=True, init='unif', random_seed=None):
        seed = random.randint(0, 100) if random_seed == None else random_seed
        random.seed(seed)
        if init == 'random':
            dir_ = 'model/IBM2_%s_%d_jump/' % (init,seed) if use_jump else 'model/IBM2_%s_%d_nojump/' % (init,seed)
        else:
            dir_ = 'model/IBM2_%s_jump/' % init if use_jump else 'model/IBM2_%s_nojump/' % init

        IBM1.__init__(self, save_dir=dir_)
        self.use_jump = use_jump
        self.init = init
        
    def initialize_parameters(self):
        if self.init == 'IBM1':
            # Load best ibm1 model
            self.load_IBM1(dir_='model/IBM1/IBM-29.pickle')
            self.initialize_gammas()
        elif self.init == 'uniform':
            total_french_words = len(self.corpus.foreign_words)
            self.theta_0 = 1/total_french_words
            self.thetas = defaultdict(dict)
            self.initialize_gammas()
        elif self.init == 'random':
            total_french_words = len(self.corpus.foreign_words)
            self.theta_0 = 1/total_french_words
            self.thetas = defaultdict(dict)
            self.initialize_gammas(rand=True)
        else:
            print('Initialization type is incorrect')
            exit(-1)

    def get_theta_0(self):
        if self.init == 'random':
            return random.uniform(0.0001, 0.05)
        else:
            return self.theta_0

    def jump(self, aj, j, E, F):
        """ Jump function from Vogel et al. (1996) """
        return aj - int(j*E/F)

    def load_IBM1(self, dir_=None):
        if dir_ == None:
            models = sorted([d for d in os.listdir('model/IBM1/') if d.split('.')[-1] == 'pickle'])
            last = models[-1]
            dir_ = os.path.join('model/IBM1/', last)
            doc = utils.load(dir_)
            self.thetas = doc['thetas']
            self.theta_0 = doc['theta_0']
            print("Initialized thetas from %s" % dir_)
        else:
            assert os.path.exists(dir_), '%s does not exists ;(' % dir_
            doc = utils.load(dir_)
            self.thetas = doc['thetas']
            self.theta_0 = doc['theta_0']
            print("Initialized thetas from %s" % dir_)


    def fit(self, iterations=10, save=False):
        likelihoods = []
        aers = []

        if self.corpus is None:
            print("Hey you forgot to give me something to work with")
            return

        print("Welcome to IBM2. Today, we'll be training for", iterations, "iterations. We wish you a pleasant journey.")

        for it in range(iterations):
            likelihood = self.Likelihood()
            aer = self.aer()
            likelihoods.append(likelihood)
            aers.append(aer)
            print("%d   Log likelihood: %.4f     AER: %.4f " % (it, likelihood, aer))
            count_ef = Counter()
            count_e = Counter()
            count_ijlm = Counter()
            count_jlm = Counter()
            for E, F in tqdm(self.corpus.corpus):
                l = len(E)
                m = len(F)

                for j, f in enumerate(F):
                    Z = 0
                    for i, e in enumerate(E):
                        Z += self.thetas[e].get(f, self.get_theta_0()) * self.get_gamma(i,j,l,m)
                    for i, e in enumerate(E):
                        c =  self.thetas[e].get(f, self.get_theta_0()) * self.get_gamma(i,j,l,m)/ Z
                        count_ef[(e, f)] += c
                        count_e[e] += c
                        count_ijlm[(i,j,l,m)] += c
                        count_jlm[(j,l,m)] += c

            for e, f in count_ef:
                self.thetas[e][f] = count_ef[(e, f)]/count_e[e]


            if self.use_jump:
                values = Counter()
                for i,j,l,m in count_ijlm:
                    value = count_ijlm[(i,j,l,m)] / count_jlm[(j,l,m)]
                    values[self.jump(i,j,l,m)] += value

                sum_ = sum(values.values())
                for i in values:
                    self.set_gamma(i, 0, 1, 1, values[i]/sum_)

            else:
                for i,j,l,m in count_ijlm:
                    value = count_ijlm[(i,j,l,m)] / count_jlm[(j,l,m)]
                    self.set_gamma(i, j, l, m, value)

            if save:
                self.save('IBM-%d' % it)

        likelihood = self.Likelihood()
        aer = self.aer()
        likelihoods.append(likelihood)
        aers.append(aer)
        print("%d   Log likelihood: %.4f     AER: %.4f " % (iterations, likelihood, aer))

        with open(os.path.join(self.save_dir, 'results.csv'), 'w') as f:
            for i, (aer, likelihood) in enumerate(zip(aers, likelihoods)):
                f.write('%d,%.4f,%.4f\n' % (i,aer,likelihood))

    def set_gamma(self, i, j, l, m, value):
        if self.use_jump:
            self.gammas[self.jump(i,j,l,m)] = value
        else:
            self.gammas[i][j][l][m] = value


    def get_gamma(self, i, j, l, m, min_=10e-6):
        if self.use_jump:
            return self.gammas.get(self.jump(i,j,l,m), min_)
        else:
            return self.gammas[i][j][l].get(m, min_)


    def initialize_gammas(self, rand=False):
        L = self.corpus.get_L()
        self.gammas = utils.nested_dict()
        for E, F in self.corpus.corpus:
            l = len(E)
            m = len(F)
            initial_prob = 1 / (l + 1)
            for i in range(l+1):
                for j in range(m+1):
                    if self.use_jump:
                        self.gammas[self.jump(i,j,l,m)] = 1./L if not rand else random.uniform(0, 0.05)
                    else:
                        self.gammas[i][j][l][m] = initial_prob if not rand else random.uniform(0, 0.05)
        print('Gammas initialized')
    
    def save(self, name):
        doc = {'thetas': self.thetas, 'theta_0': self.theta_0, 'gammas':self.gammas}
        utils.save(doc, os.path.join(self.save_dir, name+'.pickle'))

    def load(self, name='IBM-9'):
        d = os.path.join(self.save_dir, name+'.pickle')
        if os.path.exists(d):
            doc = utils.load(d)
            self.thetas = doc['thetas']
            self.theta_0 = doc['theta_0']
            self.gammas = doc['gammas']
            print("Loaded %s" % d)
        else:
            raise Exception('No model at {}'.format(d))

    def Likelihood(self):
        LL = 0
        for E, F in self.corpus.corpus:
            for j, f in enumerate(F):
                theta_sum = 0
                for i, e in enumerate(E):
                    theta_sum += self.get_gamma(i, j, len(E), len(F)) * self.thetas[e].get(f, self.theta_0)
                LL += np.log(theta_sum)
        return LL

    def viterbi_alignment(self, source, target, split=True):
        if split:
            source = source.replace(" \n", "").split(" ")
            target = target.replace(" \n", "").split(" ")

        alignment_p = np.zeros(shape=(len(source),len(target)))

        for i, word_source in enumerate(source):
            for j, word_target in enumerate(target):
                alignment_p[i,j] = self.get_gamma(i, j, len(source), len(target)) * self.thetas[word_source].get(word_target, self.theta_0)

        alignments_sum = np.sum(alignment_p, axis=1, keepdims=True)

        alignment_p /= (alignments_sum + 10e-8)

        alignments = np.argmax(alignment_p, axis=1)

        return alignment_p, alignments