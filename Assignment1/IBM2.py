from read_data import read_data
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import utils
import os
import plot_alignment
from IBM1 import IBM1

class IBM2(IBM1):
    def jump(self, aj, j, E, F):
        """ Jump function from Vogel et al. (1996) """
        return aj - int(j*len(E)/len(F))

    def fit(self, iterations=10, save=False):
        if self.corpus is None:
            print("Hey you forgot to give me something to work with")
            return

        print("Welcome to IBM2. Today, we'll be training for", iterations, "iterations. We wish you a pleasant journey.")

        total_french_words = len(self.corpus.foreign_words)
        self.theta_0 = 1/total_french_words
        self.thetas = defaultdict(dict)

        L = self.corpus.get_L()
        self.gammas = {l: 1./(2*L + 1) for l in range(-L, L+1)}
        for i in range(iterations):
            print("Log likelihood:", self.Likelihood())
            count_ef = Counter()
            count_e = Counter()
            count_gamma = Counter()
            for E, F in tqdm(self.corpus.corpus):
                for j, f in enumerate(F):
                    Z = 0
                    for aj, e in enumerate(E):
                        x = self.jump(aj, j, E, F)
                        Z += self.gammas[x]*self.thetas[e].get(f, self.theta_0)
                    for j, e in enumerate(E):
                        x = self.jump(aj, j, E, F)
                        c = self.thetas[e].get(f, self.theta_0)*self.gammas[x]/Z

                        count_ef[(e, f)] += c
                        count_e[e] += c
                        count_gamma[x] += c

            for e, f in count_ef:
                self.thetas[e][f] = count_ef[(e, f)]/count_e[e]
            for x in range(-L, L+1):
                self.gammas[x] = count_gamma[x]/sum(count_gamma)

            if save:
                self.save('IBM2-%d' % i)

        print("Log likelihood:", self.Likelihood())

    def Likelihood(self):
        LL = 0
        for E, F in self.corpus.corpus:
            for j, f in enumerate(F):
                theta_sum = 0
                for aj, e in enumerate(E):
                    x = self.jump(aj, j, E, F)
                    theta_sum += self.gammas[x]*self.thetas[e].get(f, self.theta_0)
                LL += np.log(theta_sum)
        return LL

    def viterbi_alignment(self, source, target, split=True):
        if split:
            source = source.replace(" \n", "").split(" ")
            target = target.replace(" \n", "").split(" ")

        alignment_p = np.zeros(shape=(len(source),len(target)))

        for i, word_source in enumerate(source):
            for j, word_target in enumerate(target):
                x = self.jump(i, j, source, target)
                alignment_p[i,j] = self.gammas[x]*self.thetas[word_source].get(word_target, self.theta_0)

        alignments_sum = np.sum(alignment_p, axis=1, keepdims=True)

        alignment_p /= alignments_sum

        alignments = np.argmax(alignment_p, axis=1)

        return alignment_p, alignments
