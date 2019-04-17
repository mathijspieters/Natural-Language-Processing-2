from read_data import read_data
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import utils
import os
import plot_alignment
from IBM1 import IBM1

class IBM2(IBM1):
    def jump(self, aj, j, l, m):
        """ Jump function from Vogel et al. (1996) """
        return aj - int(j*l/m)

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
            #print("Log likelihood:", self.Likelihood())
            count_ef = Counter()
            count_e = Counter()
            count_gamma = Counter()
            for E, F in tqdm(self.corpus.corpus):
                for j, f in enumerate(F):
                    Z = 0
                    for aj, e in enumerate(E):
                        x = self.jump(aj, j, len(E), len(F))
                        if abs(x) > L:
                            print(E)
                            print(F)
                            print(aj, j, len(E), len(F))
                        Z += self.gammas[x]*self.thetas[e].get(f, self.theta_0)
                    for j, e in enumerate(E):
                        x = self.jump(aj, j, len(E), len(F))
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
