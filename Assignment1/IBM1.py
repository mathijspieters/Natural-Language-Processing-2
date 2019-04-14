from read_data import read_data
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import utils
import os

class IBM1():
    def __init__(self):
        self.corpus = None
        self.thetas = None
        self.theta_0 = None

    def get_corpus(self, e_path, f_path, l=-1):
        self.corpus = read_data(e_path, f_path)[:l]

    def save(self, name):
        doc = {'thetas':self.thetas, 'theta_0':self.theta_0}
        utils.save(doc, os.path.join('model', name+'.pickle'))

    def load(self, name):
        doc = utils.load( os.path.join('model', name+'.pickle'))
        self.thetas = doc['thetas']
        self.theta_0 = doc['theta_0']

    def fit(self, iterations=10, save=False):
        if self.corpus is None:
            print("Hey you forgot to give me something to work with")
            return

        print("Welcome to IBM1. Today, we'll be training for", iterations, "iterations. We wish you a pleasant journey.")

        tmp = set()
        for _, F in self.corpus:
            tmp = tmp | set(F.c.keys())

        total_french_words = len(tmp)
        self.theta_0 = 1/total_french_words

        self.thetas = defaultdict(dict)

        for i in range(iterations):
            print("Log likelihood:", self.Likelihood())
            count_ef = Counter()
            count_e = Counter()
            for E, F in tqdm(self.corpus):
                for f in F:
                    Z = 0
                    for e in E:
                        Z += self.thetas[e].get(f, self.theta_0)
                    for e in E:
                        c = self.thetas[e].get(f, self.theta_0)/Z
                        count_ef[(e,f)] += c
                        count_e[e] += c

            for e,f in count_ef:
                self.thetas[e][f] = count_ef[(e,f)]/count_e[e]

            if save:
                self.save('IBM-%d' % i)

        print("Log likelihood:", self.Likelihood())

    def Likelihood(self):
        LL = 0
        for E, F in self.corpus:
            for f in F:
                theta_sum = 0
                for e in E:
                    theta_sum += self.thetas[e].get(f, self.theta_0)
                LL += np.log(theta_sum)
        return LL

    def translate(self, source):
        source = source.replace("\n", "").split(" ")

        target = []

        for token in source:
            best_p = 0
            best_f = "empty"
            for f in self.thetas[token]:
                if self.thetas[token][f] > best_p:
                    best_p = self.thetas[token][f]
                    best_f = f
            target.append(best_f)

        return " ".join(target)

    def validation(self, e_path, f_path, save_file='results/results.out'):
        validation_corpus = read_data(e_path, f_path)
        
        with open(save_file, 'w') as f:
            for i, (E, F) in enumerate(validation_corpus):
                _, alignment = self.viterbi_alignment(E.s, F.s, split=False)
                for j in range(alignment.shape[0]):
                    f.write("%d %d %d S\n" % (i+1, j+1, alignment[j]))


    def viterbi_alignment(self, source, target, split=True):
        if split:
            source = source.replace("\n", "").split(" ")
            target = target.replace("\n", "").split(" ")

        alignment_p = np.zeros(shape=(len(source),len(target)))
        
        for i, word_source in enumerate(source):
            for j, word_target in enumerate(target):
                alignment_p[i,j] = self.thetas[word_source].get(word_target, self.theta_0)

        
        alignments_sum = np.sum(alignment_p, axis=1, keepdims=True)

        alignment_p /= alignments_sum

        alignments = np.argmax(alignment_p, axis=0)

        return alignment_p, alignments