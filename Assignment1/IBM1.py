from read_data import read_data
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np

class IBM1():
    def __init__(self, iterations=10):
        self.corpus = None
        self.thetas = None
        self.iterations = iterations

    def get_corpus(self, e_path, f_path, l=-1):
        self.corpus = read_data(e_path, f_path)[:l]

    def fit(self):
        if self.corpus is None:
            print("Hey you forgot to give me something to work with")
            return

        print("Welcome to IBM1. Today, we'll be training for", self.iterations, "iterations. We wish you a pleasant journey.")

        tmp = set()
        for _, F in self.corpus:
            tmp = tmp | set(F.c.keys())

        total_french_words = len(tmp)
        self.theta_0 = 1/total_french_words

        self.thetas = defaultdict(dict)

        for i in tqdm(range(self.iterations)):
            print("Log likelihood:", self.Likelihood())
            count_ef = Counter()
            count_e = Counter()
            for E, F in self.corpus:
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
        source = source.replace("\n", "")
        source = source.split(" ")

        target = []


        for token in source:
            best_p = 0
            best_f = "empty"
            for f in self.thetas[token]:
                if self.thetas[token][f] > best_p:
                    best_p = self.thetas[token][f]
                    best_f = f
            target.append(f)

        return " ".join(target)
