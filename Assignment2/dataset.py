import torch
import nltk

class Dataloader():
    def __init__(self, path):
        self.train = open(path + "02-21.10way.clean", "r")
        self.val = open(path + "22.auto.clean", "r")
        self.test = open(path + "23.auto.clean", "r")


        
