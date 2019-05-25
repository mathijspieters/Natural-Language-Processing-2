import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset
import metrics

from sent_vae import SentVAE
from train_vae import evaluate

import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--num_hidden', type=int, default=100, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=20, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--word_dropout', type=float, default=0.2, help='Learning rate')


    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=1000, help='Learning rate step')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')

    # Misc params
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=5000, help='How often to sample from the model')

    parser.add_argument('--saved_model', type=str, default='model.pt')

    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--latent_size', type=int, default=16)

    config = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset, data_loader = load_dataset(config, type_='train', dropout_rate=config.word_dropout)
    dataset_test_eval, data_loader_test_eval = load_dataset(config, type_='test', sorted_words=dataset.sorted_words)
    dataset_validation_eval, data_loader_validation_eval = load_dataset(config, type_='validation', sorted_words=dataset.sorted_words)
    dataset_train_eval, data_loader_train_eval = load_dataset(config, type_='train_eval', sorted_words=dataset.sorted_words)

    print("Size of train dataset: %d" % len(dataset))
    print("Size of test dataset: %d" % len(dataset_test_eval))

    model = SentVAE(dataset.vocab_size, config.embedding_size, config.num_hidden, config.latent_size, config.num_layers, dataset.word_2_idx(dataset.PAD), dataset.word_2_idx(dataset.SOS), device)

    ppls = []
    ppls_std = []
    ss = []

    for s in tqdm(range(1, 20)):
        data = []
        for i in range(10):
            _, ppl, _ = evaluate(model, data_loader, dataset, device, s)
            data.append(ppl)
        data = np.array(data)
        ppls.append(data.mean())
        ppls_std.append(data.std())
        ss.append(s)

    plt.errorbar(ss, ppls, yerr=ppls_std)
    plt.xlabel("No. Samples")
    plt.ylabel("Perplexity estimate")
    plt.show()
