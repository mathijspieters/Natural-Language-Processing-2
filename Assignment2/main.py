import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Dataset
from dataset import DataLoader

from RNNLM import RNNLM


def load_dataset(config):
    # Initialize the dataset and data loader (note the +1)
    dataset = Dataset('data')
    data_loader = DataLoader(dataset, batch_size=config.batch_size)
    return dataset, data_loader

def train(config):

    device = torch.device(config.device)
    dataset, data_loader = load_dataset(config)

    model = RNNLM(dataset.vocab_size, config.embedding_size, config.num_hidden, config.num_layers, dataset.word_2_idx(dataset.PAD))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets, masks, lengths) in enumerate(data_loader):
        optimizer.zero_grad()
        batch_inputs = batch_inputs.t().to(device)
        batch_targets = batch_targets.t().to(device)

        predictions = model.forward(batch_inputs, lengths)

        predicted_targets = predictions.argmax(dim=-1)

        accuracy = (predicted_targets[:,masks] == batch_targets[:,masks]).float().mean()

        batch_targets = batch_targets * masks.t()

        predictions = predictions.view(-1, dataset.vocab_size)
        batch_targets = batch_targets.contiguous()
        batch_targets = batch_targets.view(-1)
        
        loss = criterion(predictions, batch_targets)

        loss.backward()
        optimizer.step()

        if step % config.print_every == 0:
            print("Accuracy: %.3f   Loss: %.3f" % (accuracy.item(), loss.item()))
        
        if step % config.sample_every == 0:
            data_loader.print_batch(predicted_targets.t())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=8, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-2, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')

    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--generated_output', type=str, default='generated_output.txt')
    parser.add_argument('--saved_model', type=str, default='model.pt')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--csv', type=str, default='model.csv')

    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--latent_size', type=int, default=20)

    config = parser.parse_args()

    # print('vocab_size: {} - max_len: {}'.format(len(dl.vocab), dl.batch_max_len))

    # for s in dl.train:
    #     print(s)

    train(config)
