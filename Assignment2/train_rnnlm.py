import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Dataset
from dataset import DataLoader

from RNNLM import RNNLM


def load_dataset(config):
    dataset = Dataset('data')
    data_loader = DataLoader(dataset, batch_size=config.batch_size)
    return dataset, data_loader

def CE(predictions, targets, masks):
    targets = targets.contiguous().view(-1)
    predictions = predictions.view(-1, predictions.shape[-1])
    masks = masks.contiguous().view(-1).float()

    nb_tokens = torch.sum(masks)

    Y_hat = predictions[range(predictions.shape[0]), targets] * masks
    ce_loss = -torch.sum(Y_hat) / nb_tokens

    return ce_loss

def train(config):

    device = torch.device(config.device)
    dataset, data_loader = load_dataset(config)

    model = RNNLM(dataset.vocab_size, config.embedding_size, config.num_hidden, config.num_layers, dataset.word_2_idx(dataset.PAD))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_ce_sum, accuracy_sum  = 0, 0

    for step, (batch_inputs, batch_targets, masks, lengths) in enumerate(data_loader):
        optimizer.zero_grad()
        batch_inputs = batch_inputs.t().to(device)
        batch_targets = batch_targets.t().to(device)
        masks = masks.to(device)

        predictions = model.forward(batch_inputs, lengths)

        predicted_targets = predictions.argmax(dim=-1)

        accuracy = (predicted_targets[masks.t()] == batch_targets[masks.t()]).float().mean()

        loss = CE(predictions, batch_targets, masks)

        loss.backward()
        optimizer.step()

        loss_ce_sum += loss.item()
        accuracy_sum += accuracy.item()

        if step % config.print_every == 0:
            print("Epoch: %2d   STEP %4d      Accuracy: %.3f   CE-loss: %.3f " %\
                (data_loader.epoch, step, accuracy_sum/config.print_every, loss_ce_sum/config.print_every))

            loss_ce_sum, accuracy_sum  = 0, 0

        if step % config.sample_every == 0:
            data_loader.print_batch(predicted_targets.t())
            print()
            data_loader.print_batch(batch_targets.t())
            print()
            sample = model.sample(dataset.word_2_idx(dataset.SOS), 30)
            data_loader.print_batch(sample)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=8, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')

    # Misc params
    parser.add_argument('--print_every', type=int, default=50, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--saved_model', type=str, default='model.pt')

    parser.add_argument('--embedding_size', type=int, default=100)

    config = parser.parse_args()

    # print('vocab_size: {} - max_len: {}'.format(len(dl.vocab), dl.batch_max_len))

    # for s in dl.train:
    #     print(s)

    train(config)
