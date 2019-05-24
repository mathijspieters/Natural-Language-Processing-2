import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset
import metrics

from RNNLM import RNNLM


def evaluate(model, data_loader, dataset, device):
    accuracy = 0
    likelihood = 0
    perplexity = 0

    num_samples = len(dataset)

    start_epoch = data_loader.epoch

    for step, (batch_inputs, batch_targets, masks, lengths) in enumerate(data_loader):
        if data_loader.epoch != start_epoch:
            data_loader.epoch = start_epoch
            break

        with torch.no_grad():
            batch_inputs = batch_inputs.t().to(device)
            batch_targets = batch_targets.t().to(device)
            masks = masks.t().to(device)
            lengths = lengths.to(device)
            predictions = model.forward(batch_inputs, lengths)
            predicted_targets = predictions.argmax(dim=-1)

            acc = metrics.ACC(predicted_targets, batch_targets, masks, lengths)
            ll, ppl = metrics.ppl_RNN(predictions, batch_targets, masks)

            N = batch_inputs.size(1)

            accuracy += acc*N
            likelihood += ll*N
            perplexity += ppl*N

    return accuracy.item()/num_samples, likelihood.item()/num_samples, perplexity.item()/num_samples


def train(config):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset, data_loader = load_dataset(config, type_='train')
    dataset_test_eval, data_loader_test_eval = load_dataset(config, type_='test', sorted_words=dataset.sorted_words)
    dataset_train_eval, data_loader_train_eval = load_dataset(config, type_='train_eval', sorted_words=dataset.sorted_words)

    print("Size of train dataset: %d" % len(dataset))
    print("Size of test dataset: %d" % len(dataset_test_eval))

    model = RNNLM(dataset.vocab_size, config.embedding_size, config.num_hidden, config.num_layers, dataset.word_2_idx(dataset.PAD), device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)


    loss_ce_sum, accuracy_sum  = 0, 0

    for step, (batch_inputs, batch_targets, masks, lengths) in enumerate(data_loader):
        optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        batch_inputs = batch_inputs.t().to(device)
        batch_targets = batch_targets.t().to(device)

        masks = masks.t().to(device)
        lengths = lengths.to(device)

        predictions = model.forward(batch_inputs, lengths)

        predicted_targets = predictions.argmax(dim=-1)

        accuracy = metrics.ACC(predicted_targets, batch_targets, masks, lengths)

        loss = metrics.compute_loss(predictions.transpose(1,0).contiguous(), batch_targets.t().contiguous(), masks.t())

        loss.backward()
        optimizer.step()
        #scheduler.step()

        loss_ce_sum += loss.item()
        accuracy_sum += accuracy

        if step % config.print_every == 0:
            print("Epoch: %2d   STEP %4d    Accuracy: %.3f   CE-loss: %.3f" %\
                (data_loader.epoch, step, accuracy_sum/config.print_every, loss_ce_sum/config.print_every))

            loss_ce_sum, accuracy_sum  = 0, 0

        if step % config.sample_every == 0:
            data_loader.print_batch(predicted_targets.t())
            print()
            data_loader.print_batch(batch_targets.t())
            print()
            sample = model.sample(dataset.word_2_idx(dataset.SOS), 30)
            data_loader.print_batch(sample)

        if step % 10000 == 0:
            eval_acc, eval_ppl = evaluate(model, data_loader_test_eval, dataset_test_eval, device)
            train_acc, train_ppl = evaluate(model, data_loader_train_eval, dataset_train_eval, device)

            print("Train accuracy-perplexity: %.3f-%.3f     Test accuracy-perplexity: %.3f-%.3f" % (train_acc, train_ppl, eval_acc, eval_ppl))
            torch.save(model.state_dict(), 'rnn-model-%d.pt' % step)

        if step == config.train_steps:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--num_hidden', type=int, default=600, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=20, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=1000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=100000, help='Number of training steps')

    # Misc params
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=5000, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--saved_model', type=str, default='model.pt')

    parser.add_argument('--embedding_size', type=int, default=100)

    config = parser.parse_args()

    train(config)
