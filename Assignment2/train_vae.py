import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset
import metrics

from sent_vae import SentVAE

from torch.utils.tensorboard import SummaryWriter

from utils import markdown_hyperparams

def evaluate(model, data_loader, dataset, device):
    accuracy = 0
    perplexity = 0
    likelihood = 0
    sum_lengths = 0

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
            predictions, mu, sigma = model.forward(batch_inputs, lengths, greedy=True)
            predicted_targets = predictions.argmax(dim=-1)

            N = batch_inputs.size(1)

            accuracy += metrics.ACC(predicted_targets, batch_targets, masks, lengths)*N
            ll, ppl = metrics.eval_VAE(model, batch_inputs, batch_targets, masks, device)
            perplexity += ppl
            sum_lengths += lengths.sum()
            likelihood += ll*N


    return accuracy.item()/num_samples, np.exp(perplexity.item()/sum_lengths.item()), likelihood.item()/num_samples


def train(config):
    print("Thank you for choosing the Sentence VAE today!")
    print(config)
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset, data_loader = load_dataset(config, type_='train', dropout_rate=config.word_dropout)
    dataset_test_eval, data_loader_test_eval = load_dataset(config, type_='test', sorted_words=dataset.sorted_words)
    dataset_validation_eval, data_loader_validation_eval = load_dataset(config, type_='validation', sorted_words=dataset.sorted_words)
    dataset_train_eval, data_loader_train_eval = load_dataset(config, type_='train_eval', sorted_words=dataset.sorted_words)

    print("Size of train dataset: %d" % len(dataset))
    print("Size of test dataset: %d" % len(dataset_test_eval))

    model = SentVAE(dataset.vocab_size, config.embedding_size, config.num_hidden, config.latent_size, config.num_layers, dataset.word_2_idx(dataset.PAD), dataset.word_2_idx(dataset.SOS), device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    loss_sum, loss_kl_sum, loss_ce_sum, accuracy_sum  = 0, 0, 0, 0

    current_epoch = -1
    for step, (batch_inputs, batch_targets, masks, lengths) in enumerate(data_loader):
        optimizer.zero_grad()
        batch_inputs = batch_inputs.t().to(device)
        batch_targets = batch_targets.t().to(device)
        masks = masks.t().to(device)
        lengths = lengths.to(device)

        predictions, mu, sigma = model.forward(batch_inputs, lengths)

        predicted_targets = predictions.argmax(dim=-1)

        accuracy = metrics.ACC(predicted_targets, batch_targets, masks, lengths)

        ce_loss = metrics.compute_loss(predictions.transpose(1,0).contiguous(), batch_targets.t().contiguous(), masks.t())
        kl_loss = metrics.KL(mu, sigma)
        accuracy = metrics.ACC(predicted_targets, batch_targets, masks, lengths)

        ce_loss = metrics.compute_loss(predictions.transpose(1,0).contiguous(), batch_targets.t().contiguous(), masks.t())
        kl_loss = metrics.KL(mu, sigma).mean()

        loss = ce_loss + kl_loss

        loss.backward()
        optimizer.step()
        #scheduler.step()

        loss_sum += loss.item()
        loss_kl_sum += kl_loss.item()
        loss_ce_sum += ce_loss.item()
        accuracy_sum += accuracy.item()

        if step % config.print_every == 0:
            print("Epoch: %2d      STEP %4d     Accuracy: %.3f   Total-loss: %.3f    CE-loss: %.3f   KL-loss: %.3f" %\
                (data_loader.epoch, step, accuracy_sum/config.print_every, loss_sum/config.print_every, loss_ce_sum/config.print_every, loss_kl_sum/config.print_every))

            loss_sum, loss_kl_sum, loss_ce_sum, accuracy_sum  = 0, 0, 0, 0

        if step % config.sample_every == 0:

            targets = data_loader.print_batch(batch_targets.t())
            predictions = data_loader.print_batch(predicted_targets.t())
            for i in range(len(targets)):
                print("----------------------------")
                print(targets[i])
                print()
                print(predictions[i])
                print()



            print("%s\nSAMPLES:" % ("-"*60))
            sample = model.sample()
            sample = data_loader.print_batch(sample.t(), stop_after_EOS=True)
            for s in sample:
                print(s)
                print()


            print("%s\nINTERPOLATION:" % ("-"*60))
            result = model.interpolation(n_steps=10)
            result = data_loader.print_batch(result.t(), stop_after_EOS=True)
            for s in result:
                print(s)
                print()


        # if step % 5000 == 0:
        if data_loader.epoch != current_epoch:
            current_epoch = data_loader.epoch

            eval_acc, eval_ppl, eval_ll = evaluate(model, data_loader_test_eval, dataset_test_eval, device)
            val_acc, val_ppl, val_ll = evaluate(model, data_loader_validation_eval, dataset_validation_eval, device)
            train_acc, train_ppl, train_ll = evaluate(model, data_loader_train_eval, dataset_train_eval, device)

            print("Train accuracy-perplexity_likelihood: %.3f %.3f %.3f" % (eval_acc, eval_ppl, eval_ll))
            print("Test accuracy-perplexity-likelihood: %.3f %.3f %.3f" % (train_acc, train_ppl, train_ll))
            print("Validation accuracy-perplexity-likelihood: %.3f %.3f %.3f" % (val_acc, val_ppl, val_ll))

            writer.add_scalar('SVAE/Train accuracy', train_acc, current_epoch)
            writer.add_scalar('SVAE/Train perplexity', train_ppl, current_epoch)
            writer.add_scalar('SVAE/Train likelihood', train_ll, current_epoch)

            writer.add_scalar('SVAE/Test accuracy', eval_acc, current_epoch)
            writer.add_scalar('SVAE/Test perplexity', eval_ppl, current_epoch)
            writer.add_scalar('SVAE/Test likelihood', eval_ll, current_epoch)

            writer.add_scalar('SVAE/Valid accuracy', val_acc, current_epoch)
            writer.add_scalar('SVAE/Valid perplexity', val_ppl, current_epoch)
            writer.add_scalar('SVAE/Valid likelihood', val_ll, current_epoch)

            markdown_str = ''
            sample = model.sample()
            sample = data_loader.print_batch(sample.t(), stop_after_EOS=True)
            for s in sample:
                markdown_str += '{}  \n'.format(s)
            writer.add_text('SVAE/Samples', markdown_str, current_epoch)

            markdown_str = ''
            result = model.interpolation(n_steps=10)
            result = data_loader.print_batch(result.t(), stop_after_EOS=True)
            for s in result:
                markdown_str += '{}  \n'.format(s)
            writer.add_text('SVAE/Interpolation', markdown_str, current_epoch)


            torch.save(model.state_dict(), 'models/vae-model-%d.pt' % step)

        if data_loader.epoch == config.epochs:
            break



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

    parser.add_argument('--embedding_size', type=int, default=200)
    parser.add_argument('--latent_size', type=int, default=16)

    parser.add_argument('--comment', type=str, default='')

    config = parser.parse_args()

    writer = SummaryWriter(comment='-'+config.comment)

    # print hyperparameters to tensorboard
    params = markdown_hyperparams(config)
    writer.add_text('SVAE/Hyperparameters', params)

    train(config)
