import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_dataset
import metrics

from sent_vae import SentVAE

def compute_loss(logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()
    return loss

def evaluate(model, data_loader, dataset, device):
    accuracy = 0
    perplexity = 0
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
            predictions, mu, sigma = model.forward(batch_inputs, lengths)
            predicted_targets = predictions.argmax(dim=-1)

            acc = metrics.ACC(predicted_targets, batch_targets, masks, lengths)
            ppl = metrics.ppl(predictions, batch_targets, masks)

            accuracy += (acc * batch_inputs.size(1))
            perplexity += ppl.item()
            sum_lengths += lengths.sum().item()

    return accuracy/num_samples, np.exp(perplexity/sum_lengths)

def train(config):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset, data_loader = load_dataset(config, type_='train')
    dataset_test_eval, data_loader_test_eval = load_dataset(config, type_='test', sorted_words=dataset.sorted_words)
    dataset_train_eval, data_loader_train_eval = load_dataset(config, type_='train_eval', sorted_words=dataset.sorted_words)

    model = SentVAE(dataset.vocab_size, config.embedding_size, config.num_hidden, config.latent_size, config.num_layers, dataset.word_2_idx(dataset.PAD), dataset.word_2_idx(dataset.SOS), device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    loss_sum, loss_kl_sum, loss_ce_sum, accuracy_sum  = 0, 0, 0, 0

    for step, (batch_inputs, batch_targets, masks, lengths) in enumerate(data_loader):
        optimizer.zero_grad()
        batch_inputs = batch_inputs.t().to(device)
        batch_targets = batch_targets.t().to(device)
        masks = masks.t().to(device)
        lengths = lengths.to(device)

        predictions, mu, sigma = model.forward(batch_inputs, lengths)

        predicted_targets = predictions.argmax(dim=-1)

        accuracy = metrics.ACC(predicted_targets, batch_targets, masks, lengths)
        
        ce_loss = compute_loss(predictions.transpose(1,0).contiguous(), batch_targets.t().contiguous(), masks.t())
        kl_loss = metrics.KL(mu, sigma)

        loss = ce_loss + kl_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()
        loss_kl_sum += kl_loss.item()
        loss_ce_sum += ce_loss.item()
        accuracy_sum += accuracy.item()

        if step % config.print_every == 0:
            print("Epoch: %2d      STEP %4d     Accuracy: %.3f   Total-loss: %.3f    CE-loss: %.3f   KL-loss: %.3f" %\
                (data_loader.epoch, step, accuracy_sum/config.print_every, loss_sum/config.print_every, loss_ce_sum/config.print_every, loss_kl_sum/config.print_every))

            loss_sum, loss_kl_sum, loss_ce_sum, accuracy_sum  = 0, 0, 0, 0

        if step % config.sample_every == 0:
            print("%s\nBATCH TARGETS:" % ("-"*60))
            data_loader.print_batch(batch_targets.t())
            print("%s\nPREDICTED TARGETS:" % ("-"*60))
            data_loader.print_batch(predicted_targets.t())
            print("%s\nSAMPLES:" % ("-"*60))
            sample = model.sample()
            data_loader.print_batch(sample.t())
            print("%s\nINTERPOLATION:" % ("-"*60))
            result = model.interpolation(n_steps=10)
            data_loader.print_batch(result.t())


        if step % 10000 == 0:
            eval_acc, eval_ppl = evaluate(model, data_loader_test_eval, dataset_test_eval, device)
            train_acc, train_ppl = evaluate(model, data_loader_train_eval, dataset_train_eval, device)

            print("Train accuracy-perplexity: %.3f-%.3f     Test accuracy-perplexity: %.3f-%.3f" % (train_acc, train_ppl, eval_acc, eval_ppl))
            torch.save(model.state_dict(), 'vae-model-%d.pt' % step)

        if step == config.train_steps:
            break



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--num_hidden', type=int, default=600, help='Number of hidden units in the LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=20, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=1000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=100000, help='Number of training steps')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=5000, help='How often to sample from the model')

    parser.add_argument('--saved_model', type=str, default='model.pt')

    parser.add_argument('--embedding_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=16)

    config = parser.parse_args()

    train(config)
