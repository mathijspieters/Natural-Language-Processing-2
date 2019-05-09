import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Dataset
from dataset import DataLoader

train_dataset = Dataset('data')

loader = DataLoader(train_dataset)

for batch_inputs, batch_outputs, masks, lengths in loader:
    loader.print_batch(batch_inputs)
    print(lengths)
    print(masks)
    