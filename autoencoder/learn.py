"""
Script to learn an autoencoder.
"""
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import IterableDataset
import glob
import pathlib
import pandas as pd
from typing import List
from itertools import product
import numpy as np
from recorded_data.data_util import load_min_max_csv, scale_with_min_max_1d

"""
Implementation from: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
"""


class AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.hidden_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, features):
        activation = self.input_layer(features)
        activation = torch.relu(activation)
        code = self.hidden_layer(activation)
        code = torch.relu(code)
        activation = self.output_layer(code)
        reconstructed = torch.relu(activation)
        return reconstructed
    
    def save(self, path : str):
        torch.save(self.state_dict(), path)
    
    def load(self, path : str):
        self.load_state_dict(torch.load(path))

def string_product(separator : str, *lists : List[str]):
    return [separator.join(x) for x in product(*lists)]

class RocketLeagueDataset(IterableDataset):
    INPUT_SIZE = 18
    """    env_variables = {'ARITHMETIC': ['inverted_ball/pos_x',
                                    'inverted_ball/pos_y',
                                    'inverted_ball/pos_z',
                                    'inverted_ball/vel_x',
                                    'inverted_ball/vel_y',
                                    'inverted_ball/vel_z',
                                    'player1/pos_x',
                                    'player1/pos_y',
                                    'player1/pos_z',
                                    'player1/vel_x',
                                    'player1/vel_y',
                                    'player1/vel_z',
                                    'inverted_player2/pos_x',
                                    'inverted_player2/pos_y',
                                    'inverted_player2/pos_z',
                                    'inverted_player2/vel_x',
                                    'inverted_player2/vel_y',
                                    'inverted_player2/vel_z'] """
    def __init__(self, path : str, half_count = 100000):
        self.path = path
        self.half_count = half_count
        self.x = 0
        self.min_max_data, self.min_max_header = load_min_max_csv()

    def extract_information(self, row : pd.Series, ids : List[str], inverted : bool):
        player1 = 'inverted_' + ids[1] if inverted else ids[0]
        player2 = 'inverted_' + ids[0] if inverted else ids[1]
        ball = 'inverted_ball' if inverted else 'ball'

        data_of_interest1 = [player1, player2, ball]
        data_of_interest2 = ['/pos_x', '/pos_y', '/pos_z', '/vel_x', '/vel_y', '/vel_z']

        header = string_product('', ['player1', 'player2', 'ball'], data_of_interest2)
        data = row[string_product('', data_of_interest1, data_of_interest2)].to_numpy(dtype=np.float32)
        return scale_with_min_max_1d(data, header, self.min_max_data, self.min_max_header)
        

    
    def __iter__(self):
        paths = glob.glob(self.path + '/*.parquet')

        i = 0
        
        for path in paths:
            df = pd.read_parquet(path)
            ids = [name_column[:-len('/name')] for name_column in df.columns[df.columns.str.contains('/name')]]
            for index, row in df.iterrows():
                i += 1
                if i > self.half_count:
                    return

                yield self.extract_information(row, ids, False)
                yield self.extract_information(row, ids, True)
    
    def __len__(self):
        return 2 * self.half_count


def learn_mnist():
    epochs = 100

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_size=784, hidden_size=32).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

def learn_rl(path : str = 'ae_data', epochs: int = 1000, hidden_size = 10, output_path = None):

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_size= RocketLeagueDataset.INPUT_SIZE, hidden_size=hidden_size).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = RocketLeagueDataset(path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=1, pin_memory=True)

    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, RocketLeagueDataset.INPUT_SIZE).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    
    if (output_path is not None):
        model.save(output_path)