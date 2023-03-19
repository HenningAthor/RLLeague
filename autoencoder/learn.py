"""
Script to learn an autoencoder.
"""
import pickle

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

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def forward_to_hidden_layer(self, features):
        activation = self.input_layer(features)
        activation = torch.relu(activation)
        code = self.hidden_layer(activation)
        code = torch.relu(code)
        return code


def string_product(separator: str, *lists: List[str]):
    return [separator.join(x) for x in product(*lists)]


class RocketLeagueDataset(IterableDataset):
    INPUT_SIZE = 37

    def __init__(self, path: str, half_count=100000):
        self.path = path
        self.half_count = half_count
        self.x = 0
        self.min_max_data, self.min_max_header = load_min_max_csv()

    def extract_information(self, row: pd.Series, ids: List[str], inverted: bool):
        pandas_header = [f'{ids[0]}/pos_x',
                         f'{ids[0]}/pos_y',
                         f'{ids[0]}/pos_z',
                         f'{ids[0]}/vel_x',
                         f'{ids[0]}/vel_y',
                         f'{ids[0]}/vel_z',
                         f'{ids[0]}/ang_vel_x',
                         f'{ids[0]}/ang_vel_y',
                         f'{ids[0]}/ang_vel_z',
                         f'{ids[0]}/boost_amount',
                         f'{ids[0]}/on_ground',
                         f'{ids[0]}/ball_touched',
                         f'{ids[0]}/has_jump',
                         f'{ids[0]}/has_flip',
                         f'inverted_{ids[1]}/pos_x',
                         f'inverted_{ids[1]}/pos_y',
                         f'inverted_{ids[1]}/pos_z',
                         f'inverted_{ids[1]}/vel_x',
                         f'inverted_{ids[1]}/vel_y',
                         f'inverted_{ids[1]}/vel_z',
                         f'inverted_{ids[1]}/ang_vel_x',
                         f'inverted_{ids[1]}/ang_vel_y',
                         f'inverted_{ids[1]}/ang_vel_z',
                         f'{ids[1]}/boost_amount',
                         f'{ids[1]}/on_ground',
                         f'{ids[1]}/ball_touched',
                         f'{ids[1]}/has_jump',
                         f'{ids[1]}/has_flip',
                         f'ball/pos_x',
                         f'ball/pos_y',
                         f'ball/pos_z',
                         f'ball/vel_x',
                         f'ball/vel_y',
                         f'ball/vel_z',
                         f'ball/ang_vel_x',
                         f'ball/ang_vel_y',
                         f'ball/ang_vel_z']

        cleaned_header = [f'player1/pos_x',
                          f'player1/pos_y',
                          f'player1/pos_z',
                          f'player1/vel_x',
                          f'player1/vel_y',
                          f'player1/vel_z',
                          f'player1/ang_vel_x',
                          f'player1/ang_vel_y',
                          f'player1/ang_vel_z',
                          f'player1/boost_amount',
                          f'player1/on_ground',
                          f'player1/ball_touched',
                          f'player1/has_jump',
                          f'player1/has_flip',
                          f'inverted_player2/pos_x',
                          f'inverted_player2/pos_y',
                          f'inverted_player2/pos_z',
                          f'inverted_player2/vel_x',
                          f'inverted_player2/vel_y',
                          f'inverted_player2/vel_z',
                          f'inverted_player2/ang_vel_x',
                          f'inverted_player2/ang_vel_y',
                          f'inverted_player2/ang_vel_z',
                          f'player2/boost_amount',
                          f'player2/on_ground',
                          f'player2/ball_touched',
                          f'player2/has_jump',
                          f'player2/has_flip',
                          f'ball/pos_x',
                          f'ball/pos_y',
                          f'ball/pos_z',
                          f'ball/vel_x',
                          f'ball/vel_y',
                          f'ball/vel_z',
                          f'ball/ang_vel_x',
                          f'ball/ang_vel_y',
                          f'ball/ang_vel_z']

        data = row[pandas_header].to_numpy(dtype=np.float32)
        return scale_with_min_max_1d(data, cleaned_header, self.min_max_data, self.min_max_header)

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

    def __len__(self):
        return 2 * self.half_count


def learn_rl(path: str = 'ae_data', epochs: int = 1000, hidden_size=10, output_path=None):
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_size=RocketLeagueDataset.INPUT_SIZE, hidden_size=hidden_size).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = RocketLeagueDataset(path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=1, pin_memory=True)

    loss_list = []
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
        loss_list.append(loss)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    if output_path is not None:
        model.save(output_path)

    with open(f'{output_path}_loss.pickle', 'wb') as f:
        pickle.dump(loss_list, f)
