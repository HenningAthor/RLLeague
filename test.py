"""
Script to train one agent via evolution.
It will use a downloaded match to test the performance of the agent.
The agent will be mutated 100 times and the mutated agent with the smallest error is
chosen next for mutation. This process is repeated 100 times.
"""
import itertools
import os.path
import pickle
import random
import time

import numpy as np

from agent.agent import recombine_agents, Agent, recombine_agents_by_tree
from autoencoder.learn import AE
from recorded_data.data_util import load_match, split_data, load_min_max_csv, scale_with_min_max, load_all_parquet_paths, concat_multiple_datasets, generate_env_stats, generate_env, load_all_silver_parquet_paths, keep_certain_columns

if __name__ == '__main__':
    arr = np.array([[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                    [9, 10, 11]])

    print(arr)

    print(arr[:, [2, 0]])
