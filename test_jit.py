import itertools
import os.path
import pickle
import random
import time

import numpy as np
from numba import njit

from agent.agent import recombine_agents, Agent, recombine_agents_by_tree
from recorded_data.data_util import load_match, split_data, load_min_max_csv, scale_with_min_max, load_all_parquet_paths, concat_multiple_datasets, generate_env_stats, generate_env

if __name__ == '__main__':
    np.set_printoptions(precision=3)

    # load data
    n_data = 10
    file_list = load_all_parquet_paths()
    list_game_data = []
    list_headers = []
    for i in range(n_data):
        # load the game data
        game_data, headers = load_match(file_list[i])
        list_game_data.append(game_data)
        list_headers.append(headers)
    game_data, headers = concat_multiple_datasets(list_game_data, list_headers)

    # operate on data
    min_max_data, min_max_headers = load_min_max_csv()
    features, player1, player2, features_header, player1_header, player2_header = split_data(game_data, headers)
    features = scale_with_min_max(features, features_header, min_max_data, min_max_headers)

    env_variables = {'ARITHMETIC': ['ball/pos_x',
                                    'ball/pos_y',
                                    'ball/pos_z',
                                    'ball/vel_x',
                                    'ball/vel_y',
                                    'ball/vel_z',
                                    'ball/ang_vel_x',
                                    'ball/ang_vel_y',
                                    'ball/ang_vel_z',
                                    'player1/pos_x',
                                    'player1/pos_y',
                                    'player1/pos_z',
                                    'player1/vel_x',
                                    'player1/vel_y',
                                    'player1/vel_z',
                                    'player1/quat_w',
                                    'player1/quat_x',
                                    'player1/quat_y',
                                    'player1/quat_z',
                                    'player1/ang_vel_x',
                                    'player1/ang_vel_y',
                                    'player1/ang_vel_z',
                                    'player1/boost_amount',
                                    'inverted_player2/pos_x',
                                    'inverted_player2/pos_y',
                                    'inverted_player2/pos_z',
                                    'inverted_player2/vel_x',
                                    'inverted_player2/vel_y',
                                    'inverted_player2/vel_z',
                                    'inverted_player2/quat_w',
                                    'inverted_player2/quat_x',
                                    'inverted_player2/quat_y',
                                    'inverted_player2/quat_z',
                                    'inverted_player2/ang_vel_x',
                                    'inverted_player2/ang_vel_y',
                                    'inverted_player2/ang_vel_z',
                                    'player2/boost_amount'],
                     'LOGIC': ['player1/on_ground',
                               'player1/ball_touched',
                               'player1/has_jump',
                               'player1/has_flip',
                               'player2/on_ground',
                               'player2/ball_touched',
                               'player2/has_jump',
                               'player2/has_flip']}

    # fill the environment and the environment stats
    env = generate_env(env_variables, features, features_header)
    env_stats = generate_env_stats(env_variables, min_max_data, min_max_headers)

    agent = Agent(0, '', 3, 5, env_variables)
    agent.bloat_analysis(env_stats)
    agent.assert_agent()
    print(agent.count_nodes())
    print(agent.count_non_bloat_nodes())

    t = time.time()
    res_normal = agent.eval_all(env)
    print('Without Jit:', time.time() - t)

    agent.numba_jit(env_variables, features_header)
    t = time.time()
    res_jit = agent.eval_all(env, features)
    print('First run:', time.time() - t)

    t = time.time()
    res_jit2 = agent.eval_all(env, features)
    print('Second run:', time.time() - t)

    assert np.allclose(res_normal, res_jit)
    assert np.allclose(res_normal, res_jit2)
