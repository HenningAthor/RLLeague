"""
Script to train one agent via evolution.
It will use a downloaded match to test the performance of the agent.
The agent will be mutated 100 times and the mutated agent with the smallest error is
chosen next for mutation. This process is repeated 100 times.
"""
import itertools
import os.path
import pickle
import time

import numpy as np

from agent.tree import Tree, recombine_trees
from recorded_data.data_util import load_match, split_data, load_min_max_csv, scale_with_min_max, concat_multiple_datasets, generate_env_stats, generate_env, load_all_silver_parquet_paths

if __name__ == '__main__':
    np.set_printoptions(precision=4)

    # load data
    n_data = 5
    file_list = load_all_silver_parquet_paths()
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

    assert np.max(features) <= 1.0
    assert np.min(features) >= 0.0

    max_data = np.max(features, axis=0)
    min_data = np.min(features, axis=0)
    real_min_max_data = np.array([max_data, min_data])

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
    env_stats = generate_env_stats(env_variables, real_min_max_data, min_max_headers)

    print(np.unique(player1[:, 0], return_counts=True))

    mutate_probabilities = [0.01]
    tree_sizes = [(8, 10)]

    for mutate_p, (min_tree_size, max_tree_size) in itertools.product(mutate_probabilities, tree_sizes):

        print(mutate_p, min_tree_size, max_tree_size)
        # set parameters
        n_epochs = 200
        n_trees = 500
        errors = np.zeros(n_trees)
        n_nodes_list = np.zeros(n_trees)
        n_non_bloat_nodes_list = np.zeros(n_trees)
        weights = np.zeros(n_trees)

        # storage so we can analyze later
        errors_storage = np.zeros((n_epochs, n_trees))
        n_nodes_storage = np.zeros((n_epochs, n_trees))
        n_non_bloat_nodes_storage = np.zeros((n_epochs, n_trees))

        # create agents
        t = time.time()
        tree_list = []
        for i in range(n_trees):
            tree = Tree(min_tree_size, max_tree_size, env_variables, -1.0, 1.0, discrete_return=False)
            tree.determine_bloat(env_stats)
            tree.assert_tree()
            tree_list.append(tree)
        print('Trees created in', time.time() - t)
        print(features.shape)
        print(player1.shape)

        # start learning
        for epoch in range(n_epochs):
            print('Epoch:', epoch, end=' ')

            for i, tree in enumerate(tree_list):
                n_nodes_list[i] = np.array(tree.count_nodes())
                n_non_bloat_nodes_list[i] = np.array(tree.count_non_bloat_nodes())

            t = time.time()
            for i, tree in enumerate(tree_list):
                x = tree.eval(env)
                res = np.average(np.square(player1[:, 0] - x))
                print(res)
                assert np.all(np.max(x) <= 1.0)
                assert np.all(np.min(x) >= -1.0)
                assert np.all(res <= 4.0)
                assert np.all(res >= 0.0)
                assert np.all(np.max(player1[:, 0]) <= 1.0)
                assert np.all(np.min(player1[:, 0]) >= -1.0)
                errors[i] = res
            print('Eval', "{:.4f}".format(time.time() - t), end=' ')

            errors_storage[epoch] = errors
            n_nodes_storage[epoch] = n_nodes_list
            n_non_bloat_nodes_storage[epoch] = n_non_bloat_nodes_list

            # generate weights for each tree
            inv_w = 1 / errors
            weights = inv_w / np.sum(inv_w)

            new_tree_list = []
            top_10_trees = []
            top_10_indices = np.argpartition(weights, -10)[-10:]
            for idx in top_10_indices:
                top_10_trees.append(tree_list[idx])
                new_tree_list.append(tree_list[idx])

            # mutate result agent
            t = time.time()
            for i in range(int(n_trees * 0.5)):
                mutated_tree = np.random.choice(a=top_10_trees).__deepcopy__()
                mutated_tree.mutate(mutate_p)
                new_tree_list.append(mutated_tree)
            print('Mutation', "{:.4f}".format(time.time() - t), end=' ')

            # recombine the best agents
            t = time.time()
            for i in range(int(n_trees * 0.2)):
                trees = np.random.choice(a=top_10_trees, size=2)
                tree_1, tree_2 = trees[0].__deepcopy__(), trees[1].__deepcopy__()
                recombine_trees(tree_1, tree_2)
                new_tree_list.append(tree_1)
                new_tree_list.append(tree_2)
            print('Recombination', "{:.4f}".format(time.time() - t), end=' ')

            # insert new agents
            t = time.time()
            for i in range(n_trees - len(new_tree_list)):
                tree = Tree(min_tree_size, max_tree_size, env_variables, -1.0, 1.0, discrete_return=False)
                tree.assert_tree()
                new_tree_list.append(tree)
            print('Creation', "{:.4f}".format(time.time() - t), end=' ')

            # bloat analysis
            t = time.time()
            for tree in new_tree_list:
                tree.determine_bloat(env_stats)
                tree.assert_tree()
            print('Bloat', "{:.4f}".format(time.time() - t), end=' ')
            print('Lowest Error:', np.min(errors))

            tree_list = new_tree_list

        path = "temp_data/"
        if not os.path.exists(path):
            os.mkdir(path)

        file_name = f"{mutate_p}_{min_tree_size}_{max_tree_size}.pickle"
        with open(path + file_name, 'wb') as handle:
            pickle.dump((errors_storage, n_nodes_storage, n_non_bloat_nodes_storage), handle, protocol=pickle.HIGHEST_PROTOCOL)
