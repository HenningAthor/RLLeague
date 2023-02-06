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
from recorded_data.data_util import load_match, split_data, load_min_max_csv, scale_with_min_max, load_all_parquet_paths, concat_multiple_datasets, generate_env_stats, generate_env, load_all_silver_parquet_paths

if __name__ == '__main__':
    np.set_printoptions(precision=4)

    # load data
    n_data = 2
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
    print(features.shape)

    mutate_probabilities = [0.001, 0.01, 0.05, 0.1]
    recombine_probabilities = [0.01, 0.125, 0.5]
    tree_sizes = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

    for mutate_p, recombine_p, (min_tree_size, max_tree_size) in itertools.product(mutate_probabilities, recombine_probabilities, tree_sizes):

        print(mutate_p, recombine_p, min_tree_size, max_tree_size)
        # set parameters
        n_epochs = 50
        n_agents = 100
        errors = np.zeros((n_agents, 8))
        n_nodes_list = np.zeros((n_agents, 8))
        n_non_bloat_nodes_list = np.zeros((n_agents, 8))
        weights = np.zeros((n_agents, 8))

        # storage so we can analyze later
        errors_storage = np.zeros((n_epochs, n_agents, 8))
        n_nodes_storage = np.zeros((n_epochs, n_agents, 8))
        n_non_bloat_nodes_storage = np.zeros((n_epochs, n_agents, 8))
        err_result_agent = np.zeros((n_epochs, 8))

        # this agent will hold all the best trees
        res_id = random.randint(0, 10000)
        result_agent = Agent(res_id, f'agent_{res_id}', 1, 3, env_variables)
        result_agent.numba_jit(env_variables, features_header)

        # create agents
        t = time.time()
        agent_list = []
        for i in range(n_agents):
            agent = Agent(0, '', min_tree_size, max_tree_size, env_variables)
            agent.bloat_analysis(env_stats)
            agent.assert_agent()
            # agent.python_npy_jit(env_variables, features_header)
            agent_list.append(agent)
        print('Agents created in', time.time() - t)

        # start learning
        for epoch in range(n_epochs):
            print('Epoch:', epoch, end=' ')

            for i, agent in enumerate(agent_list):
                n_nodes_list[i] = np.array(agent.count_nodes())
                n_non_bloat_nodes_list[i] = np.array(agent.count_non_bloat_nodes())

            t = time.time()
            for i, agent in enumerate(agent_list):
                res = np.average(np.square(player1 - agent.eval_all(env, features)), axis=0)
                errors[i] = res
            print('Eval', np.array([time.time() - t]), end=' ')

            errors_storage[epoch] = errors
            n_nodes_storage[epoch] = n_nodes_list
            n_non_bloat_nodes_storage[epoch] = n_non_bloat_nodes_list

            # get best tree from all agents
            res = np.average(np.square(player1 - result_agent.eval_all(env, features)), axis=0)
            for i in range(8):
                # print(i, errors[:, i])
                minimum = np.min(errors[:, i])
                idx = np.argmin(errors[:, i])

                if res[i] > minimum:
                    result_agent.tree_list[i] = agent_list[idx].tree_list[i].__deepcopy__()

            # check error of result agent
            result_agent.assert_agent()
            result_agent.bloat_analysis(env_stats)
            err_result_agent[epoch] = np.average(np.square(player1 - result_agent.eval_all(env)), axis=0)

            new_agent_list = []
            # mutate result agent
            t = time.time()
            for i in range(int(n_agents * 0.5)):
                mutated_agent = result_agent.mutate(mutate_p)
                new_agent_list.append(mutated_agent)
            print('Mutation', np.array([time.time() - t]), end=' ')

            # generate weights for each tree
            for i in range(8):
                inv_w = 1 / errors[:, i]
                weights[:, i] = inv_w / np.sum(inv_w)

            # recombine the best agents
            t = time.time()
            for i in range(int(n_agents * 0.2)):
                agent_1 = np.random.choice(agent_list)
                agent_2 = np.random.choice(agent_list)
                rec_agent_1, rec_agent_2 = recombine_agents(agent_1, agent_2, recombine_p)
                new_agent_list.append(rec_agent_1)
                new_agent_list.append(rec_agent_2)
            print('Recombination', np.array([time.time() - t]), end=' ')

            # insert new agents
            t = time.time()
            for i in range(n_agents - len(new_agent_list)):
                agent = Agent(0, '', min_tree_size, max_tree_size, env_variables)
                new_agent_list.append(agent)
            print('Creation', np.array([time.time() - t]), end=' ')

            agent_list = new_agent_list

            # bloat analysis
            t = time.time()
            for agent in agent_list:
                agent.bloat_analysis(env_stats)
                agent.assert_agent()
                # agent.python_npy_jit(env_variables, features_header)
            print('Bloat', np.array([time.time() - t]))

        print('Result Agent')
        for i in range(n_epochs):
            print(err_result_agent[i])

        result_agent.prepare_for_rlbot()

        path = "temp_data/"
        if not os.path.exists(path):
            os.mkdir(path)

        file_name = f"{mutate_p}_{recombine_p}_{min_tree_size}_{max_tree_size}.pickle"
        with open(path + file_name, 'wb') as handle:
            pickle.dump((errors_storage, n_nodes_storage, n_non_bloat_nodes_storage), handle, protocol=pickle.HIGHEST_PROTOCOL)
