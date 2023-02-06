"""
Script to train one agent via evolution.
It will use a downloaded match to test the performance of the agent.
The agent will be mutated 100 times and the mutated agent with the smallest error is
chosen next for mutation. This process is repeated 100 times.
"""
import random
import time

import numpy as np

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

    # set parameters
    n_epochs = 100
    n_agents = 100
    errors = np.zeros((n_agents, 8))
    n_nodes_list = np.zeros((n_agents, 8))
    n_non_bloat_nodes_list = np.zeros((n_agents, 8))
    weights = np.zeros((n_agents, 8))
    mutate_p = 0.05
    recombine_p = 0.125
    min_tree_size = 3
    max_tree_size = 7

    # storage so we can analyze later
    errors_storage = np.zeros((n_epochs, n_agents, 8))
    n_nodes_storage = np.zeros((n_epochs, n_agents, 8))
    n_non_bloat_nodes_storage = np.zeros((n_epochs, n_agents, 8))
    err_result_agent = np.zeros((n_epochs, 8))

    # this agent will hold all the best trees
    result_agent = Agent(0, '', 1, 3, env_variables)
    result_agent.python_npy_jit()

    # create agents
    t = time.time()
    agent_list = []
    for i in range(n_agents):
        agent = Agent(0, '', min_tree_size, max_tree_size, env_variables)
        agent.bloat_analysis(env_stats)
        agent.assert_agent()
        agent.python_npy_jit()
        agent_list.append(agent)
    print('Agents created in', time.time() - t)

    # start learning
    for epoch in range(n_epochs):
        print('Epoch', epoch)
        t = time.time()
        for i, agent in enumerate(agent_list):
            errors[i] = np.average(np.square(player1 - agent.eval_all(env)), axis=0)
            n_nodes_list[i] = np.array(agent.count_nodes())
            n_non_bloat_nodes_list[i] = np.array(agent.count_non_bloat_nodes())
        print('Eval finished in', time.time() - t)

        errors_storage[epoch] = errors
        n_nodes_storage[epoch] = n_nodes_list
        n_non_bloat_nodes_storage[epoch] = n_non_bloat_nodes_list

        # get best tree from all agents
        res = np.average(np.square(player1 - result_agent.eval_all(env)), axis=0)
        for i in range(8):
            minimum = np.min(errors[:, i])
            idx = np.argmin(errors[:, i])

            if res[i] > minimum:
                result_agent.tree_list[i] = agent_list[idx].tree_list[i].__deepcopy__()

        # check error of result agent
        result_agent.bloat_analysis(env_stats)
        result_agent.assert_agent()
        err_result_agent[epoch] = np.average(np.square(player1 - result_agent.eval_all(env)), axis=0)
        print("Result Agent")
        print(err_result_agent[epoch])

        new_agent_list = []
        # mutate result agent
        t = time.time()
        for i in range(n_agents // 3):
            mutated_agent = result_agent.mutate(mutate_p)
            new_agent_list.append(mutated_agent)
        print('Agents mutated in', time.time() - t)

        # generate weights for each tree
        for i in range(8):
            inv_w = 1 / errors[:, i]
            weights[:, i] = inv_w / np.sum(inv_w)

        # recombine the best agents
        t = time.time()
        for i in range(n_agents // 6):
            idx = random.randint(0, 7)  # choose which tree to recombine
            agent_1 = np.random.choice(agent_list, p=weights[:, idx])
            agent_2 = np.random.choice(agent_list, p=weights[:, idx])
            rec_agent_1, rec_agent_2 = recombine_agents_by_tree(agent_1, agent_2, idx)
            new_agent_list.append(rec_agent_1)
            new_agent_list.append(rec_agent_2)
        print('Agents recombined in', time.time() - t)

        # insert new agents
        t = time.time()
        for i in range(n_agents - len(new_agent_list)):
            agent = Agent(0, '', min_tree_size, max_tree_size, env_variables)
            new_agent_list.append(agent)
        print('Agents created in', time.time() - t)

        agent_list = new_agent_list

        # bloat analysis
        t = time.time()
        for agent in agent_list:
            agent.bloat_analysis(env_stats)
            agent.assert_agent()
            agent.python_npy_jit()
        print('Bloat analyzed in', time.time() - t)

    print('Result Agent')
    for i in range(n_epochs):
        print(err_result_agent[i])

    result_agent.prepare_for_rlbot()
