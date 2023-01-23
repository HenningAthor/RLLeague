import copy
import csv
import time

import numpy as np

from genetic_lab.bot_generation import create_bot

if __name__ == '__main__':
    # load the game data
    csv_file_path = 'recorded_data/downloaded_matches/a11526b4-ea11-4214-bdb2-804bdcf531ee.csv'
    game_data = np.genfromtxt(csv_file_path, dtype='float64', delimiter=',', skip_header=True)[0:1, :]
    f = open(csv_file_path, 'r')
    reader = csv.reader(f)
    features_header = list(next(reader, None))
    f.close()

    # split data into features and labels
    temp = np.split(game_data, [len(features_header) - 16], axis=1)
    features = temp[0]
    s = np.split(temp[1], [8], axis=1)
    labels = s[0]
    features_header = features_header[:-16]
    labels_header = features_header[-16:-8]

    # load the min-max data
    csv_file_path = 'recorded_data/min_max.csv'
    min_max_data = np.genfromtxt(csv_file_path, dtype='float32', delimiter=',', skip_header=True)
    f = open(csv_file_path, 'r')
    reader = csv.reader(f)
    headers = list(next(reader, None))
    f.close()

    env_variables = {'ARITHMETIC': ['inverted_ball/pos_x',
                                    'inverted_ball/pos_y',
                                    'inverted_ball/pos_z',
                                    'inverted_ball/vel_x',
                                    'inverted_ball/vel_y',
                                    'inverted_ball/vel_z',
                                    'inverted_player1/pos_x',
                                    'inverted_player1/pos_y',
                                    'inverted_player1/pos_z',
                                    'inverted_player1/vel_x',
                                    'inverted_player1/vel_y',
                                    'inverted_player1/vel_z',
                                    'inverted_player2/pos_x',
                                    'inverted_player2/pos_y',
                                    'inverted_player2/pos_z',
                                    'inverted_player2/vel_x',
                                    'inverted_player2/vel_y',
                                    'inverted_player2/vel_z'],
                     'LOGIC': ['FALSE',
                               'TRUE']
                     }

    # Fill it with index of the feature
    env_index = {'ARITHMETIC': {}, 'LOGIC': {}}
    for key in env_variables['ARITHMETIC']:
        idx = features_header.index(key)
        env_index['ARITHMETIC'][key] = idx

    env = {'ARITHMETIC': {}, 'LOGIC': {'FALSE': False, 'TRUE': True}}
    env_stats = {'ARITHMETIC': {}, 'LOGIC': {'FALSE': {'min': False, 'max': False}, 'TRUE': {'min': True, 'max': True}}}
    for key in env_variables['ARITHMETIC']:
        env['ARITHMETIC'][key] = features[:, env_index['ARITHMETIC'][key]]
        env_stats['ARITHMETIC'][key] = {'min': 0.0, 'max': 1.0}

    # scale environment to [0.0, 1.0]
    for key in env_variables['ARITHMETIC']:
        idx = env_index['ARITHMETIC'][key]
        env['ARITHMETIC'][key] = (env['ARITHMETIC'][key] - min_max_data[1][idx]) / (min_max_data[0][idx] - min_max_data[1][idx])

        assert np.all(env['ARITHMETIC'][key] >= 0.0)
        assert np.all(env['ARITHMETIC'][key] <= 1.0)

    np.random.seed(0)
    bot = create_bot(0, 10, 12, env_variables)
    copy_bot = copy.deepcopy(bot)
    t = time.time()
    err = np.sum(np.square((labels - bot.eval_all(env))))
    t = time.time() - t
    print(err, t)

    bot.bloat_analysis(env_stats)
    t = time.time()
    err = np.sum(np.square((labels - bot.eval_all(env))))
    t = time.time() - t
    print(err, t)

    bot.unmark_bloat()
    t = time.time()
    err = np.sum(np.square((labels - bot.eval_all(env))))
    t = time.time() - t
    print(err, t)

    copy_bot.bloat_analysis(env_stats)
    work_list_1 = [bot.steering_root]
    work_list_2 = [copy_bot.steering_root]

    save_list_1 = []
    save_list_2 = []

    while work_list_1 and work_list_2:
        node_1 = work_list_1.pop(0)
        node_2 = work_list_2.pop(0)

        save_list_1.append(node_1)
        save_list_2.append(node_2)

        assert len(node_1.children) == len(node_2.children)

        work_list_1.extend(node_1.children)
        work_list_2.extend(node_2.children)

    assert len(save_list_1) == len(save_list_2)

    print('checking')
    for node_1, node_2 in zip(reversed(save_list_1), reversed(save_list_2)):
        res_1 = node_1.eval(env)
        res_2 = node_2.eval(env)

        if np.any(res_1 != res_2):
            print(node_1, node_2)
            print(res_1, [type(c) for c in node_1.children], [c.eval(env) for c in node_1.children])
            print(res_2, node_2.is_bloat, node_2.bloat_min, node_2.bloat_max)
            print()

