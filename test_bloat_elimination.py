import copy
import time

import numpy as np

from genetic_lab.bot_generation import create_bot
from recorded_data.data_util import load_min_max_csv, load_match, scale_with_min_max, split_data

if __name__ == '__main__':
    # load the game data
    game_data, headers = load_match(f'recorded_data/out/a7303c84-7d9e-472d-aa60-403388827a41.parquet')
    game_data = game_data[0:2, :]

    # load min-max data
    min_max_data, min_max_headers = load_min_max_csv()

    # normalize game data
    game_data = scale_with_min_max(game_data, headers, min_max_data, min_max_headers)

    # split the data
    features, player1, player2, features_header, player1_header, player2_header = split_data(game_data, headers)

    env_variables = {'ARITHMETIC': ['inverted_ball/pos_x',
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
                                    'inverted_player2/vel_z'],
                     'LOGIC': ['FALSE',
                               'TRUE']
                     }

    # fill environment stats
    env_stats = {'ARITHMETIC': dict(),
                 'LOGIC':
                     {'FALSE': {'min': False, 'max': False},
                      'TRUE': {'min': True, 'max': True}
                      }
                 }
    for key in env_variables['ARITHMETIC']:
        env_stats['ARITHMETIC'][key] = {'min': 0.0, 'max': 1.0}

    # fill the environment
    env = {'ARITHMETIC': {}, 'LOGIC': {'FALSE': False, 'TRUE': True}}
    for key in env_variables['ARITHMETIC']:
        env['ARITHMETIC'][key] = features[:, features_header.index(key)]

    np.random.seed(0)
    bot = create_bot(0, 10, 12, env_variables)
    copy_bot = copy.deepcopy(bot)
    t = time.time()
    err = np.sum(np.square((player1 - bot.eval_all(env))))
    t = time.time() - t
    print(err, t)

    bot.bloat_analysis(env_stats)
    t = time.time()
    err = np.sum(np.square((player1 - bot.eval_all(env))))
    t = time.time() - t
    print(err, t)

    bot.unmark_bloat()
    t = time.time()
    err = np.sum(np.square((player1 - bot.eval_all(env))))
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

