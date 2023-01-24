"""
Script to train one bot via evolution.
It will use a downloaded match to test the performance of the bot.
The bot will be mutated 100 times and the mutated bot with the smallest error is
chosen next for mutation. This process is repeated 100 times.
"""
import numpy as np

from bot.bot import recombine_bots
from genetic_lab.bot_generation import create_bot
from recorded_data.data_util import load_match, split_data, load_min_max_csv, scale_with_min_max

if __name__ == '__main__':
    # load the game data
    game_data, headers = load_match(f'recorded_data/out/a7303c84-7d9e-472d-aa60-403388827a41.parquet')

    # load min-max data
    min_max_data, min_max_headers = load_min_max_csv()

    # normalize game data
    game_data = scale_with_min_max(game_data, headers, min_max_data, min_max_headers)

    # split the data
    features, player1, player2, features_header, player1_header, player2_header = split_data(game_data, headers)

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
                               'player2/has_flip']
                     }

    # fill environment stats
    env_stats = {'ARITHMETIC': dict(),
                 'LOGIC': dict()
                 }
    for key in env_variables['ARITHMETIC']:
        env_stats['ARITHMETIC'][key] = {'min': 0.0, 'max': 1.0}
    for key in env_variables['LOGIC']:
        env_stats['LOGIC'][key] = {'min': False, 'max': True}

    # fill the environment
    env = {'ARITHMETIC': dict(), 'LOGIC': dict()}
    for key in env_variables['ARITHMETIC']:
        env['ARITHMETIC'][key] = features[:, features_header.index(key)]
    for key in env_variables['LOGIC']:
        env['LOGIC'][key] = np.array(features[:, features_header.index(key)], dtype=int)

    # create the bots
    bot_list = []
    err_list = []
    n_bots = 200
    mutate_chance = 0.01

    for i in range(n_bots):
        bot = create_bot(0, 10, 12, env_variables)
        bot_list.append(bot)
        err_list.append(0.0)
    print('Bots created')

    for bot in bot_list:
        bot.bloat_analysis(env_stats)
        bot.connect_trees()

    # start learning
    epochs = 100
    min_err_list = []
    err_mask = np.array([1, 1, 0, 0, 0, 1, 1, 1])  # we only predict those items

    for epoch in range(epochs):
        for i in range(len(bot_list)):
            res = bot_list[i].eval_all(env)
            err = (player1 - bot_list[i].eval_all(env)) * err_mask
            err_list[i] += np.average(np.square(err))
            bot_list[i].unmark_bloat()

        print('Errors', err_list)

        sorted_idx = sorted(range(len(err_list)), key=lambda x: err_list[x])
        bot_list = [bot_list[i] for i in sorted_idx]
        err_list = [err_list[i] for i in sorted_idx]

        min_bot = bot_list[0]
        min_err_list.append(err_list[0])
        new_bot_list = [min_bot]
        new_err_list = [0.0]

        # weights for recombination
        err_list_inverted = [1/err for err in err_list]
        error_sum = sum(err_list_inverted)
        weights = [error/error_sum for error in err_list_inverted]

        # mutate the best bot
        for i in range(34*2):
            mutated_bot = min_bot.mutate(mutate_chance)
            new_bot_list.append(mutated_bot)
            new_err_list.append(0.0)

        # recombine the best bots
        for i in range(16*2):
            bot_1 = np.random.choice(bot_list, p=weights)
            bot_2 = np.random.choice(bot_list, p=weights)
            rec_bot_1, rec_bot_2 = recombine_bots(bot_1, bot_2, 0.3)
            new_bot_list.append(rec_bot_1)
            new_bot_list.append(rec_bot_2)
            new_err_list.append(0.0)
            new_err_list.append(0.0)

        # insert new bots
        for i in range(33*2):
            bot = create_bot(0, 7, 10, env_variables)
            new_bot_list.append(bot)
            new_err_list.append(0.0)

        bot_list = new_bot_list
        err_list = new_err_list

        for bot in bot_list:
            bot.bloat_analysis(env_stats)
            bot.connect_trees()

    print('Min Err each epoch', min_err_list)
