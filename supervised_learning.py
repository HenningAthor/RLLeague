import csv

import numpy as np
from numpy import genfromtxt

from genetic_lab.bot_generation import create_bot

if __name__ == '__main__':
    # load the game data
    game_data = genfromtxt('recorded_data/res.csv', delimiter=',', skip_header=True)
    f = open('recorded_data/res.csv', 'r')
    reader = csv.reader(f)
    headers = list(next(reader, None))
    f.close()

    # split data into features and labels
    temp = np.split(game_data, [len(headers) - 16], axis=1)
    features = temp[0]
    s = np.split(temp[1], [8], axis=1)
    labels = s[0]
    features_header = headers[:-16]
    labels_header = headers[-16:-8]

    # remove ticks_since_last_transmit from features
    features = features[:, 1:]
    features_header = features_header[1:]

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

    bot_list = []
    err_list = []
    n_bots = 100
    # create a bot
    bot = create_bot(0, 5, 10, env_variables)
    bot_list.append(bot)
    err_list.append(0.0)
    for i in range(n_bots-1):
        mutated_bot = bot.mutate(0.05)
        bot_list.append(mutated_bot)
        err_list.append(0.0)

    epochs = 100
    min_err_list = []
    err_mask = np.array([1, 1, 0, 0, 0, 1, 1, 1])  # we only predict those items

    env = {'ARITHMETIC': {}, 'LOGIC': {'FALSE': False, 'TRUE': True}}
    for key in env_variables['ARITHMETIC']:
        env['ARITHMETIC'][key] = features[:, env_index['ARITHMETIC'][key]]

    for epoch in range(epochs):
        for i in range(len(bot_list)):
            res = bot_list[i].eval_all(env)
            err = (labels - bot_list[i].eval_all(env)) * err_mask
            err_list[i] += np.sum(np.square(err))

        print('Errors', err_list)

        # determine bot with minimal error
        min_err = min(err_list)
        min_idx = err_list.index(min_err)
        min_bot = bot_list[min_idx]
        min_err_list.append(min_err)

        # take the best bot and mutate it
        bot_list = [min_bot]
        err_list = [0.0]
        for i in range(n_bots-1):
            mutated_bot = min_bot.mutate(0.05)
            bot_list.append(mutated_bot)
            err_list.append(0.0)

    print('Min Err each epoch', min_err_list)
