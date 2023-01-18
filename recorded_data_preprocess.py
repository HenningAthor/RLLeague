"""
Script to convert all .parquet files into .csv files.
Also, the scripts cleans the inputs and converts the datatype to be homogenous.
It will:
    - change the unique player_ids to player1 and player2
        - player1 will always have team_num 0 and player2 will always have team_num 1
    - drop the following columns:
        - ticks_since_last_transmit
        - .../car_id
        - .../team_num
        - .../match_goals
        - .../match_saves
        - .../match_shots
        - .../match_demos
        - .../match_pickups
        - .../name
    - convert all other columns to float32 (not float64 because it uses much more memory)
Note that the column ordering is not defined (you have to read the header in each csv to know the columns variable).

This script takes around 2h / #num_procs to complete and generates about 150GB of data.
"""
import multiprocessing
import re
import glob
import os

import pandas as pd
from tqdm import tqdm


def process_file(file_path):
    file_name = os.path.basename(file_path)[:-8]

    df = pd.read_parquet(file_path)
    column_names = list(df.columns)

    # search player_ids
    player_ids = []
    player_team_num = {}
    team_num_to_player = {0: 'player1', 1: 'player2'}
    for name in column_names:
        if re.search(r'\d+(?:\.\d+)?/car_id', name):
            player_ids.append(name[:-7])

    player_team_num[player_ids[0]] = df[f'{player_ids[0]}/team_num'].values[0]
    player_team_num[player_ids[1]] = df[f'{player_ids[1]}/team_num'].values[0]

    # change player_ids to player1 and player2
    for i in range(len(column_names)):
        if player_ids[0] in column_names[i]:
            column_names[i] = column_names[i].replace(player_ids[0], team_num_to_player[player_team_num[player_ids[0]]])

        if player_ids[1] in column_names[i]:
            column_names[i] = column_names[i].replace(player_ids[1], team_num_to_player[player_team_num[player_ids[1]]])

    # change column names
    current_column_names = list(df.columns)
    rename_dict = dict(zip(current_column_names, column_names))
    df = df.rename(columns=rename_dict)

    # drop columns
    columns_to_remove = ['ticks_since_last_transmit',
                         'player1/car_id',
                         'player1/team_num',
                         'player1/match_goals',
                         'player1/match_saves',
                         'player1/match_shots',
                         'player1/match_demos',
                         'player1/match_pickups',
                         'player1/name',
                         'player2/car_id',
                         'player2/team_num',
                         'player2/match_goals',
                         'player2/match_saves',
                         'player2/match_shots',
                         'player2/match_demos',
                         'player2/match_pickups',
                         'player2/name']
    column_names = list(df.columns)
    for name in columns_to_remove:
        if name in column_names:
            # only drop names, which are present
            df = df.drop(columns=[name])
        else:
            print(f'Error removing {name}')

    # convert all columns to float64
    df = df.astype('float32')

    # save to csv
    df.to_csv(f'recorded_data/downloaded_matches/{file_name}.csv', index=False)


if __name__ == '__main__':
    file_list = []
    file_list.extend(list(glob.glob('recorded_data/out/*.parquet')))
    file_list.extend(list(glob.glob('recorded_data/out2/*.parquet')))

    # use all process
    with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
        pool.map(process_file, file_list)
