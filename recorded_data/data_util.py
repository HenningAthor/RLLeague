"""
File to implement convenience functions for data loading and manipulation.
"""
import csv
import re
from typing import Tuple, List, Union, Dict

import numpy as np
import pandas as pd
import glob


def load_all_parquet_paths() -> List[str]:
    """
    Returns all .parquet paths found in recorded_data/out and
    recorded_data/out2.

    :return: List of file paths.
    """
    file_paths = []
    file_paths.extend(list(glob.glob(f"recorded_data/out/*.parquet")))
    file_paths.extend(list(glob.glob(f"recorded_data/out2/*.parquet")))
    return file_paths


def load_match(file_path: str,
               columns_to_remove: Union[List[str], None] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load a match from a .parquet file. Also cleans the file:
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

    :param file_path: Path to the .parquet file.
    :param columns_to_remove: These columns will be removed.
    """
    if columns_to_remove is None:
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

    # read the parquet file
    df = pd.read_parquet(file_path)
    column_names = list(df.columns)

    # search player_ids and rename
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

    column_names = list(df.columns)
    for name in columns_to_remove:
        if name in column_names:
            # only drop names, which are present
            df = df.drop(columns=[name])
        else:
            print(f'Error removing {name}')

    df = df.astype(np.float64)

    data = df.to_numpy(copy=True)
    column_names = list(df.columns)

    return data, column_names


def split_data(game_data: np.ndarray,
               headers: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Splits the game data into three parts. The state of the game, the actions of
    player1 and the actions of player2.

    :param game_data: The game data in one large array.
    :param headers: Name of each column.
    :return: Tuple of arrays.
    """
    # get indices of player1 and player2
    player1_col = ["player1/action_throttle",
                   "player1/action_steer",
                   "player1/action_pitch",
                   "player1/action_yaw",
                   "player1/action_roll",
                   "player1/action_jump",
                   "player1/action_boosting",
                   "player1/action_handbrake"]
    player2_col = ["player2/action_throttle",
                   "player2/action_steer",
                   "player2/action_pitch",
                   "player2/action_yaw",
                   "player2/action_roll",
                   "player2/action_jump",
                   "player2/action_boosting",
                   "player2/action_handbrake"]

    player1_idxs = np.array([headers.index(col) for col in player1_col], dtype=int)
    player2_idxs = np.array([headers.index(col) for col in player2_col], dtype=int)
    feature_idxs = np.setdiff1d(np.arange(0, len(headers)), np.union1d(player1_idxs, player2_idxs))

    features = game_data[:, feature_idxs]
    player1 = game_data[:, player1_idxs]
    player2 = game_data[:, player2_idxs]

    features_header = [headers[i] for i in feature_idxs]
    player1_header = [headers[i] for i in player1_idxs]
    player2_header = [headers[i] for i in player2_idxs]

    return features, player1, player2, features_header, player1_header, player2_header


def load_min_max_csv() -> Tuple[np.ndarray, List[str]]:
    """
    Loads the min_max.csv in recorded_data/. File contains the minimum and
    maximum value of each variable. First row holds the maximum values and
    second row contains the minimum values.

    :return: Numpy array with two roes and the header of the file.
    """
    # load the min-max data
    csv_file_path = 'recorded_data/min_max.csv'
    min_max_data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=True)
    f = open(csv_file_path, 'r')
    reader = csv.reader(f)
    headers = list(next(reader, None))
    f.close()

    return min_max_data, headers


def scale_with_min_max(features: np.ndarray,
                       features_header: List[str],
                       min_max: np.ndarray,
                       min_max_header: List[str]) -> np.ndarray:
    """
    Scales the features with the min-max data into the range of [0, 1].

    :param features: Array holding un-normalized data.
    :param features_header: List containing the name of each feature column.
    :param min_max: Array holding the min-max data.
    :param min_max_header: List containing the name of each min-max column.
    :return: Normalized features.
    """
    for i, name in enumerate(features_header):
        idx = min_max_header.index(name)  # get index in min-max data
        min_val, max_val = min_max[1][idx], min_max[0][idx]

        # scale into [0, 1]
        features[:, i] = (features[:, i] - min_val) / (max_val - min_val)

        assert np.all((features[:, i] >= 0.0))
        assert np.all((features[:, i] <= 1.0))

    return features


def reorder_columns(game_data: np.ndarray,
                    headers: List[str],
                    order: List[str]) -> np.ndarray:
    """
    Reorders the columns of the game data to fit the new order.

    :param game_data: Date of the game.
    :param headers: Current order of the headers.
    :param order: New order of the columns.
    :return: The game data newly ordered.
    """
    permutation = [headers.index(col) for col in order]
    return game_data[:, permutation]


def concat_multiple_datasets(list_game_data: List[np.ndarray],
                             list_headers: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """
    Takes multiple loaded data files and converts them into one big file.

    :param list_game_data: List of game data.
    :param list_headers: List of headers for each game data.
    """
    game_data = list_game_data[0]
    headers = list_headers[0]

    for i in range(1, len(list_game_data)):
        data = reorder_columns(list_game_data[i], list_headers[i], headers)
        game_data = np.concatenate((game_data, data), axis=0)

    return game_data, headers


def generate_env_stats(env_variables: Dict[str, List[str]],
                       min_max_data: np.ndarray,
                       headers: List[str]) -> Dict[str, Dict[str, Dict[str, Union[float, bool]]]]:
    """
    Generates the environment statistics.

    :param env_variables: Variables in the environment.
    :param min_max_data: Array holding minimum and maximum value.
    :param headers: Name of the columns.
    :return: Dictionary holding the minimum and maximum value for each variable.
    """
    # fill environment stats
    env_stats = {'ARITHMETIC': dict(), 'LOGIC': dict()}
    for key in env_variables['ARITHMETIC']:
        idx = headers.index(key)
        env_stats['ARITHMETIC'][key] = {'min': min_max_data[1][idx], 'max': min_max_data[0][idx]}
    for key in env_variables['LOGIC']:
        idx = headers.index(key)
        env_stats['LOGIC'][key] = {'min': bool(min_max_data[1][idx]), 'max': bool(min_max_data[1][idx])}

    for key in env_variables['ARITHMETIC']:
        env_stats['ARITHMETIC'][key]['min'] = 0.0
        env_stats['ARITHMETIC'][key]['max'] = 1.0
    for key in env_variables['LOGIC']:
        env_stats['LOGIC'][key]['min'] = False
        env_stats['LOGIC'][key]['max'] = True

    return env_stats


def generate_env(env_variables: Dict[str, List[str]],
                 features: np.ndarray,
                 headers: List[str]) -> Dict[str, Dict[str, Union[float, bool, np.ndarray]]]:
    """
    Generates the environment statistics.

    :param env_variables: Variables in the environment.
    :param features: Array holding the features.
    :param headers: Name of the columns.
    :return: Dictionary holding the values for all variables.
    """
    env = {'ARITHMETIC': dict(), 'LOGIC': dict()}
    for key in env_variables['ARITHMETIC']:
        env['ARITHMETIC'][key] = features[:, headers.index(key)]
    for key in env_variables['LOGIC']:
        env['LOGIC'][key] = np.array(features[:, headers.index(key)], dtype=int)

    return env
