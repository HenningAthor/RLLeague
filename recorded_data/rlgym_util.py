from typing import List

import numpy as np

from rlgym.utils.gamestates import GameState


def state_to_feature_vector(state: GameState) -> np.ndarray:
    """
    Converts a game-state to a feature vector. Use the function
    get_headers_for_feature_vector() to get the header for each entry.

    :param state: The game state.
    :return: Feature vector
    """
    features = np.zeros((1, 119), dtype=float)

    # game
    features[0, 0] = state.blue_score
    features[0, 1] = state.orange_score
    features[0, 2:36] = state.boost_pads

    # ball
    features[0, 36:39] = state.ball.position
    features[0, 39:42] = state.ball.linear_velocity
    features[0, 42:45] = state.ball.angular_velocity
    features[0, 45:48] = state.inverted_ball.position
    features[0, 48:51] = state.inverted_ball.linear_velocity
    features[0, 51:54] = state.inverted_ball.angular_velocity

    # player1
    features[0, 54:57] = state.players[0].car_data.position
    features[0, 57:60] = state.players[0].car_data.linear_velocity
    features[0, 60:63] = state.players[0].car_data.angular_velocity
    features[0, 63:67] = state.players[0].car_data.quaternion
    features[0, 67:70] = state.players[0].inverted_car_data.position
    features[0, 70:73] = state.players[0].inverted_car_data.linear_velocity
    features[0, 73:76] = state.players[0].inverted_car_data.angular_velocity
    features[0, 76:80] = state.players[0].inverted_car_data.quaternion
    features[0, 80] = state.players[0].is_demoed
    features[0, 81] = state.players[0].on_ground
    features[0, 82] = state.players[0].ball_touched
    features[0, 83] = state.players[0].has_jump
    features[0, 84] = state.players[0].has_flip
    features[0, 85] = state.players[0].boost_amount

    # player2
    features[0, 86:89] = state.players[1].car_data.position
    features[0, 89:92] = state.players[1].car_data.linear_velocity
    features[0, 92:95] = state.players[1].car_data.angular_velocity
    features[0, 95:99] = state.players[1].car_data.quaternion
    features[0, 99:102] = state.players[1].inverted_car_data.position
    features[0, 102:105] = state.players[1].inverted_car_data.linear_velocity
    features[0, 105:108] = state.players[1].inverted_car_data.angular_velocity
    features[0, 108:112] = state.players[1].inverted_car_data.quaternion
    features[0, 112] = state.players[1].is_demoed
    features[0, 113] = state.players[1].on_ground
    features[0, 114] = state.players[1].ball_touched
    features[0, 115] = state.players[1].has_jump
    features[0, 116] = state.players[1].has_flip
    features[0, 117] = state.players[1].boost_amount

    return features


def get_headers_for_feature_vector() -> List[str]:
    """
    This function returns the list of headers for the feature vector from
    state_to_feature_vector(). If state_to_feature_vector() changes this
    function has to change accordingly.

    :return: List of headers
    """
    return ['blue_score',
            'orange_score',
            'pad_0',
            'pad_1',
            'pad_2',
            'pad_3',
            'pad_4',
            'pad_5',
            'pad_6',
            'pad_7',
            'pad_8',
            'pad_9',
            'pad_10',
            'pad_11',
            'pad_12',
            'pad_13',
            'pad_14',
            'pad_15',
            'pad_16',
            'pad_17',
            'pad_18',
            'pad_19',
            'pad_20',
            'pad_21',
            'pad_22',
            'pad_23',
            'pad_24',
            'pad_25',
            'pad_26',
            'pad_27',
            'pad_28',
            'pad_29',
            'pad_30',
            'pad_31',
            'pad_32',
            'pad_33',
            'ball/pos_x',
            'ball/pos_y',
            'ball/pos_z',
            'ball/vel_x',
            'ball/vel_y',
            'ball/vel_z',
            'ball/ang_vel_x',
            'ball/ang_vel_y',
            'ball/ang_vel_z',
            'inverted_ball/pos_x',
            'inverted_ball/pos_y',
            'inverted_ball/pos_z',
            'inverted_ball/vel_x',
            'inverted_ball/vel_y',
            'inverted_ball/vel_z',
            'inverted_ball/ang_vel_x',
            'inverted_ball/ang_vel_y',
            'inverted_ball/ang_vel_z',
            'player1/pos_x',
            'player1/pos_y',
            'player1/pos_z',
            'player1/quat_w',
            'player1/quat_x',
            'player1/quat_y',
            'player1/quat_z',
            'player1/vel_x',
            'player1/vel_y',
            'player1/vel_z',
            'player1/ang_vel_x',
            'player1/ang_vel_y',
            'player1/ang_vel_z',
            'inverted_player1/pos_x',
            'inverted_player1/pos_y',
            'inverted_player1/pos_z',
            'inverted_player1/quat_w',
            'inverted_player1/quat_x',
            'inverted_player1/quat_y',
            'inverted_player1/quat_z',
            'inverted_player1/vel_x',
            'inverted_player1/vel_y',
            'inverted_player1/vel_z',
            'inverted_player1/ang_vel_x',
            'inverted_player1/ang_vel_y',
            'inverted_player1/ang_vel_z',
            'player1/is_demoed',
            'player1/on_ground',
            'player1/ball_touched',
            'player1/has_jump',
            'player1/has_flip',
            'player1/boost_amount',
            'player2/pos_x',
            'player2/pos_y',
            'player2/pos_z',
            'player2/quat_w',
            'player2/quat_x',
            'player2/quat_y',
            'player2/quat_z',
            'player2/vel_x',
            'player2/vel_y',
            'player2/vel_z',
            'player2/ang_vel_x',
            'player2/ang_vel_y',
            'player2/ang_vel_z',
            'inverted_player2/pos_x',
            'inverted_player2/pos_y',
            'inverted_player2/pos_z',
            'inverted_player2/quat_w',
            'inverted_player2/quat_x',
            'inverted_player2/quat_y',
            'inverted_player2/quat_z',
            'inverted_player2/vel_x',
            'inverted_player2/vel_y',
            'inverted_player2/vel_z',
            'inverted_player2/ang_vel_x',
            'inverted_player2/ang_vel_y',
            'inverted_player2/ang_vel_z',
            'player2/is_demoed',
            'player2/on_ground',
            'player2/ball_touched',
            'player2/has_jump',
            'player2/has_flip',
            'player2/boost_amount']

