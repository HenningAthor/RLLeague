"""
File which specifies all available reward functions. The functions are categorized into two sets.
1. Step reward functions: The reward is given at each step of the game to the agent.
2. Game reward functions: The reward is given at the end of the game.

This file also specifies which rewards are used for which league.
"""
from typing import List

import numpy as np
from rlgym.utils.gamestates import GameState

"""
1. Step Reward functions
"""


def base_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    This is a placeholder function for a step reward function.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    raise NotImplementedError


def movement_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward the agent for moving.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    return np.linalg.norm(state.players[agent_idx].inverted_car_data.linear_velocity)


"""
2. Game Reward functions
"""


def base_game_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    This is a placeholder function for a game reward function.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    raise NotImplementedError


def winning_game_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    This is a placeholder function for a game reward function.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    if agent_idx == 0:
        if state.blue_score > state.orange_score:
            return 1
        else:
            return 0

    if agent_idx == 1:
        if state.blue_score < state.orange_score:
            return 1
        else:
            return 0


"""
League reward functions.
"""
# dictionary of int: (callable, callable) (league_id: (step_reward, game_reward))
league_reward_functions = {}


def league_1_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of each step for agents in league 1.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    return movement_step_reward(agent_idx, state, state_history)


def league_1_game_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of each step for agents in league 1.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    return winning_game_reward(agent_idx, state, state_history)


league_reward_functions[1] = (league_1_step_reward, league_1_game_reward)
