"""
File which specifies all available reward functions. The functions are categorized into two sets.
1. Step reward functions: The reward is given at each step of the game to the agent.
2. Game reward functions: The reward is given at the end of the game.

This file also specifies which rewards are used for which league.
"""
from typing import List

import numpy as np
from numpy import ndarray
from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_NET_Y, BACK_WALL_Y, BALL_RADIUS, \
    ORANGE_TEAM, CAR_MAX_SPEED
from rlgym.utils.gamestates import GameState

"""
Functions are inspired by reward functions of https://github.com/lucas-emery/rocket-league-gym
"""

"""
1. Step Reward functions
"""

def get_enemy_goal_pos(agent_idx: int, state: GameState) -> ndarray:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :return: position of enemy goal
    """
    if state.players[agent_idx].team_num == BLUE_TEAM:
        return np.array(ORANGE_GOAL_BACK)
    else:
        return np.array(BLUE_GOAL_BACK)


def constant_reward_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Gives a constant reward of 1.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    return 1


def boost_amount_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: the square root of the players boost amount (the lower the boost, the higher the negative impact)
    """

    return np.sqrt(state.players[agent_idx].boost_amount)


def boost_difference_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward is given if boost of current state is different to previous state (boost gained or used)

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: difference of boost_amount_step_reward() of previous and
             current state (reward in range [0.0, 1.0])
    """

    prev_state = state_history[-1]
    prev_boost = prev_state.players[agent_idx].boost_amount
    curr_boost = state.players[agent_idx].boost_amount

    if prev_boost != curr_boost:
        return abs(boost_amount_step_reward(agent_idx, state) - boost_amount_step_reward(agent_idx, prev_state))
    else:
        return 0


# TODO Wie bzw. soll hinzugefügt werden, das Reward geringer, wenn Ball öfter berührt, aber kein Tor fällt? (Dribbeln wird schlechter gemacht)
def ball_touched_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None, aerial_weight: float = 0.0) -> float:
    """
    :param aerial_weight: value in range [0.0, 1.0] (importance of ball height during touch)
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: if ball is touched, a reward in range [0.0, 1.0]...
    """

    if state.players[agent_idx].ball_touched:
        return ((state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)) ** aerial_weight
    else:
        return 0

# Sollte zusätzlich überprüft werden ob der Demo sinnvoll war? oder regelt sich das durch z.B. ein Gegentor selbst?
# TODO überarbeiten, player2 sind aktuell beide Spieler
def demo_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    This is a placeholder function for a step reward function.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    for player2 in state.players:
        if player2.is_demoed:
            return 1
        else:
            return 0


def distance_player_ball_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Calculate distance between car and ball (minus ball radius, because center of ball not reachable).

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: reward in range [0.0, 1.0] (result is exponential curve -> is higher,
             the closer the car is to the ball)
    """

    dist = np.linalg.norm(state.players[agent_idx].car_data.position - state.ball.position) - BALL_RADIUS

    return np.exp(-0.5 * dist / CAR_MAX_SPEED)


def distance_ball_enemy_goal_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Get position of enemy goal.
    Calculate distance between ball and enemy goal (position of goal is set from center to back,
    so distance could never be zero).

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: reward in range [0.0, 1.0] (result is exponential curve -> is higher,
             the closer the ball is to the enemy goal)
    """

    enemy_goal_pos = get_enemy_goal_pos(agent_idx, state)

    # Compensate for moving objective to back of net
    dist = np.linalg.norm(state.ball.position - enemy_goal_pos) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)

    return np.exp(-0.5 * dist / BALL_MAX_SPEED)


def facing_ball_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Calculates the vector of the player to the car.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: rotation of the front of the player to the ball, as reward in range [0.0, 1.0]
    """

    pos_diff = state.ball.position - state.players[agent_idx].car_data.position
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)

    return float(np.dot(state.players[agent_idx].car_data.forward(), norm_pos_diff))

# TODO Exception wenn Gewichte zusammen über 1.0
def align_ball_goal_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None,
                                defense_weight: float = 0.5, offense_weight: float = 0.5) -> float:
    """
    Gets positions of ball, player, and both goals.
    Calculates with the offensive and defensive weights the similarity of the vectors
    player <-> ball and player <-> net.

    :param offense_weight: value in range [0.0, 1.0] (importance of the offensive reward,
           should be in sum with defensive weight -> 1.0)
    :param defense_weight: value in range [0.0, 1.0] (importance of the defensive reward ,
           should be in sum with offensive weight -> 1.0)
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: sum of offensive and defensive reward
    """

    ball = state.ball.position
    pos = state.players[agent_idx].car_data.position
    protect_goal_pos = np.array(BLUE_GOAL_BACK)
    attack_goal_pos = np.array(ORANGE_GOAL_BACK)

    if state.players[agent_idx].team_num == ORANGE_TEAM:
        protect_goal_pos, attack_goal_pos = attack_goal_pos, protect_goal_pos

    # Align player->ball and net->player vectors
    defensive_reward = defense_weight * math.cosine_similarity(ball - pos,
                                                             pos - protect_goal_pos)

    # Align player->ball and player->net vectors
    offensive_reward = offense_weight * math.cosine_similarity(ball - pos,
                                                             attack_goal_pos - pos)

    return defensive_reward + offensive_reward


def closest_to_ball_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: if player closer to ball, return 1, else 0
    """

    player1 = state.players[agent_idx]
    dist = np.linalg.norm(player1.car_data.position - state.ball.position)

    for player2 in state.players:
        dist2 = np.linalg.norm(player2.car_data.position - state.ball.position)

        if dist2 < dist:
            return 0

    return 1


def touched_ball_last_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: if player touched ball last, return 1, else 0
    """
    if state.last_touch == agent_idx:
        return 1
    else:
        return 0


# TODO vllt. inverted Position des Balls
def behind_ball_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: if distance of ball to enemy backwall closer than player to backwall, return 1, else 0
    """
    player = state.players[agent_idx]

    if player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] or \
            player.team_num == ORANGE_TEAM and player.car_data.position[1] > state.ball.position[1]:
        return 1
    else:
        return 0


def velocity_player_to_ball_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Calculates velocity of the player and sets it in relation to the maximum possible velocity.
    Calculates the vector of the player to the car.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: velocity of player in direction to ball, as reward in range [0.0, 1.0]
    """

    player = state.players[agent_idx]
    vel = player.car_data.linear_velocity
    vel /= CAR_MAX_SPEED

    pos_diff = state.ball.position - player.car_data.position
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)

    return float(np.dot(norm_pos_diff, vel))


def velocity_ball_to_goal_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Determines first which goal is the one of the enemy.
    Calculates velocity of the ball and sets it in relation to the maximum possible velocity.
    Calculates the vector of the ball to the goal.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: velocity of ball in direction to goal, as reward in range [0.0, 1.0]
    """

    enemy_goal_pos = get_enemy_goal_pos(agent_idx, state)

    vel = state.ball.linear_velocity
    vel /= BALL_MAX_SPEED

    pos_diff = enemy_goal_pos - state.ball.position
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)

    return float(np.dot(norm_pos_diff, vel))


def velocity_player_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: the velocity of the player as reward in range [0.0, 1.0]
    """

    return np.linalg.norm(state.players[agent_idx].car_data.linear_velocity) / CAR_MAX_SPEED
    # why this "* (1 - 2 * self.negative)" -> negativ is false = 0 or true = 1
    # return np.linalg.norm(state.players[agent_idx].car_data.linear_velocity) / CAR_MAX_SPEED * (1 - 2 * self.negative)


# ist die Velocity nach vorne an Stelle 0? oder woanders?
def forward_velocity_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: the velocity in forward direction in range [0.0, 1.0] (punishment for backwards velocity)
    """
    player = state.players[agent_idx]

    return player.car_data.linear_velocity[0] * (np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED)


def distance_ball_to_own_backwall_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None, exponent: int = 1) -> float:
    """
    Calculates the ball distance to the backwall of the player.

    :param exponent: should be odd so that negative y -> negative reward
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: reward in range [0.0, 1.0]
    """

    if state.players[agent_idx].team_num == BLUE_TEAM:
        return (state.ball.position[1] / (
                    BACK_WALL_Y + BALL_RADIUS)) ** exponent
    else:
        return (state.inverted_ball.position[1] / (
                    BACK_WALL_Y + BALL_RADIUS)) ** exponent


# TODO inverted Ball Position wenn anderes Team
#def time_ball_enemy_half_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
#    """
#    This is a placeholder function for a step reward function.
#
#    :param agent_idx: The index of the agent in state.
#    :param state: The state of the game.
#    :param state_history: (optional) The previous states of the game.
#    :return: float
#    """
#
#    if state.ball.position[1] > 0:
#        return 1
#    else:
#        return 0

# auch hier "* (1 - 2 * self.negative)"?
def velocity_ball_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    This is a placeholder function for a step reward function.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """

    return np.linalg.norm(state.ball.linear_velocity) / BALL_MAX_SPEED

"""
2. Goal score Reward functions
"""

# TODO wenn ein Tor mehr bei einem Team, dann Reward geben
def goal_scored_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of 0.5 is given for a goal and a additional reward up to 0.5 depending on the ball speed when scoring.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: reward in range [0.0, 1.0]
    """

    ball_velocity = velocity_ball_to_goal_step_reward(agent_idx, state)

    return 0.5 + (0.5 * ball_velocity)

#def get_scored_on_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
#    """
#
#
#    :param agent_idx: The index of the agent in state.
#    :param state: The state of the game.
#    :param state_history: (optional) The previous states of the game.
#    :return: float
#    """
#
#    # negative score?
#
#    return 0

"""
3. Kickoff Reward functions
"""

# TODO Position[1] gleich 0 und Geschwindigkeit gleich 0 (Norm bei Geschwindigkeit)
def kickoff_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: if ball position is zero (kickoff), return velocity_player_to_ball_step_reward(), else 0
    """

    if state.ball.position[1] == 0:
        return velocity_player_to_ball_step_reward(agent_idx, state)
    else:
        return 0


"""
4. Game Reward functions
"""


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
    return velocity_player_step_reward(agent_idx, state, state_history)

def league_2_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of each step for agents in league 1.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """
    return velocity_player_step_reward(agent_idx, state, state_history)


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
