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
def ball_touched_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None,
                             aerial_weight: float = 0.0) -> float:
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
def demo_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: if enemy player is demoed, return 1, else 0
    """

    for player in state.players:
        if player != state.players[agent_idx] and player.is_demoed:
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


def distance_ball_enemy_goal_step_reward(agent_idx: int, state: GameState,
                                         state_history: List[GameState] = None) -> float:
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


def align_ball_goal_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None,
                                defense_weight: float = 0.5, offense_weight: float = 0.5) -> float:
    """
    Gets positions of ball, player, and both goals.
    Calculates with the offensive and defensive weights the similarity of the vectors'
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

    if (defense_weight + offense_weight) > 1.0:
        raise ValueError("Offensive and defensive weights should be in sum -> 1.0")

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


def velocity_player_to_ball_step_reward(agent_idx: int, state: GameState,
                                        state_history: List[GameState] = None) -> float:
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


def distance_ball_to_own_backwall_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None,
                                              exponent: int = 1) -> float:
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
# def time_ball_enemy_half_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
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
    :return: reward of 1 is enemy player demoed, else 0
    """

    return np.linalg.norm(state.ball.linear_velocity) / BALL_MAX_SPEED


"""
2. Goal score Reward functions
"""


def goal_scored_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of 0.5 is given for a goal and an additional reward up to 0.5 depending on the ball speed when scoring.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: reward in range [0.0, 1.0]
    """

    player_score = None
    prev_player_score = None

    if agent_idx == 0:
        player_score = state.blue_score
        prev_player_score = state_history[-1].blue_score
    elif agent_idx == 1:
        player_score = state.orange_score
        prev_player_score = state_history[-1].orange_score

    if player_score > prev_player_score:
        ball_velocity = velocity_ball_to_goal_step_reward(agent_idx, state)

        return 0.5 + (0.5 * ball_velocity)
    else:
        return 0


# def get_scored_on_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
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


def kickoff_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: if ball position and velocity is zero (kickoff), return velocity_player_to_ball_step_reward(), else 0
    """

    if state.ball.position[1] == 0 and np.linalg.norm(state.ball.linear_velocity) == 0:
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
    elif agent_idx == 1:
        if state.blue_score < state.orange_score:
            return 1
        else:
            return 0


"""
League reward functions.
"""
# dictionary of int: (callable, callable) (league_id: (step_reward, game_reward))
league_reward_functions = {}

"""
reward += constant_reward_step_reward(agent_idx, state, state_history)
reward += kickoff_reward(agent_idx, state, state_history)
reward += goal_scored_reward(agent_idx, state, state_history)
reward += boost_amount_step_reward(agent_idx, state, state_history)
reward += boost_difference_step_reward(agent_idx, state, state_history)
reward += ball_touched_step_reward(agent_idx, state, state_history)
reward += demo_step_reward(agent_idx, state, state_history)
reward += distance_player_ball_step_reward(agent_idx, state, state_history)
reward += distance_ball_enemy_goal_step_reward(agent_idx, state, state_history)
reward += facing_ball_step_reward(agent_idx, state, state_history)
reward += align_ball_goal_step_reward(agent_idx, state, state_history)
reward += closest_to_ball_step_reward(agent_idx, state, state_history)
reward += touched_ball_last_step_reward(agent_idx, state, state_history)
reward += behind_ball_step_reward(agent_idx, state, state_history)
reward += velocity_player_to_ball_step_reward(agent_idx, state, state_history)
reward += velocity_ball_to_goal_step_reward(agent_idx, state, state_history)
reward += velocity_player_step_reward(agent_idx, state, state_history)
reward += forward_velocity_step_reward(agent_idx, state, state_history)
reward += distance_ball_to_own_backwall_step_reward(agent_idx, state, state_history)
reward += velocity_ball_step_reward(agent_idx, state, state_history)
"""


def league_1_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of each step for agents in league 1.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """

    constant_rew = constant_reward_step_reward(agent_idx, state, state_history)
    kickoff_rew = 1.0 * kickoff_reward(agent_idx, state, state_history)
    goal_scored_rew = 1.0 * goal_scored_reward(agent_idx, state, state_history)

    step_rew = 0.0
    # step_rew += boost_amount_step_reward(agent_idx, state, state_history)
    # step_rew += boost_difference_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * ball_touched_step_reward(agent_idx, state, state_history)
    # step_rew += demo_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * distance_player_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * distance_ball_enemy_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * facing_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * align_ball_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * closest_to_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * touched_ball_last_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * behind_ball_step_reward(agent_idx, state, state_history)
    # step_rew += velocity_player_to_ball_step_reward(agent_idx, state, state_history)
    # step_rew += velocity_ball_to_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * velocity_player_step_reward(agent_idx, state, state_history)
    step_rew += 0.1 * forward_velocity_step_reward(agent_idx, state, state_history)
    # step_rew += distance_ball_to_own_backwall_step_reward(agent_idx, state, state_history)
    # step_rew += velocity_ball_step_reward(agent_idx, state, state_history)

    return constant_rew + kickoff_rew + goal_scored_rew + step_rew


def league_2_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of each step for agents in league 2.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """

    constant_rew = constant_reward_step_reward(agent_idx, state, state_history)
    kickoff_rew = 1.25 * kickoff_reward(agent_idx, state, state_history)
    goal_scored_rew = 1.25 * goal_scored_reward(agent_idx, state, state_history)

    step_rew = 0.0
    # step_rew += boost_amount_step_reward(agent_idx, state, state_history)
    # step_rew += boost_difference_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * ball_touched_step_reward(agent_idx, state, state_history)
    # step_rew += demo_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * distance_player_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * distance_ball_enemy_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * facing_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * align_ball_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * closest_to_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * touched_ball_last_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * behind_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * velocity_player_to_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * velocity_ball_to_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * velocity_player_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * forward_velocity_step_reward(agent_idx, state, state_history)
    # step_rew += distance_ball_to_own_backwall_step_reward(agent_idx, state, state_history)
    step_rew += 0.077 * velocity_ball_step_reward(agent_idx, state, state_history)

    return constant_rew + kickoff_rew + goal_scored_rew + step_rew


def league_3_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of each step for agents in league 2.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """

    constant_rew = constant_reward_step_reward(agent_idx, state, state_history)
    kickoff_rew = 1.5 * kickoff_reward(agent_idx, state, state_history)
    goal_scored_rew = 1.5 * goal_scored_reward(agent_idx, state, state_history)

    step_rew = 0.0
    step_rew += 0.0666 * boost_amount_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * boost_difference_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * ball_touched_step_reward(agent_idx, state, state_history)
    # step_rew += demo_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * distance_player_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * distance_ball_enemy_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * facing_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * align_ball_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * closest_to_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * touched_ball_last_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * behind_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * velocity_player_to_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * velocity_ball_to_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * velocity_player_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * forward_velocity_step_reward(agent_idx, state, state_history)
    # step_rew += distance_ball_to_own_backwall_step_reward(agent_idx, state, state_history)
    step_rew += 0.0666 * velocity_ball_step_reward(agent_idx, state, state_history)

    return constant_rew + kickoff_rew + goal_scored_rew + step_rew


def league_4_step_reward(agent_idx: int, state: GameState, state_history: List[GameState] = None) -> float:
    """
    Reward of each step for agents in league 2.

    :param agent_idx: The index of the agent in state.
    :param state: The state of the game.
    :param state_history: (optional) The previous states of the game.
    :return: float
    """

    constant_rew = constant_reward_step_reward(agent_idx, state, state_history)
    kickoff_rew = 1.75 * kickoff_reward(agent_idx, state, state_history)
    goal_scored_rew = 1.75 * goal_scored_reward(agent_idx, state, state_history)

    step_rew = 0.0
    step_rew += 0.0588 * boost_amount_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * boost_difference_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * ball_touched_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * demo_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * distance_player_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * distance_ball_enemy_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * facing_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * align_ball_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * closest_to_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * touched_ball_last_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * behind_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * velocity_player_to_ball_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * velocity_ball_to_goal_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * velocity_player_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * forward_velocity_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * distance_ball_to_own_backwall_step_reward(agent_idx, state, state_history)
    step_rew += 0.0588 * velocity_ball_step_reward(agent_idx, state, state_history)

    return constant_rew + kickoff_rew + goal_scored_rew + step_rew


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
league_reward_functions[2] = (league_2_step_reward, league_1_game_reward)
league_reward_functions[3] = (league_3_step_reward, league_1_game_reward)
league_reward_functions[4] = (league_4_step_reward, league_1_game_reward)
