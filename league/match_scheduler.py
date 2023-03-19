import datetime
import os
import pickle
import random
from typing import Union, List

import gym.spaces
import numpy as np
import win32con
import win32gui

from rlgym.envs import Match
from rlgym.utils import StateSetter
from rlgym.utils.action_parsers import ContinuousAction
from rlgym.utils.gamestates import GameState
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters.state_wrapper import StateWrapper
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3.ppo import PPO

from league.reward_functions import league_reward_functions
from recorded_data.data_util import load_min_max_csv, scale_with_min_max
from recorded_data.rlgym_util import state_to_feature_vector, get_headers_for_feature_vector

def reward_movement(idx: int, state: GameState):
    return np.linalg.norm(state.players[idx].inverted_car_data.linear_velocity)


class RLLeagueState(StateSetter):
    SPAWN_BLUE_POS = [[0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.5 * np.pi]
    SPAWN_ORANGE_POS = [[0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.5 * np.pi]

    def __init__(self, rl_league_action: 'RLLeagueAction', verbose=False):
        super().__init__()
        self.rl_league_action = rl_league_action

        self.verbose = verbose

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self.rl_league_action.save_match_report()
        self.rl_league_action.load_match_from_file()
        # possible kickoff indices are shuffled
        spawn_inds = [0]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        for car in state_wrapper.cars:
            pos = [0, 0, 0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)


class RLLeagueAction(ContinuousAction):
    """
    Continuous Action space, that also accepts a few other input formats for QoL reasons and to remain
    compatible with older versions.
    """

    def __init__(self, action_id, verbose=False):
        """
        Initializes an ActionParser which controls agents with id1 and id2.
        """
        super().__init__()

        self.action_id = action_id
        self.initialized = False
        self.league_id = -1
        self.id_1 = -1
        self.id_2 = -1
        self.agent_1 = None
        self.agent_2 = None

        self.feature_header = get_headers_for_feature_vector()
        self.min_max_data, self.min_max_headers = load_min_max_csv()

        self.state_history = []
        self.rew_1 = 0.0
        self.rew_2 = 0.0
        self.step_reward_func = None
        self.game_reward_func = None

        self.verbose = verbose

    def load_match_from_file(self) -> None:
        """
        Loads a match from a file.

        :return: None
        """
        file_path = f"league/pending_matches/{self.action_id}.txt"
        if not os.path.exists(file_path):
            self.initialized = False
            if self.verbose: print(f"RLLeagueAction {self.action_id}: Could not load agents. No file '{file_path}' found!")
            return

        f = open(file_path, "r")
        args = f.read().replace('\n', '').split(',')
        assert len(args) == 3
        self.league_id = int(args[0])
        self.id_1 = int(args[1])
        self.id_2 = int(args[2])

        if (self.league_id, self.id_1, self.id_2) == (-1, -1, -1):
            self.agent_1 = None
            self.agent_2 = None
            self.rew_1 = 0.0
            self.rew_2 = 0.0
            self.initialized = False
            if self.verbose: print(f"RLLeagueAction {self.action_id}: Loaded Agent {self.id_1} and {self.id_2} in League {self.league_id} (Default)")
            return

        path_1 = f"agent_storage/agent_{self.id_1}/agent_{self.id_1}.pickle"
        path_2 = f"agent_storage/agent_{self.id_2}/agent_{self.id_2}.pickle"
        assert os.path.exists(path_1)
        assert os.path.exists(path_2)

        self.agent_1 = pickle.load(open(path_1, 'rb'))
        self.agent_2 = pickle.load(open(path_2, 'rb'))
        self.rew_1 = 0.0
        self.rew_2 = 0.0
        self.step_reward_func, self.game_reward_func = league_reward_functions[self.league_id]
        self.initialized = True
        if self.verbose: print(f"RLLeagueAction {self.action_id}: Loaded agent {self.id_1} and {self.id_2} in league {self.league_id}")

    def get_action_space(self) -> gym.spaces.Space:
        return super().get_action_space()

    def parse_actions(self, actions: Union[np.ndarray, List[np.ndarray], List[float]], state: GameState) -> np.ndarray:
        if not self.initialized:
            return np.zeros(actions.shape)

        features = state_to_feature_vector(state)
        features = scale_with_min_max(features, self.feature_header, self.min_max_data, self.min_max_headers)

        r1 = self.step_reward_func(0, state)
        r2 = self.step_reward_func(1, state)

        self.rew_1 += 1.0  # r1
        self.rew_2 += 1.0  # r2

        # player 1 has index 0, player 2 has index 1
        env_1 = {'ARITHMETIC': {'ball/pos_x': features[0, self.feature_header.index('ball/pos_x')],
                                'ball/pos_y': features[0, self.feature_header.index('ball/pos_y')],
                                'ball/pos_z': features[0, self.feature_header.index('ball/pos_z')],
                                'ball/vel_x': features[0, self.feature_header.index('ball/vel_x')],
                                'ball/vel_y': features[0, self.feature_header.index('ball/vel_y')],
                                'ball/vel_z': features[0, self.feature_header.index('ball/vel_z')],
                                'ball/ang_vel_x': features[0, self.feature_header.index('ball/ang_vel_x')],
                                'ball/ang_vel_y': features[0, self.feature_header.index('ball/ang_vel_y')],
                                'ball/ang_vel_z': features[0, self.feature_header.index('ball/ang_vel_z')],
                                'player1/pos_x': features[0, self.feature_header.index('player1/pos_x')],
                                'player1/pos_y': features[0, self.feature_header.index('player1/pos_y')],
                                'player1/pos_z': features[0, self.feature_header.index('player1/pos_z')],
                                'player1/vel_x': features[0, self.feature_header.index('player1/vel_x')],
                                'player1/vel_y': features[0, self.feature_header.index('player1/vel_y')],
                                'player1/vel_z': features[0, self.feature_header.index('player1/vel_z')],
                                'player1/ang_vel_x': features[0, self.feature_header.index('player1/ang_vel_x')],
                                'player1/ang_vel_y': features[0, self.feature_header.index('player1/ang_vel_y')],
                                'player1/ang_vel_z': features[0, self.feature_header.index('player1/ang_vel_z')],
                                'player1/quat_w': features[0, self.feature_header.index('player1/quat_w')],
                                'player1/quat_x': features[0, self.feature_header.index('player1/quat_x')],
                                'player1/quat_y': features[0, self.feature_header.index('player1/quat_y')],
                                'player1/quat_z': features[0, self.feature_header.index('player1/quat_z')],
                                'inverted_player2/pos_x': features[0, self.feature_header.index('inverted_player2/pos_x')],
                                'inverted_player2/pos_y': features[0, self.feature_header.index('inverted_player2/pos_y')],
                                'inverted_player2/pos_z': features[0, self.feature_header.index('inverted_player2/pos_z')],
                                'inverted_player2/vel_x': features[0, self.feature_header.index('inverted_player2/vel_x')],
                                'inverted_player2/vel_y': features[0, self.feature_header.index('inverted_player2/vel_y')],
                                'inverted_player2/vel_z': features[0, self.feature_header.index('inverted_player2/vel_z')],
                                'inverted_player2/ang_vel_x': features[0, self.feature_header.index('inverted_player2/ang_vel_x')],
                                'inverted_player2/ang_vel_y': features[0, self.feature_header.index('inverted_player2/ang_vel_y')],
                                'inverted_player2/ang_vel_z': features[0, self.feature_header.index('inverted_player2/ang_vel_z')],
                                'inverted_player2/quat_w': features[0, self.feature_header.index('inverted_player2/quat_w')],
                                'inverted_player2/quat_x': features[0, self.feature_header.index('inverted_player2/quat_x')],
                                'inverted_player2/quat_y': features[0, self.feature_header.index('inverted_player2/quat_y')],
                                'inverted_player2/quat_z': features[0, self.feature_header.index('inverted_player2/quat_z')],
                                'player1/boost_amount': features[0, self.feature_header.index('player1/boost_amount')],
                                'player2/boost_amount': features[0, self.feature_header.index('player2/boost_amount')]
                                },
                 'LOGIC': {'player1/on_ground': features[0, self.feature_header.index('player1/on_ground')],
                           'player1/ball_touched': features[0, self.feature_header.index('player1/ball_touched')],
                           'player1/has_jump': features[0, self.feature_header.index('player1/has_jump')],
                           'player1/has_flip': features[0, self.feature_header.index('player1/has_flip')],
                           'player2/on_ground': features[0, self.feature_header.index('player2/on_ground')],
                           'player2/ball_touched': features[0, self.feature_header.index('player2/ball_touched')],
                           'player2/has_jump': features[0, self.feature_header.index('player2/has_jump')],
                           'player2/has_flip': features[0, self.feature_header.index('player2/has_flip')]
                           }
                 }

        # player 1 has index 1, player 2 has index 0
        env_2 = {'ARITHMETIC': {'ball/pos_x': features[0, self.feature_header.index('inverted_ball/pos_x')],
                                'ball/pos_y': features[0, self.feature_header.index('inverted_ball/pos_y')],
                                'ball/pos_z': features[0, self.feature_header.index('inverted_ball/pos_z')],
                                'ball/vel_x': features[0, self.feature_header.index('inverted_ball/vel_x')],
                                'ball/vel_y': features[0, self.feature_header.index('inverted_ball/vel_y')],
                                'ball/vel_z': features[0, self.feature_header.index('inverted_ball/vel_z')],
                                'ball/ang_vel_x': features[0, self.feature_header.index('inverted_ball/ang_vel_x')],
                                'ball/ang_vel_y': features[0, self.feature_header.index('inverted_ball/ang_vel_y')],
                                'ball/ang_vel_z': features[0, self.feature_header.index('inverted_ball/ang_vel_z')],
                                'player1/pos_x': features[0, self.feature_header.index('inverted_player2/pos_x')],
                                'player1/pos_y': features[0, self.feature_header.index('inverted_player2/pos_y')],
                                'player1/pos_z': features[0, self.feature_header.index('inverted_player2/pos_z')],
                                'player1/vel_x': features[0, self.feature_header.index('inverted_player2/vel_x')],
                                'player1/vel_y': features[0, self.feature_header.index('inverted_player2/vel_y')],
                                'player1/vel_z': features[0, self.feature_header.index('inverted_player2/vel_z')],
                                'player1/ang_vel_x': features[0, self.feature_header.index('inverted_player2/ang_vel_x')],
                                'player1/ang_vel_y': features[0, self.feature_header.index('inverted_player2/ang_vel_y')],
                                'player1/ang_vel_z': features[0, self.feature_header.index('inverted_player2/ang_vel_z')],
                                'player1/quat_w': features[0, self.feature_header.index('inverted_player2/quat_w')],
                                'player1/quat_x': features[0, self.feature_header.index('inverted_player2/quat_x')],
                                'player1/quat_y': features[0, self.feature_header.index('inverted_player2/quat_y')],
                                'player1/quat_z': features[0, self.feature_header.index('inverted_player2/quat_z')],
                                'inverted_player2/pos_x': features[0, self.feature_header.index('player1/pos_x')],
                                'inverted_player2/pos_y': features[0, self.feature_header.index('player1/pos_y')],
                                'inverted_player2/pos_z': features[0, self.feature_header.index('player1/pos_z')],
                                'inverted_player2/vel_x': features[0, self.feature_header.index('player1/vel_x')],
                                'inverted_player2/vel_y': features[0, self.feature_header.index('player1/vel_y')],
                                'inverted_player2/vel_z': features[0, self.feature_header.index('player1/vel_z')],
                                'inverted_player2/ang_vel_x': features[0, self.feature_header.index('player1/ang_vel_x')],
                                'inverted_player2/ang_vel_y': features[0, self.feature_header.index('player1/ang_vel_y')],
                                'inverted_player2/ang_vel_z': features[0, self.feature_header.index('player1/ang_vel_z')],
                                'inverted_player2/quat_w': features[0, self.feature_header.index('player1/quat_w')],
                                'inverted_player2/quat_x': features[0, self.feature_header.index('player1/quat_x')],
                                'inverted_player2/quat_y': features[0, self.feature_header.index('player1/quat_y')],
                                'inverted_player2/quat_z': features[0, self.feature_header.index('player1/quat_z')],
                                'player1/boost_amount': features[0, self.feature_header.index('player2/boost_amount')],
                                'player2/boost_amount': features[0, self.feature_header.index('player1/boost_amount')]
                                },
                 'LOGIC': {'player1/on_ground': features[0, self.feature_header.index('player2/on_ground')],
                           'player1/ball_touched': features[0, self.feature_header.index('player2/ball_touched')],
                           'player1/has_jump': features[0, self.feature_header.index('player2/has_jump')],
                           'player1/has_flip': features[0, self.feature_header.index('player2/has_flip')],
                           'player2/on_ground': features[0, self.feature_header.index('player1/on_ground')],
                           'player2/ball_touched': features[0, self.feature_header.index('player1/ball_touched')],
                           'player2/has_jump': features[0, self.feature_header.index('player1/has_jump')],
                           'player2/has_flip': features[0, self.feature_header.index('player1/has_flip')]}
                 }

        actions = np.zeros(actions.shape)
        actions[0] = self.agent_1.eval_all(env_1)
        actions[1] = self.agent_2.eval_all(env_2)

        self.state_history.append(state)

        return actions

    def save_match_report(self) -> None:
        """
        Saves the accumulated reward of the agents in a file.

        :return: None
        """
        path = f"game_reports/"
        if self.initialized:
            file_name = f"game_reports/{self.league_id}_{self.agent_1.agent_id}_{self.agent_2.agent_id}.txt"

            if not os.path.exists(path):
                os.makedirs(path)

            f = open(file_name, "w")
            f.write(f"{self.rew_1}\n{self.rew_2}")
            f.close()

            if self.verbose: print(f"RLLeagueAction {self.action_id}: Writing game report for agent {self.agent_1.agent_id} ({self.rew_1}) and {self.agent_2.agent_id} ({self.rew_2}) in league {self.league_id}")
        else:
            if self.verbose: print(f"RLLeagueAction {self.action_id}: Not initialized, no game report written!")

        self.rew_1 = 0.0
        self.rew_2 = 0.0


class MatchScheduler(object):
    def __init__(self, n_instances, wait_time=20, game_speed=100, minimize_windows=True, verbose=False, rlgym_verbose=False) -> None:
        """
        Initializes the MatchScheduler.

        :param n_instances: Number of instances, that should be executed simultaneously.
        :param wait_time: How long to wait before opening another rl window.
        :param game_speed: Speed of the game.
        :param minimize_windows: If the Rocket League windows should be minimized.
        :param verbose: If the class should print information.
        :param rlgym_verbose: If rlgym should print (not everything can be disabled).
        """
        # misc
        self.verbose = verbose
        self.minimize_windows = minimize_windows

        self.remove_matches()

        # rlgym information
        self.n_instances = n_instances
        self.rl_actions = []
        self.rl_state = []
        self.rl_matches = []

        for i in range(n_instances):
            action = RLLeagueAction(action_id=i, verbose=verbose)
            state = RLLeagueState(rl_league_action=action, verbose=verbose)
            match = Match(
                reward_function=DefaultReward(),
                terminal_conditions=[],
                obs_builder=DefaultObs(),
                action_parser=action,
                state_setter=state,
                game_speed=game_speed,
                spawn_opponents=True)

            self.rl_actions.append(action)
            self.rl_state.append(state)
            self.rl_matches.append(match)

        self.env = SB3MultipleInstanceEnv(match_func_or_matches=self.rl_matches, num_instances=self.n_instances, wait_time=wait_time)
        self.learner = PPO(policy="MlpPolicy", env=self.env, verbose=rlgym_verbose, n_epochs=1)

        # match information
        self.pending_matches = {}

        # misc
        self.verbose = verbose

    def simulate(self) -> None:
        """
        Simulates the current pending matches. Optimally there are a multiple
        of n_instances matches for each time_step pending to not waste
        processing power.

        :return: None
        """
        while self.pending_matches:
            if self.verbose: print(f"Simulating next set! Remaining games {sum([len(val) for val in self.pending_matches.values()])}")
            self.run_next_set()
        self.learner.learn(0)

    def run_next_set(self) -> None:
        """
        Runs the next set of rl_matches.

        :return:
        """
        time_step = self.distribute_matches()
        if self.minimize_windows:
            self.minimize_rl_windows()
        self.learner.learn(self.n_instances * time_step)
        self.remove_matches()

    def add_match(self, league: int, id1: int, id2: int, time_steps: int) -> None:
        """
        Adds a match to be simulated.

        :param league: The league of both agents.
        :param id1: Id of agent 1.
        :param id2: Id of agent 2.
        :param time_steps: Length of the game.
        :return: None
        """
        if time_steps not in self.pending_matches.keys():
            self.pending_matches[time_steps] = []

        self.pending_matches[time_steps].append((league, id1, id2))

    def distribute_matches(self) -> int:
        """
        Distributes the pending matches on the instances. Only matches with the
        same length, can be executed in parallel.

        :return: Length of the executed matches.
        """
        time_step = list(self.pending_matches.keys())[0]  # only matches with same length
        for i in range(self.n_instances):
            (league, id1, id2) = (-1, -1, -1)
            if self.pending_matches[time_step]:
                (league, id1, id2) = self.pending_matches[time_step].pop(0)

            f = open(f"league/pending_matches/{i}.txt", "w")
            f.write(f"{league},{id1},{id2}")
            f.close()

            if self.verbose: print(f"Distributing Match {(league, id1, id2)} to window {i} with time_step={time_step}")

        # remove entry from dict, if empty
        if not self.pending_matches[time_step]:
            del self.pending_matches[time_step]

        return time_step

    def remove_matches(self) -> None:
        """
        Removes all files from league/pending_matches.

        :return:
        """
        path = f"league/pending_matches/"
        if not os.path.exists(path):
            os.makedirs(path)

        for file_name in os.listdir(f"league/pending_matches/"):
            file = path + file_name
            # construct full file path
            if os.path.isfile(file):
                os.remove(file)

    def minimize_rl_windows(self) -> None:
        """
        Minimizes all open Rocket League windows.

        :return: None
        """
        print(f"---WARNING! minimize_rl_windows() NOT working. Not executing function!---")
        return
        look_for_another = True

        def enumHandler(hwnd, lParam):
            if win32gui.IsWindowVisible(hwnd):
                print(win32gui.GetWindowText(hwnd))
                if 'Rocket League' in win32gui.GetWindowText(hwnd):
                    global look_for_another
                    look_for_another = True
                    print('found rl window')
                    win32gui.ShowWindow(hwnd, win32con.SW_HIDE)

        while look_for_another:
            look_for_another = False
            win32gui.EnumWindows(enumHandler, None)
