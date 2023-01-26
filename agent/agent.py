"""
Implements the agent.
"""
import os
import pickle
import _pickle as cPickle
import random
from typing import Dict, Union, Tuple, List

import numpy as np
from numba import njit

from agent.nodes import all_branch_nodes, recombine_nodes
from agent.tree import Tree, recombine_trees
from agent.util import scale_discrete


class Agent(object):
    def __init__(self,
                 agent_id: int,
                 name: str,
                 min_depth: int,
                 max_depth: int,
                 env_vars: Dict[str, List[str]]) -> None:
        self.agent_id: int = agent_id
        self.name: str = name

        self.tree_names = ['Throttle', 'Steer', 'Pitch', 'Yaw',
                           'Roll', 'Jump', 'Boost', 'Handbrake']

        self.creation_variables: Dict[str, List[str]] = env_vars

        if not min_depth == max_depth == -1:
            # 0 - throttle, 1 - steer, 2 - pitch, 3 - yaw
            # 4 - roll, 5 - jump, 6 - boost, 7 - handbrake
            self.tree_list: List[Tree] = [Tree(min_depth, max_depth, env_vars, -1.0, 1.0, True),
                                          Tree(min_depth, max_depth, env_vars, -1.0, 1.0, True),
                                          Tree(min_depth, max_depth, env_vars, -1.0, 1.0, True),
                                          Tree(min_depth, max_depth, env_vars, -1.0, 1.0, True),
                                          Tree(min_depth, max_depth, env_vars, -1.0, 1.0, True),
                                          Tree(min_depth, max_depth, env_vars, 0.0, 1.0, True),
                                          Tree(min_depth, max_depth, env_vars, 0.0, 1.0, True),
                                          Tree(min_depth, max_depth, env_vars, 0.0, 1.0, True)]
        else:
            self.tree_list: List[Tree] = [Tree(-1, -1, env_vars, -1.0, 1.0, True),
                                          Tree(-1, -1, env_vars, -1.0, 1.0, True),
                                          Tree(-1, -1, env_vars, -1.0, 1.0, True),
                                          Tree(-1, -1, env_vars, -1.0, 1.0, True),
                                          Tree(-1, -1, env_vars, -1.0, 1.0, True),
                                          Tree(-1, -1, env_vars, 0.0, 1.0, True),
                                          Tree(-1, -1, env_vars, 0.0, 1.0, True),
                                          Tree(-1, -1, env_vars, 0.0, 1.0, True)]

    def __str__(self):
        return f"{self.name}"

    def __deepcopy__(self, memodict={}) -> 'Agent':
        """
        Deep-copies the agent.

        :param memodict: Dictionary to save already seen variables.
        :return: Deepcopy of the agent.
        """
        new_agent = Agent(self.agent_id, self.name, -1, -1, self.creation_variables)
        new_agent.tree_names = self.tree_names

        for i in range(8):
            new_agent.tree_list[i] = self.tree_list[i].__deepcopy__(memodict)

        return new_agent

    def assert_agent(self) -> None:
        """
        Asserts that the agent is correct.

        :return: None
        """
        for i in range(8):
            self.tree_list[i].assert_tree()

    def eval_throttle(self,
                      environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the throttle tree.
        "1.0" means accelerate, "0.0" means neutral and "-1.0" means decelerate.

        :param environment: Dict holding values for parameters.
        :return: Float between [-1.0, 1.0] or float in {-1, 0, 1}
        """
        return self.tree_list[0].eval(environment)

    def eval_steering(self,
                      environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the steering tree.

        :param environment: Dict holding values for parameters.
        :return: Float between [-1.0, 1.0] or float in {-1, 0, 1}
        """
        return self.tree_list[1].eval(environment)

    def eval_pitch(self,
                   environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the pitch tree.

        :param environment: Dict holding values for parameters.
        :return: Float between [-1.0, 1.0] or float in {-1, 0, 1}
        """
        return self.tree_list[2].eval(environment)

    def eval_yaw(self,
                 environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the yaw tree.

        :param environment: Dict holding values for parameters.
        :return: Float between [-1.0, 1.0] or float in {-1, 0, 1}
        """
        return self.tree_list[3].eval(environment)

    def eval_roll(self,
                  environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the yaw tree.

        :param environment: Dict holding values for parameters.
        :return: Float between [-1.0, 1.0] or float in {-1, 0, 1}
        """
        return self.tree_list[4].eval(environment)

    def eval_jump(self,
                  environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the jump tree.
        "1" means jump, "0" means don't jump.

        :param environment: Dict holding values for parameters.
        :return: Float in {0, 1}
        """
        return self.tree_list[5].eval(environment)

    def eval_boost(self,
                   environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the boost tree.
        "1" means boost, "0" means don't boost.

        :param environment: Dict holding values for parameters.
        :return: Float in {0, 1}
        """
        return self.tree_list[6].eval(environment)

    def eval_handbrake(self,
                       environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> Union[float, np.ndarray]:
        """
        Evaluates the handbrake tree.
        "1" means handbrake, "0" means no handbrake.

        :param environment: Dict holding values for parameters.
        :return: Float in {0, 1}
        """
        return self.tree_list[7].eval(environment)

    def eval_all(self,
                 environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> np.ndarray:
        """
        Evaluates all trees and returns the result as numpy array.

        :param environment: Dict holding values for parameters.
        :return: Array of size 8 * n
        """
        n = 1
        # check type of first item, numpy 1D arrays are also accepted
        node_type = list(environment.keys())[0]
        parameter = list(environment[node_type].keys())[0]
        if type(environment[node_type][parameter]) == np.ndarray:
            n = environment[node_type][parameter].shape[0]

        res = np.zeros(shape=(n, 8), dtype=float)
        for i in range(8):
            res[:, i] = self.tree_list[i].eval(environment)

        return res

    @staticmethod
    def random_eval_all(environment: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> np.ndarray:
        """
        Returns results result as numpy array.

        :param environment: Dict holding values for parameters.
        :return: Array of size 8
        """
        n = 1
        # check type of first item, numpy 1D arrays are also accepted
        node_type = list(environment.keys())[0]
        parameter = list(environment[node_type].keys())[0]
        if type(environment[node_type][parameter]) == np.ndarray:
            n = environment[node_type][parameter].shape[0]

        res = np.zeros(shape=(n, 8), dtype=float)
        res[:, 0] = scale_discrete(np.random.random(n), 0.0, 1.0, -1.0, 1.0)  # throttle [-1, 0, 1]
        res[:, 1] = scale_discrete(np.random.random(n), 0.0, 1.0, -1.0, 1.0)  # steer [-1, 0, 1]
        res[:, 2] = scale_discrete(np.random.random(n), 0.0, 1.0, -1.0, 1.0)  # pitch [-1, 0, 1]
        res[:, 3] = scale_discrete(np.random.random(n), 0.0, 1.0, -1.0, 1.0)  # yaw [-1, 0, 1]
        res[:, 4] = scale_discrete(np.random.random(n), 0.0, 1.0, -1.0, 1.0)  # roll [-1, 0, 1]
        res[:, 5] = scale_discrete(np.random.random(n), 0.0, 1.0, 0.0, 1.0)  # jump [0, 1]
        res[:, 6] = scale_discrete(np.random.random(n), 0.0, 1.0, 0.0, 1.0)  # boost [0, 1]
        res[:, 7] = scale_discrete(np.random.random(n), 0.0, 1.0, 0.0, 1.0)  # handbrake [0, 1]

        return res

    def bloat_analysis(self,
                       env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Analysis all trees for bloat.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        for i in range(8):
            self.tree_list[i].determine_bloat(env_stats)

    def unmark_bloat(self) -> None:
        """
        Marks all nodes in the trees as not bloat, resetting the tree.

        :return: None
        """
        for i in range(8):
            self.tree_list[i].unmark_bloat()

    def mutate(self,
               p: float):
        """
        Deep-copies the agent and mutates it. The copy is returned.

        :param p: Probability of a mutation in each node.
        :return: Mutated copy
        """
        # deepcopy the agent
        new_agent = self.__deepcopy__()

        for i in range(8):
            new_agent.tree_list[i].mutate(p)

        return new_agent

    def count_nodes(self) -> List[int]:
        """
        Counts the number of nodes in each tree.

        :return: List of integer
        """
        n_nodes = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(8):
            n_nodes[i] = self.tree_list[i].count_nodes()

        return n_nodes

    def count_non_bloat_nodes(self) -> List[int]:
        """
        Counts the number of non-bloat nodes in each tree.

        :return: List of integer
        """
        n_nodes = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(8):
            n_nodes[i] = self.tree_list[i].count_non_bloat_nodes()

        return n_nodes

    def info(self):
        """
        Gives info on the agent.

        :return: String of info for printing.
        """
        n_nodes = self.count_nodes()
        n_non_bloat_nodes = self.count_non_bloat_nodes()

        s = f'*** {self.name} ***\n'
        s += 'Tree:\t'
        for i in range(8):
            s += f'{self.tree_names[i]}\t'
        s += '\n#Nodes:\t'
        for i in range(8):
            s += f'{n_nodes[i]}\t'
        s += '\n#nbNodes:\t'
        for i in range(8):
            s += f'{n_non_bloat_nodes[i]}\n'
        return s

    def python_npy_jit(self) -> None:
        """
        Creates python compiled code to be executed instead of the eval
        function of the tree.

        :return: None
        """
        for i in range(8):
            self.tree_list[i].python_npy_jit()

    def prepare_for_rlbot(self):
        """
        Prepares the agent to be used by RLBot. Generates the python file, which
        will be loaded by RLBot as the actual agent. Pickle this object to store
        and load it later. Generates a cpp file, which will be compiled through
        cython and imported by this agent, to speedup execution times.

        :return: None
        """
        # generate directory in agent storage
        if not os.path.exists(f'agent_storage/'):
            os.mkdir(f'agent_storage/')

        if not os.path.exists(f'agent_storage/{self.name}/'):
            os.mkdir(f'agent_storage/{self.name}/')

        self._to_pickle()
        self._to_cpp()
        self._to_rlbot_agent()

    def _to_rlbot_agent(self):
        """
        Generated the python file which is needed by the RLBot framework. Also
        generates the appearance.cfg and agent.cfg file.

        :return: None
        """
        # generate the py file
        content = self._get_rlbot_py_file_content()
        file = open(f'agent_storage/{self.name}/{self.name}.py', 'w')
        file.write(content)
        file.close()

        # generate the agent cfg
        content = self._get_rlbot_agent_cfg_content()
        file = open(f'agent_storage/{self.name}/{self.name}.cfg', 'w')
        file.write(content)
        file.close()

        # generate the appearance cfg
        content = self._get_rlbot_appearance_cfg_content()
        file = open(f'agent_storage/{self.name}/appearance_{self.name}.cfg', 'w')
        file.write(content)
        file.close()

    def _to_pickle(self):
        """
        Pickles this object so it can be loaded later.

        :return: None
        """
        with open(f'agent_storage/{self.name}/{self.name}.pickle', 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _to_cpp(self):
        """
        Generates a cpp file, which implements the trees. The file will be
        compiled through cython and loaded by the python file for the rlbot,
        to speedup execution time.

        :return: None
        """
        pass

    def _get_rlbot_py_file_content(self):
        """
        Generates the content for a rlbot python file, so this agent can be
        executed through RLBot.

        :return: Content of the file as a string.
        """
        content = ""
        content += f"import pickle\n"
        content += f"import os\n"
        content += f"import time\n"
        content += f"\n"
        content += f"from rlbot.agents.base_agent import BaseAgent, SimpleControllerState\n"
        content += f"from rlbot.utils.structures.game_data_struct import GameTickPacket\n"
        content += f"from rlbot.utils.game_state_util import GameState, GameInfoState"
        content += f"\n"
        content += f"from src.util.boost_pad_tracker import BoostPadTracker\n"
        content += f"from src.util.sequence import Sequence\n"
        content += f"from src.util.vec import Vec3\n"
        content += f"\n"
        content += f"\n"
        content += f"class MyBot(BaseAgent):\n"
        content += f"    def __init__(self, name, team, index):\n"
        content += f"        super().__init__(name, team, index)\n"
        content += f"        self.active_sequence: Sequence = None\n"
        content += f"        self.boost_pad_tracker = BoostPadTracker()\n"
        content += f"\n"
        content += f"        # index for extracting the game package\n"
        content += f"        self.enemy_index = -1\n"
        content += f"\n"
        content += f"        print(os.getcwd())\n"
        content += f"        file = open(f'{self.name}/{self.name}.pickle', 'rb')\n"
        content += f"        self.agent = pickle.load(file)\n"
        content += f"        file.close()\n"
        content += f"\n"
        content += f"        self.min_steering = float('inf')\n"
        content += f"        self.max_steering = -float('inf')\n"
        content += f"        self.min_throttle = float('inf')\n"
        content += f"        self.max_throttle = -float('inf')\n"
        content += f"\n"
        content += f"    def initialize_agent(self):\n"
        content += f"        # Set up information about the boost pads now that the game is active\n"
        content += f"        # and the info is available\n"
        content += f"        self.boost_pad_tracker.initialize_boosts(self.get_field_info())\n"
        content += f"        game_info_state = GameInfoState(game_speed=4.0)\n"
        content += f"        game_state = GameState(game_info=game_info_state)\n"
        content += f"        self.set_game_state(game_state)\n"
        content += f"\n"
        content += f"    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:\n"
        content += f"        \"\"\"\n"
        content += f"        This function will be called by the framework many times per second.\n"
        content += f"        This is where you can see the motion of the ball, etc. and return\n"
        content += f"        controls to drive your car.\n"
        content += f"        \"\"\"\n"
        content += f"        # Keep our boost pad info updated with which pads are currently active\n"
        content += f"        self.boost_pad_tracker.update_boost_status(packet)\n"
        content += f"\n"
        content += f"        # determine which car id we have and which one the enemy has\n"
        content += f"        if self.enemy_index == -1:\n"
        content += f"            for i in range(64):\n"
        content += f"                car = packet.game_cars[i]\n"
        content += f"                car_x = Vec3(car.physics.location).inv_w\n"
        content += f"                if car_x != 0.0 and i != self.index:\n"
        content += f"                    self.enemy_index = i\n"
        content += f"\n"
        content += f"        # read the features of our car\n"
        content += f"        my_car = packet.game_cars[self.index]\n"
        content += f"        my_car_location = Vec3(my_car.physics.location)\n"
        content += f"        my_car_velocity = Vec3(my_car.physics.velocity)\n"
        content += f"        my_car_rotation = my_car.physics.rotation\n"
        content += f"\n"
        content += f"        # read the features of the enemy car\n"
        content += f"        enemy_car = packet.game_cars[self.enemy_index]\n"
        content += f"        enemy_car_location = Vec3(enemy_car.physics.location)\n"
        content += f"        enemy_car_velocity = Vec3(enemy_car.physics.velocity)\n"
        content += f"        enemy_car_rotation = enemy_car.physics.rotation\n"
        content += f"\n"
        content += f"        # read the features of the ball\n"
        content += f"        ball_location = Vec3(packet.game_ball.physics.location)\n"
        content += f"        ball_velocity = Vec3(packet.game_ball.physics.velocity)\n"
        content += f"        ball_rotation = packet.game_ball.physics.rotation\n"
        content += f"\n"
        content += f"        # read the features of the game\n"
        content += f"        my_team = packet.teams[self.index]\n"
        content += f"        enemy_team = packet.teams[self.enemy_index]\n"
        content += f"\n"
        content += "        environment = {'ARITHMETIC': {'my_car_x': my_car_location.inv_w,\n"
        content += f"                              'my_car_y': my_car_location.y,\n"
        content += f"                              'my_car_z': my_car_location.z,\n"
        content += f"                              'my_car_velocity_x': my_car_velocity.inv_w,\n"
        content += f"                              'my_car_velocity_y': my_car_velocity.y,\n"
        content += f"                              'my_car_velocity_z': my_car_velocity.z,\n"
        content += f"                              'my_car_rotation_yaw': my_car_rotation.yaw,\n"
        content += f"                              'my_car_rotation_pitch': my_car_rotation.pitch,\n"
        content += f"                              'my_car_rotation_roll': my_car_rotation.roll,\n"
        content += f"                              'enemy_car_x': enemy_car_location.inv_w,\n"
        content += f"                              'enemy_car_y': enemy_car_location.y,\n"
        content += f"                              'enemy_car_z': enemy_car_location.z,\n"
        content += f"                              'enemy_car_velocity_x': enemy_car_velocity.inv_w,\n"
        content += f"                              'enemy_car_velocity_y': enemy_car_velocity.y,\n"
        content += f"                              'enemy_car_velocity_z': enemy_car_velocity.z,\n"
        content += f"                              'enemy_car_rotation_yaw': enemy_car_rotation.yaw,\n"
        content += f"                              'enemy_car_rotation_pitch': enemy_car_rotation.pitch,\n"
        content += f"                              'enemy_car_rotation_roll': enemy_car_rotation.roll,\n"
        content += f"                              'ball_x': ball_location.inv_w,\n"
        content += f"                              'ball_y': ball_location.y,\n"
        content += f"                              'ball_z': ball_location.z,\n"
        content += f"                              'ball_velocity_x': ball_velocity.inv_w,\n"
        content += f"                              'ball_velocity_y': ball_velocity.y,\n"
        content += f"                              'ball_velocity_z': ball_velocity.z,\n"
        content += f"                              'ball_rotation_yaw': ball_rotation.yaw,\n"
        content += f"                              'ball_rotation_pitch': ball_rotation.pitch,\n"
        content += f"                              'ball_rotation_roll': ball_rotation.roll,\n"
        content += f"                              'my_team_score': my_team.score,\n"
        content += f"                              'enemy_team_score': enemy_team.score,\n"
        content += "                              'remaining_time': packet.game_info.game_time_remaining},\n"
        content += f"\n"
        content += "               'LOGIC': {'kickoff': int(packet.game_info.is_kickoff_pause) == 1,\n"
        content += "                         'overtime': int(packet.game_info.is_overtime) == 1}\n"
        content += "               }\n"
        content += f"\n"
        content += f"        t = time.time()\n"
        content += f"        steer = self.agent.eval_steering(environment)\n"
        content += f"        t_steer = time.time() - t\n"
        content += f"\n"
        content += f"        t = time.time()\n"
        content += f"        throttle = self.agent.eval_throttle(environment)\n"
        content += f"        t_throttle = time.time() - t\n"
        content += f"\n"
        content += f"        self.max_steering = max(self.max_steering, steer)\n"
        content += f"        self.min_steering = min(self.min_steering, steer)\n"
        content += f"        self.max_throttle = max(self.max_throttle, throttle)\n"
        content += f"        self.min_throttle = min(self.min_throttle, throttle)\n"
        content += f"\n"
        content += f"        norm_steer = 0.0\n"
        content += f"        if self.max_steering - self.min_steering != 0.0:\n"
        content += f"            norm_steer = (steer - self.min_steering) / (self.max_steering - self.min_steering)\n"
        content += f"            norm_steer = norm_steer * (1 - -1) + -1\n"
        content += f"\n"
        content += f"        norm_throttle = 0.0\n"
        content += f"        if self.max_throttle - self.min_throttle != 0.0:\n"
        content += f"            norm_throttle = (throttle - self.min_throttle) / (self.max_throttle - self.min_throttle)\n"
        content += f"            norm_throttle = norm_throttle * (1 - -1) + -1\n"
        content += f"\n"
        content += f"        print(self.agent.name, norm_steer, norm_throttle, t_steer, t_throttle)\n"
        content += f"\n"
        content += f"        controls = SimpleControllerState()\n"
        content += f"        controls.steer = norm_steer\n"
        content += f"        controls.throttle = norm_throttle\n"
        content += f"\n"
        content += f"        return controls\n"

        return content

    def _get_rlbot_agent_cfg_content(self):
        """
        Generates the content for the agent.cfg, so this agent can be
        executed through RLBot.

        :return: Content of the file as a string.
        """
        content = f"[Locations]\n"
        content += f"# Path to loadout config. Can use relative path from here.\n"
        content += f"looks_config = ./appearance_{self.name}.cfg\n"
        content += f"\n"
        content += f"# Path to python file. Can use relative path from here.\n"
        content += f"python_file = ./{self.name}.py\n"
        content += f"\n"
        content += f"# Name of the agent in-game\n"
        content += f"name = {self.name}\n"
        content += f"\n"
        content += f"# The maximum number of ticks per second that your agent wishes to receive.\n"
        content += f"maximum_tick_rate_preference = 10\n"
        content += f"\n"
        content += f"[Details]\n"
        content += f"# These values are optional but useful metadata for helper programs\n"
        content += f"# Name of the agent's creator/developer\n"
        content += f"developer = RLLeague\n"
        content += f"\n"
        content += f"# Short description of the agent\n"
        content += f"description = This is a multi-line description\n"
        content += f"    of the official python example agent\n"
        content += f"\n"
        content += f"# Fun fact about the agent\n"
        content += f"fun_fact = \'Life Is Suffering\' - {self.name}, 2022 \n"
        content += f"\n"
        content += f"# Link to github repository\n"
        content += f"github = \n"
        content += f"\n"
        content += f"# Programming language\n"
        content += f"language = python\n"

        return content

    def _get_rlbot_appearance_cfg_content(self):
        """
        Generates the content for the appearance.cfg, so this agent can be
        visualized through RLBot.

        :return: Content of the file as a string.
        """
        content = f"# You don't have to manually edit this file!\n"
        content += f"# RLBotGUI has an appearance editor with a nice colorpicker, database of items and more!\n"
        content += f"# To open it up, simply click the (i) icon next to your agent's name and then click Edit Appearance"
        content += f"\n"
        content += f"[agent Loadout]\n"
        content += f"team_color_id = 60\n"
        content += f"custom_color_id = 0\n"
        content += f"car_id = 23\n"
        content += f"decal_id = 0\n"
        content += f"wheels_id = 1565\n"
        content += f"boost_id = 35\n"
        content += f"antenna_id = 0\n"
        content += f"hat_id = 0\n"
        content += f"paint_finish_id = 1681\n"
        content += f"custom_finish_id = 1681\n"
        content += f"engine_audio_id = 0\n"
        content += f"trails_id = 3220\n"
        content += f"goal_explosion_id = 3018\n"
        content += f"\n"
        content += f"[agent Loadout Orange]\n"
        content += f"team_color_id = 3\n"
        content += f"custom_color_id = 0\n"
        content += f"car_id = 23\n"
        content += f"decal_id = 0\n"
        content += f"wheels_id = 1565\n"
        content += f"boost_id = 35\n"
        content += f"antenna_id = 0\n"
        content += f"hat_id = 0\n"
        content += f"paint_finish_id = 1681\n"
        content += f"custom_finish_id = 1681\n"
        content += f"engine_audio_id = 0\n"
        content += f"trails_id = 3220\n"
        content += f"goal_explosion_id = 3018\n"
        content += f"\n"
        content += f"[agent Paint Blue]\n"
        content += f"car_paint_id = 12\n"
        content += f"decal_paint_id = 0\n"
        content += f"wheels_paint_id = 7\n"
        content += f"boost_paint_id = 7\n"
        content += f"antenna_paint_id = 0\n"
        content += f"hat_paint_id = 0\n"
        content += f"trails_paint_id = 2\n"
        content += f"goal_explosion_paint_id = 0\n"
        content += f"\n"
        content += f"[agent Paint Orange]\n"
        content += f"car_paint_id = 12\n"
        content += f"decal_paint_id = 0\n"
        content += f"wheels_paint_id = 14\n"
        content += f"boost_paint_id = 14\n"
        content += f"antenna_paint_id = 0\n"
        content += f"hat_paint_id = 0\n"
        content += f"trails_paint_id = 14\n"
        content += f"goal_explosion_paint_id = 0\n"

        return content


def recombine_agents(agent_1: Agent,
                     agent_2: Agent,
                     p: float) -> Tuple['Agent', 'Agent']:
    """
    Recombines both agents. Will copy the agents and swap two subtrees.

    :param agent_1: Agent 1.
    :param agent_2: Agent 2.
    :param p: Probability that a tree will change.
    :return: The two agents recombined
    """
    agent_1 = agent_1.__deepcopy__()
    agent_2 = agent_2.__deepcopy__()

    for tree_1, tree_2 in zip(agent_1.tree_list, agent_2.tree_list):
        if np.random.sample() < p:
            recombine_trees(tree_1, tree_2)

    return agent_1, agent_2


def recombine_agents_by_tree(agent_1: Agent,
                             agent_2: Agent,
                             idx: int) -> Tuple['Agent', 'Agent']:
    """
    Recombines both agents. Will copy the agents and swap two subtrees.

    :param agent_1: Agent 1.
    :param agent_2: Agent 2.
    :param idx: Idx of the tree that should be recombined.
    :return: The two agents recombined
    """
    agent_1 = agent_1.__deepcopy__()
    agent_2 = agent_2.__deepcopy__()

    recombine_trees(agent_1.tree_list[idx], agent_2.tree_list[idx])

    return agent_1, agent_2
