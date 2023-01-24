"""
Implements the bot.
"""
import os
import pickle
import _pickle as cPickle
import random
from typing import Dict, Union, Tuple

import numpy as np

from bot.nodes import count_nodes, leaf_type_count, count_non_bloat_nodes, all_branch_nodes, recombine_nodes
from bot.nodes import Node


class Bot(object):
    def __init__(self,
                 bot_id: int,
                 name: str) -> None:
        self.bot_id: int = bot_id
        self.name: str = name

        self.steering_root: Union[Node, None] = None
        self.throttle_root: Union[Node, None] = None
        self.jump_root: Union[Node, None] = None
        self.boost_root: Union[Node, None] = None
        self.handbrake_root: Union[Node, None] = None

        self.creation_variables = {}

    def __str__(self):
        return f"{self.name}"

    def connect_trees(self):
        """
        Connects all trees.
        """
        self.throttle_root.connect_tree()
        self.steering_root.connect_tree()
        self.jump_root.connect_tree()
        self.boost_root.connect_tree()
        self.handbrake_root.connect_tree()

    def disconnect_trees(self):
        """
        Connects all trees.
        """
        self.throttle_root.disconnect_tree()
        self.steering_root.disconnect_tree()
        self.jump_root.disconnect_tree()
        self.boost_root.disconnect_tree()
        self.handbrake_root.disconnect_tree()

    def eval_steering(self,
                      env: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Evaluates the steering tree.

        :param env: Dict holding values for parameters.
        :return: Float between [-1.0, 1.0] or float in {-1, 0, 1}
        """
        res = self.steering_root.eval(env)
        # res = self.sigmoid(res)
        res = self.scale(res, self.steering_root.bloat_min, self.steering_root.bloat_max, -1, 1)
        res = self.round_1_0_1(res)
        return res

    def eval_throttle(self,
                      env: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Evaluates the throttle tree. "1.0" means accelerate, "0.0" means neutral
        and "-1.0" means decelerate.

        :param env: Dict holding values for parameters.
        :return: Float between [-1.0, 1.0] or float in {-1, 0, 1}
        """
        res = self.throttle_root.eval(env)
        # res = self.sigmoid(res)
        res = self.scale(res, self.throttle_root.bloat_min, self.throttle_root.bloat_max, -1, 1)
        res = self.round_1_0_1(res)
        return res

    def eval_jump(self,
                  env: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Evaluates the jump tree. "1" means jump, "0" means don't jump.

        :param env: Dict holding values for parameters.
        :return: Float in {0, 1}
        """
        res = self.jump_root.eval(env)
        # res = self.sigmoid(res)
        res = self.scale(res, self.jump_root.bloat_min, self.jump_root.bloat_max, 0, 1)
        res = self.round_0_1(res)
        return res

    def eval_boost(self,
                   env: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Evaluates the boost tree. "1" means boost, "0" means don't boost.

        :param env: Dict holding values for parameters.
        :return: Float in {0, 1}
        """
        res = self.boost_root.eval(env)
        # res = self.sigmoid(res)
        res = self.scale(res, self.boost_root.bloat_min, self.boost_root.bloat_max, 0, 1)
        res = self.round_0_1(res)
        return res

    def eval_handbrake(self,
                       env: Dict[str, Dict[str, Union[float, bool]]]) -> float:
        """
        Evaluates the handbrake tree. "1" means handbrake, "0" means no handbrake.

        :param env: Dict holding values for parameters.
        :return: Float in {0, 1}
        """
        res = self.handbrake_root.eval(env)
        # res = self.sigmoid(res)
        res = self.scale(res, self.handbrake_root.bloat_min, self.handbrake_root.bloat_max, 0, 1)
        res = self.round_0_1(res)
        return res

    def eval_all(self,
                 env: Dict[str, Dict[str, Union[float, bool, np.ndarray]]]) -> np.ndarray:
        """
        Evaluates all trees and returns the result as numpy array.

        :param env: Dict holding values for parameters.
        :return: Array of size 8
        """
        n = 1
        # check type of first item, numpy 1D arrays are also accepted
        node_type = list(env.keys())[0]
        parameter = list(env[node_type].keys())[0]
        if type(env[node_type][parameter]) == np.ndarray:
            n = env[node_type][parameter].shape[0]

        res = np.zeros(shape=(n, 8), dtype=float)
        res[:, 0] = self.eval_throttle(env)  # throttle [-1, 0, 1]
        res[:, 1] = self.eval_steering(env)  # steer [-1, 0, 1]
        res[:, 2] = 0.0  # pitch [-1, 0, 1]
        res[:, 3] = 0.0  # yaw [-1, 0, 1]
        res[:, 4] = 0.0  # roll [-1, 0, 1]
        res[:, 5] = self.eval_jump(env)  # jump [0, 1]
        res[:, 6] = self.eval_boost(env)   # boost [0, 1]
        res[:, 7] = self.eval_handbrake(env)  # handbrake [0, 1]

        return res

    @staticmethod
    def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Applies the sigmoid function on x.

        :param x: A scalar or np.ndarray
        """
        return 1 / (1 + np.exp(-x * (1/1000000)))

    @staticmethod
    def scale(x: Union[float, np.ndarray],
              x_min: float,
              x_max: float,
              a: float,
              b: float) -> Union[float, np.ndarray]:
        """
        Scales x to lie in the interval [a, b].

        :param x: The number.
        :param x_min: Minimum value x can take.
        :param x_max: Maximum value x can take.
        :param a: Lower bound of the interval.
        :param b: Upper bound of the interval.
        :return: x scaled to the interval.
        """
        return ((x - x_min) / (x_max - x_min)) * (b - a) + a

    @staticmethod
    def round_1_0_1(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Rounds x (x in [-1, 1]) to -1, 0 or 1.

        :param x: A number.
        :return: -1, 0, or 1
        """
        if isinstance(x, np.ndarray):
            first_part = (-1.0 <= x) & (x < -1/3)
            second_part = (-1/3 <= x) & (x < 1/3)
            third_part = ~first_part & ~second_part

            return (first_part * -1.0) + (second_part * 0.0) + (third_part * 1.0)
        else:
            if -1.0 <= x < -1/3:
                return -1.0
            elif -1/3 <= x < 1/3:
                return 0.0
            else:
                return 1.0

    @staticmethod
    def round_0_1(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Round x (x in [0, 1]) to 0 or 1.

        :param x: A number.
        :return: 0 or 1
        """
        if isinstance(x, np.ndarray):
            first_part = (0.0 <= x) & (x < 1/2)
            second_part = ~first_part

            return (first_part * 0.0) + (second_part * 1.0)
        else:
            return round(x)

    def bloat_analysis(self,
                       env_stats: Dict[str, Dict[str, Dict[str, Union[float, bool]]]]) -> None:
        """
        Analysis all trees for bloat.

        :param env_stats: Statistics for the parameters of the environment.
        :return: None
        """
        self.throttle_root.determine_bloat(env_stats)
        self.steering_root.determine_bloat(env_stats)
        self.jump_root.determine_bloat(env_stats)
        self.boost_root.determine_bloat(env_stats)
        self.handbrake_root.determine_bloat(env_stats)

    def unmark_bloat(self) -> None:
        """
        Marks all nodes in the trees as not bloat, resetting the tree.

        :return: None
        """
        self.throttle_root.unmark_bloat()
        self.steering_root.unmark_bloat()
        self.jump_root.unmark_bloat()
        self.boost_root.unmark_bloat()
        self.handbrake_root.unmark_bloat()

    def mutate(self,
               mutate_probability: float):
        """
        Deep-copies the bot and mutates it. The copy is returned.

        :param mutate_probability: Probability of a mutation in each node.
        :return: Mutated copy
        """
        new_bot = cPickle.loads(cPickle.dumps(self, -1))  # copy.deepcopy(self)

        work_list = []
        work_list.extend(new_bot.throttle_root.children)
        work_list.extend(new_bot.steering_root.children)
        work_list.extend(new_bot.jump_root.children)
        work_list.extend(new_bot.boost_root.children)
        work_list.extend(new_bot.handbrake_root.children)

        while work_list:
            node = work_list.pop()

            if np.random.uniform(0.0, 1.0) < mutate_probability:
                # mutate the node
                node.mutate(self.creation_variables)

            work_list.extend(node.children)

        return new_bot

    def info(self):
        """
        Gives info on the bot.

        :return: String of info for printing.
        """
        n_steering = count_nodes(self.steering_root)
        n_throttle = count_nodes(self.throttle_root)
        n_jump = count_nodes(self.jump_root)
        n_boost = count_nodes(self.boost_root)
        n_handbrake = count_nodes(self.handbrake_root)

        n_non_bloat_steering_nodes = count_non_bloat_nodes(self.steering_root)
        n_non_bloat_throttle_nodes = count_non_bloat_nodes(self.throttle_root)
        n_non_bloat_jump_nodes = count_non_bloat_nodes(self.jump_root)
        n_non_bloat_boost_nodes = count_non_bloat_nodes(self.boost_root)
        n_non_bloat_handbrake_nodes = count_non_bloat_nodes(self.handbrake_root)

        steering_leaf_nodes = leaf_type_count(self.steering_root)
        throttle_leaf_nodes = leaf_type_count(self.throttle_root)
        jump_leaf_nodes = leaf_type_count(self.jump_root)
        boost_leaf_nodes = leaf_type_count(self.boost_root)
        handbrake_leaf_nodes = leaf_type_count(self.handbrake_root)

        s = f'*** {self.name} ***\n'
        s += f'-- Steering --\n'
        s += f'#nodes: {n_steering}\n'
        s += f'#non bloat nodes: {n_non_bloat_steering_nodes}\n'
        s += f'leaf nodes: {steering_leaf_nodes}\n'
        s += f'\n'
        s += f'-- Throttle --\n'
        s += f'#nodes: {n_throttle}\n'
        s += f'#non bloat nodes: {n_non_bloat_throttle_nodes}\n'
        s += f'leaf nodes: {throttle_leaf_nodes}\n'
        s += f'\n'
        s += f'-- Jump --\n'
        s += f'#nodes: {n_jump}\n'
        s += f'#non bloat nodes: {n_non_bloat_jump_nodes}\n'
        s += f'leaf nodes: {jump_leaf_nodes}\n'
        s += f'\n'
        s += f'-- Boost --\n'
        s += f'#nodes: {n_boost}\n'
        s += f'#non bloat nodes: {n_non_bloat_boost_nodes}\n'
        s += f'leaf nodes: {boost_leaf_nodes}\n'
        s += f'\n'
        s += f'-- Handbrake --\n'
        s += f'#nodes: {n_handbrake}\n'
        s += f'#non bloat nodes: {n_non_bloat_handbrake_nodes}\n'
        s += f'leaf nodes: {handbrake_leaf_nodes}\n'
        return s

    def prepare_for_rlbot(self):
        """
        Prepares the bot to be used by RLBot. Generates the python file, which
        will be loaded by RLBot as the actual bot. Pickle this object to store
        and load it later. Generates a cpp file, which will be compiled through
        cython and imported by this bot, to speedup execution times.

        :return: None
        """
        # generate directory in bot storage
        if not os.path.exists(f'bot_storage/'):
            os.mkdir(f'bot_storage/')

        if not os.path.exists(f'bot_storage/{self.name}/'):
            os.mkdir(f'bot_storage/{self.name}/')

        self._to_pickle()
        self._to_cpp()
        self._to_rlbot_bot()

    def _to_rlbot_bot(self):
        """
        Generated the python file which is needed by the RLBot framework. Also
        generates the appearance.cfg and bot.cfg file.

        :return: None
        """
        # generate the py file
        content = self._get_rlbot_py_file_content()
        file = open(f'bot_storage/{self.name}/{self.name}.py', 'w')
        file.write(content)
        file.close()

        # generate the bot cfg
        content = self._get_rlbot_bot_cfg_content()
        file = open(f'bot_storage/{self.name}/{self.name}.cfg', 'w')
        file.write(content)
        file.close()

        # generate the appearance cfg
        content = self._get_rlbot_appearance_cfg_content()
        file = open(f'bot_storage/{self.name}/appearance_{self.name}.cfg', 'w')
        file.write(content)
        file.close()

    def _to_pickle(self):
        """
        Pickles this object so it can be loaded later.

        :return: None
        """
        with open(f'bot_storage/{self.name}/{self.name}.pickle', 'wb') as file:
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
        Generates the content for a rlbot python file, so this bot can be
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
        content += f"        self.bot = pickle.load(file)\n"
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
        content += f"                car_x = Vec3(car.physics.location).x\n"
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
        content += "        env = {'ARITHMETIC': {'my_car_x': my_car_location.x,\n"
        content += f"                              'my_car_y': my_car_location.y,\n"
        content += f"                              'my_car_z': my_car_location.z,\n"
        content += f"                              'my_car_velocity_x': my_car_velocity.x,\n"
        content += f"                              'my_car_velocity_y': my_car_velocity.y,\n"
        content += f"                              'my_car_velocity_z': my_car_velocity.z,\n"
        content += f"                              'my_car_rotation_yaw': my_car_rotation.yaw,\n"
        content += f"                              'my_car_rotation_pitch': my_car_rotation.pitch,\n"
        content += f"                              'my_car_rotation_roll': my_car_rotation.roll,\n"
        content += f"                              'enemy_car_x': enemy_car_location.x,\n"
        content += f"                              'enemy_car_y': enemy_car_location.y,\n"
        content += f"                              'enemy_car_z': enemy_car_location.z,\n"
        content += f"                              'enemy_car_velocity_x': enemy_car_velocity.x,\n"
        content += f"                              'enemy_car_velocity_y': enemy_car_velocity.y,\n"
        content += f"                              'enemy_car_velocity_z': enemy_car_velocity.z,\n"
        content += f"                              'enemy_car_rotation_yaw': enemy_car_rotation.yaw,\n"
        content += f"                              'enemy_car_rotation_pitch': enemy_car_rotation.pitch,\n"
        content += f"                              'enemy_car_rotation_roll': enemy_car_rotation.roll,\n"
        content += f"                              'ball_x': ball_location.x,\n"
        content += f"                              'ball_y': ball_location.y,\n"
        content += f"                              'ball_z': ball_location.z,\n"
        content += f"                              'ball_velocity_x': ball_velocity.x,\n"
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
        content += f"        steer = self.bot.eval_steering(env)\n"
        content += f"        t_steer = time.time() - t\n"
        content += f"\n"
        content += f"        t = time.time()\n"
        content += f"        throttle = self.bot.eval_throttle(env)\n"
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
        content += f"        print(self.bot.name, norm_steer, norm_throttle, t_steer, t_throttle)\n"
        content += f"\n"
        content += f"        controls = SimpleControllerState()\n"
        content += f"        controls.steer = norm_steer\n"
        content += f"        controls.throttle = norm_throttle\n"
        content += f"\n"
        content += f"        return controls\n"

        return content

    def _get_rlbot_bot_cfg_content(self):
        """
        Generates the content for the bot.cfg, so this bot can be
        executed through RLBot.

        :return: Content of the file as a string.
        """
        content =  f"[Locations]\n"
        content += f"# Path to loadout config. Can use relative path from here.\n"
        content += f"looks_config = ./appearance_{self.name}.cfg\n"
        content += f"\n"
        content += f"# Path to python file. Can use relative path from here.\n"
        content += f"python_file = ./{self.name}.py\n"
        content += f"\n"
        content += f"# Name of the bot in-game\n"
        content += f"name = {self.name}\n"
        content += f"\n"
        content += f"# The maximum number of ticks per second that your bot wishes to receive.\n"
        content += f"maximum_tick_rate_preference = 10\n"
        content += f"\n"
        content += f"[Details]\n"
        content += f"# These values are optional but useful metadata for helper programs\n"
        content += f"# Name of the bot's creator/developer\n"
        content += f"developer = RLLeague\n"
        content += f"\n"
        content += f"# Short description of the bot\n"
        content += f"description = This is a multi-line description\n"
        content += f"    of the official python example bot\n"
        content += f"\n"
        content += f"# Fun fact about the bot\n"
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
        Generates the content for the appearance.cfg, so this bot can be
        visualized through RLBot.

        :return: Content of the file as a string.
        """
        content = f"# You don't have to manually edit this file!\n"
        content += f"# RLBotGUI has an appearance editor with a nice colorpicker, database of items and more!\n"
        content += f"# To open it up, simply click the (i) icon next to your bot's name and then click Edit Appearance"
        content += f"\n"
        content += f"[bot Loadout]\n"
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
        content += f"[bot Loadout Orange]\n"
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
        content += f"[bot Paint Blue]\n"
        content += f"car_paint_id = 12\n"
        content += f"decal_paint_id = 0\n"
        content += f"wheels_paint_id = 7\n"
        content += f"boost_paint_id = 7\n"
        content += f"antenna_paint_id = 0\n"
        content += f"hat_paint_id = 0\n"
        content += f"trails_paint_id = 2\n"
        content += f"goal_explosion_paint_id = 0\n"
        content += f"\n"
        content += f"[bot Paint Orange]\n"
        content += f"car_paint_id = 12\n"
        content += f"decal_paint_id = 0\n"
        content += f"wheels_paint_id = 14\n"
        content += f"boost_paint_id = 14\n"
        content += f"antenna_paint_id = 0\n"
        content += f"hat_paint_id = 0\n"
        content += f"trails_paint_id = 14\n"
        content += f"goal_explosion_paint_id = 0\n"

        return content


def recombine_bots(bot_1: Bot, bot_2: Bot, tree_probability: float) -> Tuple['Bot', 'Bot']:
    """
    Recombines both bots. Will copy the bots and swap two subtrees.

    :param bot_1: Bot 1.
    :param bot_2: Bot 2.
    :param tree_probability: Probability that a tree will change.
    :return: None
    """
    bot_1 = cPickle.loads(cPickle.dumps(bot_1, -1))  # copy.deepcopy(self)
    bot_2 = cPickle.loads(cPickle.dumps(bot_2, -1))  # copy.deepcopy(self)

    bot_1_roots = [bot_1.throttle_root, bot_1.steering_root, bot_1.jump_root, bot_1.boost_root, bot_1.handbrake_root]
    bot_2_roots = [bot_2.throttle_root, bot_2.steering_root, bot_2.jump_root, bot_2.boost_root, bot_2.handbrake_root]

    for root_1, root_2 in zip(bot_1_roots, bot_2_roots):
        if np.random.sample() < tree_probability:
            # choose uniformly which node type we will swap
            node_type = random.choice(all_branch_nodes)

            tree_1_nodes = root_1.get_all_nodes(node_type)
            tree_2_nodes = root_2.get_all_nodes(node_type)

            if tree_1_nodes and tree_2_nodes:
                # choose two nodes uniformly
                node_1 = random.choice(tree_1_nodes)
                node_2 = random.choice(tree_2_nodes)

                assert type(node_1) == type(node_2) == node_type

                recombine_nodes(node_1, node_2)

    return bot_1, bot_2
