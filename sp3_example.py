# Here we import the Match object and our multi-instance wrapper
import datetime
import pickle
import random
from typing import Union, List

import gym.spaces
import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import ContinuousAction
from rlgym.utils.gamestates import GameState
from rlgym.utils.obs_builders import DefaultObs
# Since we can't use the normal rlgym.make() function, we need to import all the default configuration objects to give to our Match.
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
import win32gui, win32con

# Finally, we import the SB3 implementation of PPO.
from stable_baselines3.ppo import PPO


def reward_movement(idx: int, state: GameState):
    return np.linalg.norm(state.players[idx].inverted_car_data.linear_velocity)


class RLLeagueAction(ContinuousAction):
    """
        Continuous Action space, that also accepts a few other input formats for QoL reasons and to remain
        compatible with older versions.
    """

    def __init__(self, id1: int, id2: int):
        """
        Initializes an ActionParser which controls agents with id1 and id2.

        :param id1: Id of the first agent
        :param id2: Id of the second agent
        """
        super().__init__()

        self.agent_1 = pickle.load(open(f'bot_storage/bot_{id1}/bot_{id1}.pickle', 'rb'))
        self.agent_2 = pickle.load(open(f'bot_storage/bot_{id2}/bot_{id2}.pickle', 'rb'))
        self.rew_1 = 0.0
        self.rew_2 = 0.0

    def get_action_space(self) -> gym.spaces.Space:
        return super().get_action_space()

    def parse_actions(self, actions: Union[np.ndarray, List[np.ndarray], List[float]], state: GameState) -> np.ndarray:
        self.rew_1 += reward_movement(0, state)
        self.rew_2 += reward_movement(1, state)

        # print(f"Agent 1: {self.rew_1}")
        # print(f"Agent 2: {self.rew_2}")

        env_1 = {'ARITHMETIC': {'my_car_x': state.players[0].inverted_car_data.position[0],
                                'my_car_y': state.players[0].inverted_car_data.position[1],
                                'my_car_z': state.players[0].inverted_car_data.position[2],
                                'my_car_velocity_x': state.players[0].inverted_car_data.linear_velocity[0],
                                'my_car_velocity_y': state.players[0].inverted_car_data.linear_velocity[1],
                                'my_car_velocity_z': state.players[0].inverted_car_data.linear_velocity[2],
                                'my_car_rotation_yaw': state.players[0].inverted_car_data.angular_velocity[0],
                                'my_car_rotation_pitch': state.players[0].inverted_car_data.angular_velocity[1],
                                'my_car_rotation_roll': state.players[0].inverted_car_data.angular_velocity[2],
                                'enemy_car_x': state.players[1].inverted_car_data.position[0],
                                'enemy_car_y': state.players[1].inverted_car_data.position[1],
                                'enemy_car_z': state.players[1].inverted_car_data.position[2],
                                'enemy_car_velocity_x': state.players[1].inverted_car_data.linear_velocity[0],
                                'enemy_car_velocity_y': state.players[1].inverted_car_data.linear_velocity[1],
                                'enemy_car_velocity_z': state.players[1].inverted_car_data.linear_velocity[2],
                                'enemy_car_rotation_yaw': state.players[1].inverted_car_data.angular_velocity[0],
                                'enemy_car_rotation_pitch': state.players[1].inverted_car_data.angular_velocity[1],
                                'enemy_car_rotation_roll': state.players[1].inverted_car_data.angular_velocity[2],
                                'ball_x': state.inverted_ball.position[0],
                                'ball_y': state.inverted_ball.position[1],
                                'ball_z': state.inverted_ball.position[2],
                                'ball_velocity_x': state.inverted_ball.linear_velocity[0],
                                'ball_velocity_y': state.inverted_ball.linear_velocity[1],
                                'ball_velocity_z': state.inverted_ball.linear_velocity[2],
                                'ball_rotation_yaw': state.inverted_ball.angular_velocity[0],
                                'ball_rotation_pitch': state.inverted_ball.angular_velocity[1],
                                'ball_rotation_roll': state.inverted_ball.angular_velocity[2],
                                'my_team_score': state.blue_score,
                                'enemy_team_score': state.orange_score,
                                'remaining_time': 100},

                 'LOGIC': {'kickoff': False,
                           'overtime': False}
                 }

        env_2 = {'ARITHMETIC': {'my_car_x': state.players[1].inverted_car_data.position[0],
                                'my_car_y': state.players[1].inverted_car_data.position[1],
                                'my_car_z': state.players[1].inverted_car_data.position[2],
                                'my_car_velocity_x': state.players[1].inverted_car_data.linear_velocity[0],
                                'my_car_velocity_y': state.players[1].inverted_car_data.linear_velocity[1],
                                'my_car_velocity_z': state.players[1].inverted_car_data.linear_velocity[2],
                                'my_car_rotation_yaw': state.players[1].inverted_car_data.angular_velocity[0],
                                'my_car_rotation_pitch': state.players[1].inverted_car_data.angular_velocity[1],
                                'my_car_rotation_roll': state.players[1].inverted_car_data.angular_velocity[2],
                                'enemy_car_x': state.players[0].inverted_car_data.position[0],
                                'enemy_car_y': state.players[0].inverted_car_data.position[1],
                                'enemy_car_z': state.players[0].inverted_car_data.position[2],
                                'enemy_car_velocity_x': state.players[0].inverted_car_data.linear_velocity[0],
                                'enemy_car_velocity_y': state.players[0].inverted_car_data.linear_velocity[1],
                                'enemy_car_velocity_z': state.players[0].inverted_car_data.linear_velocity[2],
                                'enemy_car_rotation_yaw': state.players[0].inverted_car_data.angular_velocity[0],
                                'enemy_car_rotation_pitch': state.players[0].inverted_car_data.angular_velocity[1],
                                'enemy_car_rotation_roll': state.players[0].inverted_car_data.angular_velocity[2],
                                'ball_x': state.inverted_ball.position[0],
                                'ball_y': state.inverted_ball.position[1],
                                'ball_z': state.inverted_ball.position[2],
                                'ball_velocity_x': state.inverted_ball.linear_velocity[0],
                                'ball_velocity_y': state.inverted_ball.linear_velocity[1],
                                'ball_velocity_z': state.inverted_ball.linear_velocity[2],
                                'ball_rotation_yaw': state.inverted_ball.angular_velocity[0],
                                'ball_rotation_pitch': state.inverted_ball.angular_velocity[1],
                                'ball_rotation_roll': state.inverted_ball.angular_velocity[2],
                                'my_team_score': state.blue_score,
                                'enemy_team_score': state.orange_score,
                                'remaining_time': 100},

                 'LOGIC': {'kickoff': False,
                           'overtime': False}
                 }

        actions = np.zeros(actions.shape)
        actions[0][0] = self.agent_1.eval_throttle(env_1)
        actions[0][1] = self.agent_1.eval_steering(env_1)
        actions[1][0] = self.agent_2.eval_throttle(env_2)
        actions[1][1] = self.agent_2.eval_steering(env_2)

        return actions


class RLLeagueState(StateSetter):
    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17], [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17], [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    def __init__(self, rl_league_action: RLLeagueAction):
        super().__init__()
        self.rl_league_action = rl_league_action

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"game_reports/game_{self.rl_league_action.agent_1.name}_{self.rl_league_action.agent_2.name}_{date_str}.txt"
        f = open(file_name, "w")
        f.write(f"{self.rl_league_action.rew_1}\n{self.rl_league_action.rew_2}")
        f.close()
        self.rl_league_action.rew_1 = 0.0
        self.rl_league_action.rew_2 = 0.0
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
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

def minimize_rl_windows():
    look_for_another = True
    def enumHandler(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            print(win32gui.GetWindowText(hwnd))
            if 'Rocket League' in win32gui.GetWindowText(hwnd):
                global look_for_another
                look_for_another = True
                print('found rl window')
                win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
                #rl_windows.append(hwnd)
                car.boost = 0.33

    while look_for_another:
        look_for_another = False
        win32gui.EnumWindows(enumHandler, None)
        

# If we want to spawn new processes, we have to make sure our program starts in a proper Python entry point.
if __name__ == "__main__":
    """
    Now all we have to do is make an instance of the SB3MultipleInstanceEnv and pass it our get_match function, the number of instances we'd like to open, and how long it should wait between instances.
    This wait_time argument is important because if multiple Rocket League clients are opened in quick succession, they will cause each other to crash. The exact reason this happens is unknown to us,
    but the easiest solution is to delay for some period of time between launching clients. The amount of required delay will depend on your hardware, so make sure to change this number if your Rocket League
    clients are crashing before they fully launch.
    """
    rl_league_action1 = RLLeagueAction(0, 1)
    rl_league_action2 = RLLeagueAction(2, 3)
    match1 = Match(
        reward_function=DefaultReward(),
        terminal_conditions=[],
        obs_builder=DefaultObs(),
        action_parser=rl_league_action1,
        state_setter=RLLeagueState(rl_league_action1),
        game_speed=100,
        spawn_opponents=True)
    match2 = Match(
        reward_function=DefaultReward(),
        terminal_conditions=[],
        obs_builder=DefaultObs(),
        action_parser=rl_league_action2,
        state_setter=RLLeagueState(rl_league_action2),
        game_speed=100,
        spawn_opponents=True)

    matches = [match1, match2]
    env = SB3MultipleInstanceEnv(match_func_or_matches=matches, num_instances=len(matches), wait_time=20)
    learner = PPO(policy="MlpPolicy", env=env, verbose=1)
    learner.learn(50_000)
    print("fin")
    learner.learn(50_000)
    env.reset()
