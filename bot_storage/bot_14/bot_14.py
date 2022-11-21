import pickle
import os
import time

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, GameInfoState
from src.util.boost_pad_tracker import BoostPadTracker
from src.util.sequence import Sequence
from src.util.vec import Vec3


class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

        # index for extracting the game package
        self.enemy_index = -1

        print(os.getcwd())
        file = open(f'bot_14/bot_14.pickle', 'rb')
        self.bot = pickle.load(file)
        file.close()

        self.min_steering = float('inf')
        self.max_steering = -float('inf')
        self.min_throttle = float('inf')
        self.max_throttle = -float('inf')

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active
        # and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())
        game_info_state = GameInfoState(game_speed=4.0)
        game_state = GameState(game_info=game_info_state)
        self.set_game_state(game_state)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second.
        This is where you can see the motion of the ball, etc. and return
        controls to drive your car.
        """
        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # determine which car id we have and which one the enemy has
        if self.enemy_index == -1:
            for i in range(64):
                car = packet.game_cars[i]
                car_x = Vec3(car.physics.location).x
                if car_x != 0.0 and i != self.index:
                    self.enemy_index = i

        # read the features of our car
        my_car = packet.game_cars[self.index]
        my_car_location = Vec3(my_car.physics.location)
        my_car_velocity = Vec3(my_car.physics.velocity)
        my_car_rotation = my_car.physics.rotation

        # read the features of the enemy car
        enemy_car = packet.game_cars[self.enemy_index]
        enemy_car_location = Vec3(enemy_car.physics.location)
        enemy_car_velocity = Vec3(enemy_car.physics.velocity)
        enemy_car_rotation = enemy_car.physics.rotation

        # read the features of the ball
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        ball_rotation = packet.game_ball.physics.rotation

        # read the features of the game
        my_team = packet.teams[self.index]
        enemy_team = packet.teams[self.enemy_index]

        env = {'ARITHMETIC': {'my_car_x': my_car_location.x,
                              'my_car_y': my_car_location.y,
                              'my_car_z': my_car_location.z,
                              'my_car_velocity_x': my_car_velocity.x,
                              'my_car_velocity_y': my_car_velocity.y,
                              'my_car_velocity_z': my_car_velocity.z,
                              'my_car_rotation_yaw': my_car_rotation.yaw,
                              'my_car_rotation_pitch': my_car_rotation.pitch,
                              'my_car_rotation_roll': my_car_rotation.roll,
                              'enemy_car_x': enemy_car_location.x,
                              'enemy_car_y': enemy_car_location.y,
                              'enemy_car_z': enemy_car_location.z,
                              'enemy_car_velocity_x': enemy_car_velocity.x,
                              'enemy_car_velocity_y': enemy_car_velocity.y,
                              'enemy_car_velocity_z': enemy_car_velocity.z,
                              'enemy_car_rotation_yaw': enemy_car_rotation.yaw,
                              'enemy_car_rotation_pitch': enemy_car_rotation.pitch,
                              'enemy_car_rotation_roll': enemy_car_rotation.roll,
                              'ball_x': ball_location.x,
                              'ball_y': ball_location.y,
                              'ball_z': ball_location.z,
                              'ball_velocity_x': ball_velocity.x,
                              'ball_velocity_y': ball_velocity.y,
                              'ball_velocity_z': ball_velocity.z,
                              'ball_rotation_yaw': ball_rotation.yaw,
                              'ball_rotation_pitch': ball_rotation.pitch,
                              'ball_rotation_roll': ball_rotation.roll,
                              'my_team_score': my_team.score,
                              'enemy_team_score': enemy_team.score,
                              'remaining_time': packet.game_info.game_time_remaining},

               'LOGIC': {'kickoff': int(packet.game_info.is_kickoff_pause) == 1,
                         'overtime': int(packet.game_info.is_overtime) == 1}
               }

        t = time.time()
        steer = self.bot.eval_steering(env)
        t_steer = time.time() - t

        t = time.time()
        throttle = self.bot.eval_throttle(env)
        t_throttle = time.time() - t

        self.max_steering = max(self.max_steering, steer)
        self.min_steering = min(self.min_steering, steer)
        self.max_throttle = max(self.max_throttle, throttle)
        self.min_throttle = min(self.min_throttle, throttle)

        norm_steer = 0.0
        if self.max_steering - self.min_steering != 0.0:
            norm_steer = (steer - self.min_steering) / (self.max_steering - self.min_steering)
            norm_steer = norm_steer * (1 - -1) + -1

        norm_throttle = 0.0
        if self.max_throttle - self.min_throttle != 0.0:
            norm_throttle = (throttle - self.min_throttle) / (self.max_throttle - self.min_throttle)
            norm_throttle = norm_throttle * (1 - -1) + -1

        print(self.bot.name, norm_steer, norm_throttle, t_steer, t_throttle)

        controls = SimpleControllerState()
        controls.steer = norm_steer
        controls.throttle = norm_throttle

        return controls
