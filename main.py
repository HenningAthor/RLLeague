import rlbot.runner as runner

from genetic_lab.bot_generation import create_bot

if __name__ == '__main__':
    env_variables = {'ARITHMETIC': ['my_car_x',
                                    'my_car_y',
                                    'my_car_z',
                                    'my_car_velocity_x',
                                    'my_car_velocity_y',
                                    'my_car_velocity_z',
                                    'my_car_rotation_yaw',
                                    'my_car_rotation_pitch',
                                    'my_car_rotation_roll',
                                    'enemy_car_x',
                                    'enemy_car_y',
                                    'enemy_car_z',
                                    'enemy_car_velocity_x',
                                    'enemy_car_velocity_y',
                                    'enemy_car_velocity_z',
                                    'enemy_car_rotation_yaw',
                                    'enemy_car_rotation_pitch',
                                    'enemy_car_rotation_roll',
                                    'ball_x', 'ball_y', 'ball_z',
                                    'ball_velocity_x',
                                    'ball_velocity_y',
                                    'ball_velocity_z',
                                    'ball_rotation_yaw',
                                    'ball_rotation_pitch',
                                    'ball_rotation_roll'],
                     'LOGIC': ['kickoff']}

    bot = create_bot(0, 5, 10, env_variables)
    print(bot.info())
    bot.prepare_for_rlbot()
    bot = create_bot(1, 5, 10, env_variables)
    print(bot.info())
    bot.prepare_for_rlbot()
    bot = create_bot(2, 5, 10, env_variables)
    print(bot.info())
    bot.prepare_for_rlbot()
    bot = create_bot(3, 5, 10, env_variables)
    print(bot.info())
    bot.prepare_for_rlbot()
    runner.main()
