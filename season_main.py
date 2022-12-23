import os.path

from genetic_lab.bot_generation import create_bot
from league.league import League
from league.reward_functions import league_reward_functions
from league.season_manager import SeasonManager

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

    step_func, game_func = league_reward_functions[1]
    league_1 = League(league_id=1, step_reward_func=step_func, game_reward_func=game_func, time_steps=16_384)
    league_2 = League(league_id=2, step_reward_func=step_func, game_reward_func=game_func, time_steps=16_384//2)

    for i in range(5):
        if not os.path.exists(f"bot_storage/bot_{i}"):
            bot = create_bot(i, 5, 10, env_variables)
            print(bot.info())
            bot.prepare_for_rlbot()
        league_1.add_agent(i)
        league_2.add_agent(i)

    season_manager = SeasonManager(n_instances=3, wait_time=30, minimize_windows=False, verbose=True, rlgym_verbose=False)
    season_manager.add_league(league_1)
    season_manager.add_league(league_2)
    season_manager.simulate_one_season()
    season_manager.finish_season()
