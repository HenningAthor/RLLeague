import os.path

from agent.agent import Agent
from league.league import League, league_exists, load_league_from_file
from league.reward_functions import league_reward_functions
from league.season_manager import SeasonManager
from recorded_data.data_util import load_min_max_csv, generate_env_stats

if __name__ == '__main__':
    env_variables = {'ARITHMETIC': ['ball/pos_x',
                                    'ball/pos_y',
                                    'ball/pos_z',
                                    'ball/vel_x',
                                    'ball/vel_y',
                                    'ball/vel_z',
                                    'ball/ang_vel_x',
                                    'ball/ang_vel_y',
                                    'ball/ang_vel_z',
                                    'player1/pos_x',
                                    'player1/pos_y',
                                    'player1/pos_z',
                                    'player1/vel_x',
                                    'player1/vel_y',
                                    'player1/vel_z',
                                    'player1/ang_vel_x',
                                    'player1/ang_vel_y',
                                    'player1/ang_vel_z',
                                    'player1/boost_amount',
                                    'inverted_player2/pos_x',
                                    'inverted_player2/pos_y',
                                    'inverted_player2/pos_z',
                                    'inverted_player2/vel_x',
                                    'inverted_player2/vel_y',
                                    'inverted_player2/vel_z',
                                    'inverted_player2/ang_vel_x',
                                    'inverted_player2/ang_vel_y',
                                    'inverted_player2/ang_vel_z',
                                    'player2/boost_amount'],
                     'LOGIC': ['player1/on_ground',
                               'player1/ball_touched',
                               'player1/has_jump',
                               'player1/has_flip',
                               'player2/on_ground',
                               'player2/ball_touched',
                               'player2/has_jump',
                               'player2/has_flip']}

    min_max_data, min_max_headers = load_min_max_csv()
    env_stats = generate_env_stats(env_variables, min_max_data, min_max_headers)

    step_func, game_func = league_reward_functions[1]

    if league_exists(1):
        league_1 = load_league_from_file(1)
    else:
        league_1 = League(league_id=1, step_reward_func=step_func, game_reward_func=game_func, time_steps=1000)
        for i in range(10000, 10010):
            if not os.path.exists(f"agent_storage/agent_{i}"):
                agent = Agent(i, f'agent_{i}', 5, 10, env_variables)
                agent.bloat_analysis(env_stats)
                agent.prepare_for_rlbot()
            league_1.add_agent(i)

    if league_exists(2):
        league_2 = load_league_from_file(2)
    else:
        league_2 = League(league_id=2, step_reward_func=step_func, game_reward_func=game_func, time_steps=1000)
        for i in range(20000, 20010):
            if not os.path.exists(f"agent_storage/agent_{i}"):
                agent = Agent(i, f'agent_{i}', 5, 10, env_variables)
                agent.bloat_analysis(env_stats)
                agent.prepare_for_rlbot()
            league_2.add_agent(i)

    league_1.save_to_file()
    league_2.save_to_file()

    season_manager = SeasonManager(n_instances=4, wait_time=30, minimize_windows=False, verbose=True, rlgym_verbose=False)
    season_manager.add_league(league_1)
    season_manager.add_league(league_2)

    for i in range(10):
        for j in range(3):
            season_manager.simulate_one_season()
            season_manager.finish_season()
            season_manager.mutate_and_recombine()

        season_manager.simulate_one_season()
        season_manager.finish_season()
        season_manager.interchange_leagues()

        league_1.save_to_file()
        league_2.save_to_file()
