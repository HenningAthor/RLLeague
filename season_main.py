import os.path

from agent.agent import Agent
from league.league import League, league_exists, load_league_from_file
from league.reward_functions import league_reward_functions
from league.season_manager import SeasonManager
from recorded_data.data_util import load_min_max_csv, generate_env_stats
from typing import List
from pathlib import Path

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

    Path('league/league_storage').mkdir(exist_ok=True, parents=True)
    Path('league/pending_matches').mkdir(exist_ok=True, parents=True)
    Path('game_reports').mkdir(exist_ok=True, parents=True)

    LEAGUE_COUNT = 2
    leagues : List[League] = []

    for i in range(LEAGUE_COUNT):
        league_id = i + 1
        if league_exists(league_id):
            leagues.append(load_league_from_file(league_id))
        else:
            step_func, game_func = league_reward_functions[league_id]
            league = League(league_id=league_id, step_reward_func=step_func, game_reward_func=game_func, time_steps=1000)
            leagues.append(league)
            for agent_id in range(league_id * 10000, league_id * 10000 + 10):
                if not os.path.exists(f"agent_storage/agent_{agent_id}"):
                    agent = Agent(agent_id, f'agent_{agent_id}', 5, 10, env_variables)
                    agent.bloat_analysis(env_stats)
                    agent.prepare_for_rlbot()
                league.add_agent(agent_id)
        leagues[-1].save_to_file()

    season_manager = SeasonManager(n_instances=6, wait_time=30, minimize_windows=False, verbose=True, rlgym_verbose=False)
    for league in leagues:
        season_manager.add_league(league)

    for i in range(1000):
        for j in range(3):
            season_manager.simulate_one_season()
            season_manager.finish_season()
            season_manager.mutate_and_recombine()

        season_manager.simulate_one_season()
        season_manager.finish_season()
        season_manager.interchange_leagues()

        for league in leagues:
            league.save_to_file()
