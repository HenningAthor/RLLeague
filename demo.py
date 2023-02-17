from agent.agent import Agent
from league.match_scheduler import MatchScheduler
from recorded_data.data_util import load_min_max_csv, generate_env_stats


def random_agents():
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

    agent = Agent(0, f'agent_0', 5, 10, env_variables)
    agent.bloat_analysis(env_stats)
    agent.prepare_for_rlbot()
    agent = Agent(1, f'agent_1', 5, 10, env_variables)
    agent.bloat_analysis(env_stats)
    agent.prepare_for_rlbot()


if __name__ == '__main__':
    random_agents()

    ms = MatchScheduler(n_instances=2, game_speed=1, minimize_windows=True, verbose=1)
    ms.add_match(1, 3721, 469, 100_000)
    ms.add_match(1, 0, 1, 100_000)
    ms.simulate()
