import os.path
import pickle
import random
from typing import List, Tuple

import numpy as np

from agent.agent import recombine_agents
from recorded_data.data_util import load_min_max_csv, generate_env_stats


class League(object):

    def __init__(self, league_id: int, step_reward_func: callable, game_reward_func: callable, time_steps=16_384):
        """
        Initializes the league.
        Makes sure that all needed agents for the league are present, if bots
        are missing (there is no agent in agent_storage) a new agent will be
        generated.

        :param league_id: ID of the league. Lower ids are leagues with less complex reward functions.
        :param step_reward_func: Reward function, which is given at each step of the game.
        :param game_reward_func: Reward function, which is given at the end of the game.
        :param time_steps: Length of a game (Use powers of 2 for best results, 50_000 are roughly 30 minutes in game time).
        """
        self.n_played_seasons = 0
        self.league_id = league_id
        self.step_reward_function = step_reward_func
        self.game_reward_func = game_reward_func
        self.time_steps = time_steps

        self.agent_ids = []
        self.match_history = []
        self.started_matches = []

        self.ranking = []
        self.ranking_history = []

        self.season_history = []

    def save_to_file(self) -> None:
        """
        Saves the league to a file, so it can be loaded later.

        :return: None
        """
        path = 'league/league_storage/'
        if not os.path.exists(path):
            os.mkdir(path)

        file_name = f'league_{self.league_id}.pickle'
        full_path = path + file_name
        with open(full_path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def setup_matches(self) -> List[Tuple[int, int, int]]:
        """
        Creates a list of all matches, which should be played in this season.
        Currently, everyone plays against everyone.

        :return: List of tuple (league, id1, id2)
        """
        matches = []
        for i in range(len(self.agent_ids)):
            for j in range(i + 1, len(self.agent_ids)):
                matches.append((self.league_id, self.agent_ids[i], self.agent_ids[j]))
        self.started_matches = matches
        return matches

    def read_game_reports(self) -> None:
        """
        Reads the generated game reports and saves them internally.
        Deletes the file.

        :return: None
        """
        for (league_id, id_1, id_2) in self.started_matches:
            file_path = f"game_reports/{league_id}_{id_1}_{id_2}.txt"
            if os.path.exists(file_path):
                f = open(file_path, "r")
                rew_1 = float(f.readline().replace('\n', ''))
                rew_2 = float(f.readline().replace('\n', ''))
                f.close()
                os.remove(file_path)
                self.match_history.append((self.n_played_seasons, league_id, id_1, id_2, rew_1, rew_2))

    def finish_season(self) -> None:
        """
        This function is called, when all matches have been played.

        :return: None
        """
        self.season_history.append(self.match_history)
        self.compute_avg_reward()
        self.match_history = []
        self.n_played_seasons = 1

    def add_agent(self, agent_id: int) -> None:
        """
        Adds the agent to league.

        :param agent_id: ID of the agent.
        :return: None
        """
        assert agent_id not in self.agent_ids
        self.agent_ids.append(agent_id)

    def compute_avg_reward(self) -> None:
        """
        Computes the ranking of the agents, based on the matches played.

        :return: None
        """
        matches_per_agent = {agent_id: [] for agent_id in self.agent_ids}

        for match in self.match_history:
            n_played_seasons, league_id, id_1, id_2, rew_1, rew_2 = match

            matches_per_agent[id_1].append(rew_1)
            matches_per_agent[id_2].append(rew_2)

        reward_per_agent = {agent_id: 0 for agent_id in self.agent_ids}

        for agent_id, matches in matches_per_agent.items():
            reward_per_agent[agent_id] = sum(matches) / len(matches)

        temp = sorted(reward_per_agent.items(), key=lambda x: x[1])
        self.ranking = list(reversed(temp))
        self.ranking_history.append(self.ranking)
        self.print_last_ranking()

    def print_last_ranking(self) -> None:
        longest_agent_id = 0
        longest_reward = 0

        for (agent_id, reward) in self.ranking:
            longest_agent_id = max(len(str(agent_id)), longest_agent_id)
            longest_reward = max(len(str(reward)), longest_reward)

        s = f'Results for league {self.league_id} in season {self.n_played_seasons}\n'
        s += 'Agent_ID\t\t\tAvg.Reward\n'
        for (agent_id, reward) in self.ranking:
            s += f'{agent_id}\t{reward}\n'
        print(s)

    def mutate_and_recombine(self, keep_best_count = 3) -> None:
        """
        Mutates and recombines the agents. The top 3 agents will always be kept.
        The remaining agents will be replaced by new ones.

        :return: None
        """

        available_agents = self.agent_ids.copy()

        new_agent_ids = [self.ranking[i][0] for i in range(keep_best_count)]
        for agent_idx in new_agent_ids:
            available_agents.remove(agent_idx)

        new_agents = [pickle.load(open(f"agent_storage/agent_{agent_idx}/agent_{agent_idx}.pickle", 'rb')) for agent_idx in new_agent_ids]

        rewards = [rew for _, rew in self.ranking]
        agents = [agent_id for agent_id, _ in self.ranking]
        sum_rewards = sum(rewards)
        rel_rewards = [rew / sum_rewards for rew in rewards]

        while available_agents:
            if random.random() < 0.5:
                agent_1_idx = random.choices(agents, rel_rewards)[0]
                agent_2_idx = random.choices(agents, rel_rewards)[0]

                agent_1 = pickle.load(open(f"agent_storage/agent_{agent_1_idx}/agent_{agent_1_idx}.pickle", 'rb'))
                agent_2 = pickle.load(open(f"agent_storage/agent_{agent_2_idx}/agent_{agent_2_idx}.pickle", 'rb'))

                rec_agent_1, _ = recombine_agents(agent_1, agent_2, 0.125)
                rec_agent_1.agent_id = available_agents.pop()
                rec_agent_1.name = f'agent_{rec_agent_1.agent_id}'

                new_agent_ids.append(rec_agent_1.agent_id)
                new_agents.append(rec_agent_1)
            else:
                agent_idx = random.choices(agents, rel_rewards)[0]
                agent = pickle.load(open(f"agent_storage/agent_{agent_idx}/agent_{agent_idx}.pickle", 'rb'))

                mutated_agent = agent.mutate(0.01)
                mutated_agent.agent_id = available_agents.pop()
                mutated_agent.name = f'agent_{mutated_agent.agent_id}'

                new_agent_ids.append(mutated_agent.agent_id)
                new_agents.append(mutated_agent)

        env_variables = new_agents[0].creation_variables
        min_max_data, min_max_headers = load_min_max_csv()
        env_stats = generate_env_stats(env_variables, min_max_data, min_max_headers)

        for agent in new_agents:
            agent.bloat_analysis(env_stats)
            agent.prepare_for_rlbot()

        self.agent_ids = new_agent_ids


def league_exists(league_id: int) -> bool:
    """
    Checks if the league with the specified if exists.

    :param league_id: ID of the league.
    :return: True if a file is present, False else.
    """
    path = 'league/league_storage/'
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = f'league_{league_id}.pickle'
    full_path = path + file_name
    return os.path.exists(full_path)


def load_league_from_file(league_id: int) -> 'League':
    """
    Loads the league from a file.

    :return: The league.
    """
    path = 'league/league_storage/'
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = f'league_{league_id}.pickle'
    full_path = path + file_name
    with open(full_path, 'rb') as file:
        return pickle.load(file)
