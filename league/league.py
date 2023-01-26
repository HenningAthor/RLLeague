import os.path
from typing import List, Tuple


class League(object):

    def __init__(self, league_id: int, step_reward_func: callable, game_reward_func: callable, time_steps=16_384):
        """
        Initializes the league.
        Makes sure that all needed agents for the league are present, if bots
        are missing (there is no agent in agent_storage) a new agent will be
        generated.

        :param league_id: Id of the league. Lower ids are leagues with less complex reward functions.
        :param step_reward_func: Reward function, which is given at each step of the game.
        :param game_reward_func: Reward function, which is given at the end of the game.
        :param time_steps: Length of a game (Use powers of 2 for best results, 50_000 are roughly 30 minutes in game time).
        """
        self.league_id = league_id
        self.step_reward_function = step_reward_func
        self.game_reward_func = game_reward_func
        self.time_steps = time_steps

        self.agent_ids = []
        self.match_history = []
        self.started_matches = []

        self.season_history = []

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
                self.match_history.append((league_id, id_1, id_2, rew_1, rew_2))
                print(f"Started match {league_id, id_1, id_2}, Added game report!")

            else:
                print(f"Started match {league_id, id_1, id_2}, but no report found!")

    def finish_season(self) -> None:
        """
        This function is called, when all matches have been played.

        :return: None
        """
        self.season_history.append(self.match_history)
        self.match_history = []

    def add_agent(self, agent_id: int) -> None:
        """
        Adds the agent to league.

        :param agent_id: Id of the agent.
        :return: None
        """
        assert agent_id not in self.agent_ids
        self.agent_ids.append(agent_id)
