from league.league import League
from league.match_scheduler import MatchScheduler
from typing import Dict


class SeasonManager(object):
    """
    This object manages all leagues and simulates the seasons for each league.
    """
    def __init__(self, n_instances, wait_time=20, minimize_windows=True, verbose=True, rlgym_verbose=False):
        """
        Initializes the season manager.

        :param n_instances: Number of windows for simulation.
        :param wait_time: How long to wait before opening another rl window.
        :param minimize_windows: If the windows should be minimized.
        :param verbose: If the class should print.
        :param rlgym_verbose: If rlgym should print (not everything can be disabled).
        """
        self.match_scheduler = MatchScheduler(n_instances=n_instances, wait_time=wait_time, minimize_windows=minimize_windows, verbose=verbose)
        self.leagues : Dict[int, League] = {}

    def simulate_one_season(self) -> None:
        """
        For each league a season will be simulated.

        :return: None
        """
        matches = []
        for league_id, league in self.leagues.items():
            matches += league.setup_matches()

        for match in matches:
            league_id, id_1, id_2 = match
            time_steps = self.leagues[league_id].time_steps
            self.match_scheduler.add_match(league_id, id_1, id_2, time_steps)

        self.match_scheduler.simulate()

        for league_id, league in self.leagues.items():
            league.read_game_reports()

    def finish_season(self) -> None:
        """
        Finishes the season.

        :return: None
        """
        for league_id, league in self.leagues.items():
            league.finish_season()

    def mutate_and_recombine(self) -> None:
        """
        In each league mutate and recombine the agents.

        :return: None
        """
        for league_id, league in self.leagues.items():
            league.mutate_and_recombine()

    def interchange_leagues(self):
        """
        Swaps agents in the leagues. It will take the top 3 agents of the lower
        league and swaps them with the lowest 3 agents of the upper league.

        :return: None
        """
        league_ids = list(self.leagues.keys())

        for i in range(1, len(league_ids)-1):
            lower_league_ranking = self.leagues[i].ranking.copy()
            upper_league_ranking = self.leagues[i+1].ranking.copy()

            agent_id_1, _ = lower_league_ranking.pop()
            agent_id_2, _ = lower_league_ranking.pop()
            agent_id_3, _ = lower_league_ranking.pop()

            agent_id_4, _ = upper_league_ranking.pop(0)
            agent_id_5, _ = upper_league_ranking.pop(0)
            agent_id_6, _ = upper_league_ranking.pop(0)

            self.leagues[i].agent_ids.remove(agent_id_1)
            self.leagues[i].agent_ids.remove(agent_id_2)
            self.leagues[i].agent_ids.remove(agent_id_3)
            self.leagues[i].add_agent(agent_id_4)
            self.leagues[i].add_agent(agent_id_5)
            self.leagues[i].add_agent(agent_id_6)

            self.leagues[i+1].agent_ids.remove(agent_id_4)
            self.leagues[i+1].agent_ids.remove(agent_id_5)
            self.leagues[i+1].agent_ids.remove(agent_id_6)
            self.leagues[i+1].add_agent(agent_id_1)
            self.leagues[i+1].add_agent(agent_id_2)
            self.leagues[i+1].add_agent(agent_id_3)

    def add_league(self, league: League) -> None:
        """
        Adds a league to the season manager.

        :param league: The league to add.
        :return: None
        """
        assert league.league_id not in self.leagues.keys()
        self.leagues[league.league_id] = league
