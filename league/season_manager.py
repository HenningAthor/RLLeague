from league.league import League
from league.match_scheduler import MatchScheduler


class SeasonManager(object):
    """
    This object manages all leagues and simulates the seasons for each league.
    """
    def __init__(self, n_instances, time_steps_per_instance, wait_time=20, minimize_windows=True, verbose=True):
        """
        Initializes the season manager.

        :param n_instances: Number of windows for simulation.
        :param time_steps_per_instance: How long the bots play against each other.
        :param wait_time: How long to wait before opening another rl window.
        :param minimize_windows: If the windows should be minimized.
        :param verbose: If the class should print.
        """
        self.match_scheduler = MatchScheduler(n_instances=n_instances, time_steps_per_instance=time_steps_per_instance, wait_time=wait_time, minimize_windows=minimize_windows, verbose=verbose)
        self.leagues = {}

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
            self.match_scheduler.add_match(league_id, id_1, id_2)

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

    def add_league(self, league: League) -> None:
        """
        Adds a league to the season manager.

        :param league: The league to add.
        :return: None
        """
        assert league.league_id not in self.leagues.keys()
        self.leagues[league.league_id] = league
