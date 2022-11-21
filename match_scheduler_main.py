from league.match_scheduler import MatchScheduler

if __name__ == '__main__':
    ms = MatchScheduler(n_instances=4, time_steps_per_instance=1_000, minimize_windows=True, verbose=1)
    for i in range(0, 4):
        for j in range(i+1, 4):
            ms.add_match(1, i, j)
    ms.simulate()
    for i in range(0, 4):
        for j in range(i+1, 4):
            ms.add_match(2, i, j)
    ms.simulate()
