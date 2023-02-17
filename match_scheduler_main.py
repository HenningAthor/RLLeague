
import numpy as np


from league.match_scheduler import MatchScheduler

if __name__ == '__main__':
    ms = MatchScheduler(n_instances=1, minimize_windows=True, verbose=1)
    ms.add_match(1, 0, 132, 1_000)
    ms.add_match(1, 1, 186, 1_000)
    ms.add_match(1, 446, 575, 1_000)
    ms.add_match(1, 592, 790, 1_000)
    ms.simulate()
    ms.add_match(1, 929, 1422, 1_000)
    ms.add_match(1, 1516, 1647, 1_000)
    ms.add_match(1, 1751, 1864, 1_000)
    ms.add_match(1, 1943, 1951, 1_000)
    ms.simulate()
