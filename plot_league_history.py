import numpy as np
from matplotlib import pyplot as plt

from league.league import league_exists, load_league_from_file

if __name__ == '__main__':
    league_1 = load_league_from_file(1)
    league_2 = load_league_from_file(2)

    min_ranking = []
    max_ranking = []
    avg_ranking = []

    for ranking in league_1.ranking_history:
        rewards = [rew for _, rew in ranking]
        min_ranking.append(min(rewards))
        max_ranking.append(max(rewards))
        avg_ranking.append(np.average(rewards))

    x = np.arange(len(min_ranking))
    y1 = np.array(avg_ranking)
    y2 = np.array(min_ranking)
    y3 = np.array(max_ranking)

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Avg')
    ax.plot(x, y2, label='Min')
    ax.plot(x, y3, label='max')
    plt.title(f"League 1 Rewards")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    plt.legend(loc="upper left")
    plt.savefig(f'temp_data/league_1.png')
    plt.close()

    min_ranking = []
    max_ranking = []
    avg_ranking = []

    for ranking in league_2.ranking_history:
        rewards = [rew for _, rew in ranking]
        min_ranking.append(min(rewards))
        max_ranking.append(max(rewards))
        avg_ranking.append(np.average(rewards))

    x = np.arange(len(min_ranking))
    y1 = np.array(avg_ranking)
    y2 = np.array(min_ranking)
    y3 = np.array(max_ranking)

    fig, ax = plt.subplots()
    ax.plot(x, y1, label='Avg')
    ax.plot(x, y2, label='Min')
    ax.plot(x, y3, label='max')
    plt.title(f"League 2 Rewards")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    plt.legend(loc="upper left")
    plt.savefig(f'temp_data/league_2.png')
    plt.close()
