import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if __name__ == '__main__':
    # load all data
    error_normal, n_nodes_normal, _ = pickle.load(open('temp_data/0.01_8_10.pickle', 'rb'))
    error_a2, n_nodes_a2, _ = pickle.load(open('temp_data/autoencoder2_0.01_8_10.pickle', 'rb'))
    error_a4, n_nodes_a4, _ = pickle.load(open('temp_data/autoencoder4_0.01_8_10.pickle', 'rb'))
    error_a8, n_nodes_a8, _ = pickle.load(open('temp_data/autoencoder8_0.01_8_10.pickle', 'rb'))
    error_a16, n_nodes_a16, _ = pickle.load(open('temp_data/autoencoder16_0.01_8_10.pickle', 'rb'))
    error_a32, n_nodes_a32, _ = pickle.load(open('temp_data/autoencoder32_0.01_8_10.pickle', 'rb'))

    error_normal_max, error_normal_min, error_normal_avg = np.max(error_normal, axis=1), np.min(error_normal, axis=1), np.average(error_normal, axis=1)
    error_a2_max, error_a2_min, error_a2_avg = np.max(error_a2, axis=1), np.min(error_a2, axis=1), np.average(error_a2, axis=1)
    error_a4_max, error_a4_min, error_a4_avg = np.max(error_a4, axis=1), np.min(error_a4, axis=1), np.average(error_a4, axis=1)
    error_a8_max, error_a8_min, error_a8_avg = np.max(error_a8, axis=1), np.min(error_a8, axis=1), np.average(error_a8, axis=1)
    error_a16_max, error_a16_min, error_a16_avg = np.max(error_a16, axis=1), np.min(error_a16, axis=1), np.average(error_a16, axis=1)
    error_a32_max, error_a32_min, error_a32_avg = np.max(error_a32, axis=1), np.min(error_a32, axis=1), np.average(error_a32, axis=1)

    # errors
    x = np.arange(error_normal_max.shape[0])

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    ax.plot(x, error_normal_avg, label='Original Parameters')
    ax.plot(x, error_a2_avg, label='Autoencoder 2')
    ax.plot(x, error_a4_avg, label='Autoencoder 4')
    ax.plot(x, error_a8_avg, label='Autoencoder 8')
    ax.plot(x, error_a16_avg, label='Autoencoder 16')
    ax.plot(x, error_a32_avg, label='Autoencoder 32')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    plt.title(f"Average Error per Generation")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    plt.yscale('log')
    plt.legend(loc="upper right")
    plt.savefig(f'temp_data/avg.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    ax.plot(x, error_normal_min, label='Original Parameters')
    ax.plot(x, error_a2_min, label='Autoencoder 2')
    ax.plot(x, error_a4_min, label='Autoencoder 4')
    ax.plot(x, error_a8_min, label='Autoencoder 8')
    ax.plot(x, error_a16_min, label='Autoencoder 16')
    ax.plot(x, error_a32_min, label='Autoencoder 32')
    plt.title(f"Minimal Error per Generation")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    plt.yscale('log')
    plt.legend(loc="upper right")
    plt.savefig(f'temp_data/min.png', dpi=300)
    plt.close()

    n_nodes_normal_top = []
    n_nodes_a2_top = []
    n_nodes_a4_top = []
    n_nodes_a8_top = []
    n_nodes_a16_top = []
    n_nodes_a32_top = []

    for i in range(error_normal.shape[0]):
        n_nodes_normal_top.append(n_nodes_normal[i, np.argmin(error_normal[i])])
        n_nodes_a2_top.append(n_nodes_a2[i, np.argmin(error_a2[i])])
        n_nodes_a4_top.append(n_nodes_a4[i, np.argmin(error_a4[i])])
        n_nodes_a8_top.append(n_nodes_a8[i, np.argmin(error_a8[i])])
        n_nodes_a16_top.append(n_nodes_a16[i, np.argmin(error_a16[i])])
        n_nodes_a32_top.append(n_nodes_a32[i, np.argmin(error_a32[i])])

    n_nodes_normal_top = np.array(n_nodes_normal_top)
    n_nodes_a2_top = np.array(n_nodes_a2_top)
    n_nodes_a4_top = np.array(n_nodes_a4_top)
    n_nodes_a8_top = np.array(n_nodes_a8_top)
    n_nodes_a16_top = np.array(n_nodes_a16_top)
    n_nodes_a32_top = np.array(n_nodes_a32_top)

    # nodes
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    ax.plot(x, n_nodes_normal_top, label='Original Parameters')
    ax.plot(x, n_nodes_a2_top, label='Autoencoder 2')
    ax.plot(x, n_nodes_a4_top, label='Autoencoder 4')
    ax.plot(x, n_nodes_a8_top, label='Autoencoder 8')
    ax.plot(x, n_nodes_a16_top, label='Autoencoder 16')
    ax.plot(x, n_nodes_a32_top, label='Autoencoder 32')
    plt.title(f"Number of Nodes of the Best Tree in each Generation")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("#Nodes", fontsize=12)
    plt.legend(loc="best")
    plt.savefig(f'temp_data/n_nodes.png', dpi=300)
    plt.close()

    # autoencoder losses
    a2_loss = np.array(pickle.load(open('autoencoder/autoencoder_2_loss.pickle', 'rb')))
    a4_loss = np.array(pickle.load(open('autoencoder/autoencoder_4_loss.pickle', 'rb')))
    a8_loss = np.array(pickle.load(open('autoencoder/autoencoder_8_loss.pickle', 'rb')))
    a16_loss = np.array(pickle.load(open('autoencoder/autoencoder_16_loss.pickle', 'rb')))
    a32_loss = np.array(pickle.load(open('autoencoder/autoencoder_32_loss.pickle', 'rb')))

    print(error_normal_min)

    x = np.arange(a2_loss.shape[0])

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    ax.plot(x, a2_loss, label='Autoencoder 2')
    ax.plot(x, a4_loss, label='Autoencoder 4')
    ax.plot(x, a8_loss, label='Autoencoder 8')
    ax.plot(x, a16_loss, label='Autoencoder 16')
    ax.plot(x, a32_loss, label='Autoencoder 32')
    plt.title(f"Loss per Epoch")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    # plt.yscale('log')
    plt.legend(loc="upper center")
    plt.savefig(f'temp_data/loss.png', dpi=300)
    plt.close()


