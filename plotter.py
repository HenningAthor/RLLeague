import itertools
import os.path
import pickle

import matplotlib.pyplot as plt
import glob

from PIL import Image

import numpy as np


def generate_single_plots():
    tree_names = ['Throttle', 'Steer', 'Pitch', 'Yaw', 'Roll', 'Jump', 'Boost', 'Handbrake']

    for file in glob.glob('temp_data/*.pickle'):
        file_name = os.path.basename(file)[:-7]
        print(file, file_name)

        parameters = file_name.split('_')
        mutate_p, recombine_p, min_size, max_size = parameters[0], parameters[1], parameters[2], parameters[3]

        print(mutate_p, recombine_p, min_size, max_size)

        errors, n_nodes, n_non_bloat_nodes = None, None, None

        with open(file, 'rb') as handle:
            errors, n_nodes, n_non_bloat_nodes = pickle.load(handle)

        avg_error_per_epoch = []
        min_error_per_epoch = []
        max_error_per_epoch = []

        avg_n_nodes_per_epoch = []
        min_n_nodes_per_epoch = []
        max_n_nodes_per_epoch = []

        avg_n_non_bloat_nodes_per_epoch = []
        min_n_non_bloat_nodes_per_epoch = []
        max_n_non_bloat_nodes_per_epoch = []

        for i in range(errors.shape[0]):
            avg_errors = np.average(errors[i], axis=0)
            max_errors = np.max(errors[i], axis=0)
            min_errors = np.min(errors[i], axis=0)

            avg_n_nodes = np.average(n_nodes[i], axis=0)
            max_n_nodes = np.max(n_nodes[i], axis=0)
            min_n_nodes = np.min(n_nodes[i], axis=0)

            avg_n_non_bloat_nodes = np.average(n_non_bloat_nodes[i], axis=0)
            max_n_non_bloat_nodes = np.max(n_non_bloat_nodes[i], axis=0)
            min_n_non_bloat_nodes = np.min(n_non_bloat_nodes[i], axis=0)

            avg_error_per_epoch.append(avg_errors)
            min_error_per_epoch.append(min_errors)
            max_error_per_epoch.append(max_errors)

            avg_n_nodes_per_epoch.append(avg_n_nodes)
            max_n_nodes_per_epoch.append(max_n_nodes)
            min_n_nodes_per_epoch.append(min_n_nodes)

            avg_n_non_bloat_nodes_per_epoch.append(avg_n_non_bloat_nodes)
            max_n_non_bloat_nodes_per_epoch.append(max_n_non_bloat_nodes)
            min_n_non_bloat_nodes_per_epoch.append(min_n_non_bloat_nodes)

        for i in range(8):
            avg_error_per_epoch_per_tree = []
            min_error_per_epoch_per_tree = []
            max_error_per_epoch_per_tree = []

            avg_n_nodes_per_epoch_per_tree = []
            min_n_nodes_per_epoch_per_tree = []
            max_n_nodes_per_epoch_per_tree = []

            avg_n_non_bloat_nodes_per_epoch_per_tree = []
            min_n_non_bloat_nodes_per_epoch_per_tree = []
            max_n_non_bloat_nodes_per_epoch_per_tree = []

            for j in range(errors.shape[0]):
                avg_error_per_epoch_per_tree.append(avg_error_per_epoch[j][i])
                min_error_per_epoch_per_tree.append(min_error_per_epoch[j][i])
                max_error_per_epoch_per_tree.append(max_error_per_epoch[j][i])

                avg_n_nodes_per_epoch_per_tree.append(avg_n_nodes_per_epoch[j][i])
                min_n_nodes_per_epoch_per_tree.append(min_n_nodes_per_epoch[j][i])
                max_n_nodes_per_epoch_per_tree.append(max_n_nodes_per_epoch[j][i])

                avg_n_non_bloat_nodes_per_epoch_per_tree.append(avg_n_non_bloat_nodes_per_epoch[j][i])
                min_n_non_bloat_nodes_per_epoch_per_tree.append(min_n_non_bloat_nodes_per_epoch[j][i])
                max_n_non_bloat_nodes_per_epoch_per_tree.append(max_n_non_bloat_nodes_per_epoch[j][i])

            # errors
            x = np.arange(errors.shape[0])
            y1 = np.array(avg_error_per_epoch_per_tree)
            y2 = np.array(min_error_per_epoch_per_tree)
            y3 = np.array(max_error_per_epoch_per_tree)

            fig, ax = plt.subplots()
            ax.plot(x, y1, label='Avg')
            ax.plot(x, y2, label='Min')
            ax.plot(x, y3, label='max')
            plt.title(f"{tree_names[i]} p1={mutate_p} p2={recombine_p} size=({min_size}, {max_size})")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Error", fontsize=12)
            plt.legend(loc="upper left")
            plt.savefig(f'temp_data/{tree_names[i]}_errors_{mutate_p}_{recombine_p}_{min_size}_{max_size}.png')
            plt.close()

            # n_nodes
            x = np.arange(errors.shape[0])
            y1 = np.array(avg_n_nodes_per_epoch_per_tree)
            y2 = np.array(min_n_nodes_per_epoch_per_tree)
            y3 = np.array(max_n_nodes_per_epoch_per_tree)

            fig, ax = plt.subplots()
            ax.plot(x, y1, label='Avg')
            ax.plot(x, y2, label='Min')
            ax.plot(x, y3, label='max')
            plt.title(f"{tree_names[i]} p1={mutate_p} p2={recombine_p} size=({min_size}, {max_size})")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("#Nodes", fontsize=12)
            plt.legend(loc="upper left")
            plt.savefig(f'temp_data/{tree_names[i]}_n_nodes_{mutate_p}_{recombine_p}_{min_size}_{max_size}.png')
            plt.close()

            # n_non_bloat_nodes
            x = np.arange(errors.shape[0])
            y1 = np.array(avg_n_non_bloat_nodes_per_epoch_per_tree)
            y2 = np.array(min_n_non_bloat_nodes_per_epoch_per_tree)
            y3 = np.array(max_n_non_bloat_nodes_per_epoch_per_tree)

            fig, ax = plt.subplots()
            ax.plot(x, y1, label='Avg')
            ax.plot(x, y2, label='Min')
            ax.plot(x, y3, label='max')
            plt.title(f"{tree_names[i]} p1={mutate_p} p2={recombine_p} size=({min_size}, {max_size})")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("#Non-Bloat Nodes", fontsize=12)
            plt.legend(loc="upper left")
            plt.savefig(f'temp_data/{tree_names[i]}_n_non_bloat_nodes_{mutate_p}_{recombine_p}_{min_size}_{max_size}.png')
            plt.close()


def combine_plots():
    tree_names = ['Throttle', 'Steer', 'Pitch', 'Yaw', 'Roll', 'Jump', 'Boost', 'Handbrake']
    mutate_probabilities = [0.001, 0.01, 0.05, 0.1]
    recombine_probabilities = [0.01, 0.125, 0.5]
    tree_sizes = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

    for p1, p2, (min_size, max_size) in itertools.product(mutate_probabilities, recombine_probabilities, tree_sizes):

        error_files = []
        n_node_files = []
        n_non_bloat_node_files = []
        for name in tree_names:
            error_files.append(f'temp_data/{name}_errors_{p1}_{p2}_{min_size}_{max_size}.png')
            n_node_files.append(f'temp_data/{name}_n_nodes_{p1}_{p2}_{min_size}_{max_size}.png')
            n_non_bloat_node_files.append(f'temp_data/{name}_n_non_bloat_nodes_{p1}_{p2}_{min_size}_{max_size}.png')

        # error image
        images = [Image.open(x) for x in error_files]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        error_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            error_image.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        # n_node image
        images = [Image.open(x) for x in n_node_files]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        n_node_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            n_node_image.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        # n_non_bloat_image
        images = [Image.open(x) for x in n_non_bloat_node_files]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        n_non_bloat_node_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            n_non_bloat_node_image.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        images = [error_image, n_node_image, n_non_bloat_node_image]
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        complete_image = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            complete_image.paste(im, (0, y_offset))
            y_offset += im.size[1]

        path = 'plots/'
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = f'{p1}_{p2}_{min_size}_{max_size}.png'

        complete_image.save(path+file_name)


if __name__ == '__main__':
    generate_single_plots()
    combine_plots()
