"""
Script to analyze the downloaded matches. It will determine the min and max value
for each column over all matches. This information can help with determining
bloat in the evolution trees.
The result will be printed in the min_max.csv file and saved in recorded_data.
The first row are the max values and the second row are the min values.
The script needs about 30 min to complete.
"""
import numpy as np
from tqdm import tqdm

from recorded_data.data_util import load_all_parquet_paths, load_match

if __name__ == '__main__':
    file_list = load_all_parquet_paths()

    headers = []
    min_values = []
    max_values = []

    for file_path in tqdm(file_list):
        game_data, file_headers = load_match(file_path)

        # initialize the headers
        if not headers:
            headers = file_headers
            min_values = [float('inf') for i in range(len(headers))]
            max_values = [-float('inf') for i in range(len(headers))]

        for i, name in enumerate(file_headers):
            idx = headers.index(name)

            min_values[i] = min(min_values[i], np.min(game_data[:, i]))
            max_values[i] = max(max_values[i], np.max(game_data[:, i]))

    s = ','.join(headers) + '\n'
    s = s + ','.join([str(x) for x in max_values]) + '\n'
    s = s + ','.join([str(x) for x in min_values])

    f = open('recorded_data/min_max.csv', 'w')
    f.write(s)
    f.close()
