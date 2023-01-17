import pandas as pd
import re

if __name__ == '__main__':
    df = pd.read_parquet('recorded_data/out/a7303c84-7d9e-472d-aa60-403388827a41.parquet')

    column_names = list(df.columns)
    data_types = list(df.dtypes)

    for name, dtype in zip(column_names, data_types):
        print(name, dtype)
        if str(dtype) == "object":
            print("dropped")
            df = df.drop(columns=[name])
        else:
            df[name] = df[name].astype(float)

    column_names = list(df.columns)
    data_types = list(df.dtypes)
    for name, dtype in zip(column_names, data_types):
        print(name, dtype)

    player_ids = []
    for name in column_names:
        if re.search(r'\d+(?:\.\d+)?/car_id', name):
            player_ids.append(name[:-7])
    print("player_ids", player_ids)

    for i in range(len(column_names)):
        if player_ids[0] in column_names[i]:
            column_names[i] = column_names[i].replace(player_ids[0], 'player1')

        if player_ids[1] in column_names[i]:
            column_names[i] = column_names[i].replace(player_ids[1], 'player2')

    for name in column_names:
        print(name)

    current_column_names = list(df.columns)
    rename_dict = dict(zip(current_column_names, column_names))
    df = df.rename(columns=rename_dict)

    for col in df.columns:
        print(col)

    df.to_csv('recorded_data/res.csv', index=False)


