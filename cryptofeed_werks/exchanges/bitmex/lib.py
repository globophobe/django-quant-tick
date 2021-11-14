import numpy as np


def calculate_index(data_frame):
    symbols = data_frame.symbol.unique()
    data_frame["index"] = np.nan  # B/C pandas index
    for symbol in symbols:
        index = data_frame.index[data_frame.symbol == symbol]
        # 0-based index according to symbol
        data_frame.loc[index, "index"] = index.values - index.values[0]
    return data_frame
