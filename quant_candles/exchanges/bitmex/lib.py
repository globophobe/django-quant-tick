import numpy as np
from pandas import DataFrame


def calculate_index(data_frame: DataFrame) -> DataFrame:
    """Calculate index."""
    symbols = data_frame.symbol.unique()
    data_frame["index"] = np.nan  # B/C pandas index
    for symbol in symbols:
        index = data_frame.index[data_frame.symbol == symbol]
        # 0-based index according to symbol
        data_frame.loc[index, "index"] = index.values - index.values[0]
    return data_frame
