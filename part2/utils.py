import numpy as np
import pandas as pd
from common.config import ORIGINAL_DATA_DIR
from part2.config import COLUMN_NAMES, DATA_FILENAMES


def load_original_data():
    train = pd.read_csv(
        f'{ORIGINAL_DATA_DIR}/{DATA_FILENAMES["train"]}',
        header=None,
        names=COLUMN_NAMES,
    )
    test = pd.read_csv(
        f'{ORIGINAL_DATA_DIR}/{DATA_FILENAMES["test"]}',
        header=None,
        names=COLUMN_NAMES,
        skiprows=1,
    )

    train = train.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    test = test.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    train.replace('?', np.nan, inplace=True)
    test.replace('?', np.nan, inplace=True)

    return train, test
