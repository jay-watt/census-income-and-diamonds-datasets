import numpy as np
import pandas as pd
from common.config import ORIGINAL_DATA_DIR, SEED
from sklearn.model_selection import train_test_split


def load_original_data(filename):
    data = pd.read_csv(f'{ORIGINAL_DATA_DIR}/{filename}.csv', index_col=0)
    return data.replace(0, np.nan)


def split_data(class_name, data):
    train_X, test_X, train_y, test_y = train_test_split(
        data.drop(columns=class_name),
        data[class_name],
        test_size=0.3,
        random_state=SEED,
    )
    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)
    return train, test
