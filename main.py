import os
import sys

import pandas as pd
from common.config import ORIGINAL_DATA_DIR, SEED
from part1.analysis import analyse
from part1.config import DATA_FILENAME
from part1.models import model
from part1.preprocessing import preprocess_and_eda
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_original_data(filename):
    return pd.read_csv(f'{ORIGINAL_DATA_DIR}/{filename}.csv', index_col=0)


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


def main():
    # Part 1
    data = load_original_data(DATA_FILENAME)
    class_name = analyse(data)
    train, test = split_data(class_name, data)
    preprocess_and_eda(class_name, train, 'train', 'part1')
    preprocess_and_eda(class_name, test, 'test', 'part1')
    model(class_name)

    # Part 2


if __name__ == "__main__":
    main()
