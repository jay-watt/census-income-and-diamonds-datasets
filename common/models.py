import pandas as pd

from common.config import CLEANED_DATA_DIR


def load_cleaned_data(filename):
    data = pd.read_csv(f'{CLEANED_DATA_DIR}/{filename}.csv', index_col=0)
    class_ = data.columns[:-1]
    return data.drop(columns=class_), data[class_]
