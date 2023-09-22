import pandas as pd
from initial_analysis import analyse

DATA_PATH = '/home/j/machine-learning/a4/Part1/data/original/diamonds.csv'


def load_data():
    return pd.read_csv(DATA_PATH, index_col=0)


def main():
    df = load_data()
    analyse(df)


if __name__ == "__main__":
    main()
