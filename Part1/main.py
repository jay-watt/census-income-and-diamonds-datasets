import pandas as pd
from sklearn.model_selection import train_test_split
from initial_analysis import analyse_raw_data
from preprocessing import preprocess
from exploratory_analysis import analyse_cleaned_data

DATA_PATH = '/home/j/machine-learning/a4/Part1/data/original/diamonds.csv'
SEED = 309


def load_data():
    return pd.read_csv(DATA_PATH, index_col=0)


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
    data = load_data()
    class_name = analyse_raw_data(data)
    train, _ = split_data(class_name, data)
    cleaned_train = preprocess(class_name, train, 'train')
    analyse_cleaned_data(class_name, cleaned_train)


if __name__ == "__main__":
    main()
