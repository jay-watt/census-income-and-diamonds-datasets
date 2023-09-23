from initial_analysis import analyse
from preprocessing import preprocess_and_eda
from utils import load_data, split_data


def main():
    data = load_data()
    class_name = analyse(data)
    train, _ = split_data(class_name, data)
    preprocess_and_eda(class_name, train, 'train')


if __name__ == "__main__":
    main()
