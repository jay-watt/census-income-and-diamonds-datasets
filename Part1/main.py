from utils import load_data, split_data
from initial_analysis import analyse
from preprocessing import preprocess


def main():
    data = load_data()
    class_name = analyse(data)
    train, _ = split_data(class_name, data)
    preprocess(class_name, train, 'train')


if __name__ == "__main__":
    main()
