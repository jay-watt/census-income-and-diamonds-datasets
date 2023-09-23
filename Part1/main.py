from initial_analysis import analyse
from models import model
from preprocessing import preprocess_and_eda
from utils import load_original_data, split_data


def main():
    data = load_original_data()
    class_name = analyse(data)
    train, test = split_data(class_name, data)
    preprocess_and_eda(class_name, train, 'train')
    preprocess_and_eda(class_name, test, 'test')
    model(class_name)


if __name__ == "__main__":
    main()
