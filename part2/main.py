from initial_analysis import analyse
from utils import load_original_data


def main():
    train, test = load_original_data()
    analyse(train)


if __name__ == "__main__":
    main()
