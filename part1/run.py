from part1.analysis import analyse
from part1.config import DATA_FILENAME
from part1.models import model
from part1.preprocessing import preprocess_and_eda
from part1.utils import load_original_data, split_data


def run_part1():
    data = load_original_data(DATA_FILENAME)
    class_name = analyse(data)
    train, test = split_data(class_name, data)
    preprocess_and_eda(class_name, train, 'train', 'part1')
    preprocess_and_eda(class_name, test, 'test', 'part1')
    # model(class_name)
