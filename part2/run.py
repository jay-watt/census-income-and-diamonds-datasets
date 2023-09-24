from part2.analysis import analyse
from part2.preprocessing import preprocess_and_eda
from part2.utils import load_original_data


def run_part2():
    train, test = load_original_data()
    class_name = analyse(train)
    redundant_features = preprocess_and_eda(
        class_name, train, 'train', 'part2', []
    )
    _ = preprocess_and_eda(
        class_name, test, 'test', 'part2', redundant_features
    )
    # model(class_name)
