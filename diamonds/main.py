import argparse
import os
import sys
import warnings

diamonds_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(diamonds_dir)
sys.path.append(parent_dir)

warnings.filterwarnings("ignore")

#fmt: off
from common.config import CLEANED_DATA_DIR
from modelling import run_modelling
from preprocessing import run_preprocessing_and_analysis
#fmt: on


def main():
    parser = argparse.ArgumentParser(
        description="Run specific parts of the program"
    )
    parser.add_argument("--all", action="store_true", help="Run all parts")
    parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Run the preprocessing and analysis",
    )
    parser.add_argument(
        "--modelling", action="store_true", help="Run the modelling"
    )

    args = parser.parse_args()

    # Run preprocessing and modelling
    if args.all:
        run_preprocessing_and_analysis()
        run_modelling()

    # Run preprocessing
    if args.preprocessing:
        run_preprocessing_and_analysis()

    # Run modelling
    if args.modelling:
        run_modelling() if os.listdir(CLEANED_DATA_DIR) else print(
            "Cleaned data does not exist, please run the preprocessing"
        )


if __name__ == "__main__":
    main()
