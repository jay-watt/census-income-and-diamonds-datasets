import argparse
import os
import sys
import warnings

part1_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(part1_dir)
sys.path.append(parent_dir)

from Part1_2_Common.config import CLEANED_DATA_DIR
from Part1.analysis import run_analysis
from Part1.modelling import run_modelling
from Part1.preprocessing import run_preprocessing

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(
        description='Run specific parts of the program'
    )
    parser.add_argument('--all', action='store_true', help='Run all parts')
    parser.add_argument(
        '--analysis', action='store_true', help='Run the analysis'
    )
    parser.add_argument(
        '--preprocessing',
        action='store_true',
        help='Run the preprocessing',
    )
    parser.add_argument(
        '--modelling', action='store_true', help='Run the modelling'
    )

    args = parser.parse_args()

    if args.all:
        run_analysis()
        run_preprocessing()
        run_modelling()
    if args.analysis:
        run_analysis()
    if args.preprocessing:
        run_preprocessing()
    if args.modelling:
        if os.listdir(CLEANED_DATA_DIR):
            run_modelling()
        else:
            print('Cleaned data does not exist, please run the preprocessing')


if __name__ == "__main__":
    main()
