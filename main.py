import os
import sys

from part1.run import run_part1
from part2.run import run_part2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    run_part1()
    run_part2()


if __name__ == "__main__":
    main()
