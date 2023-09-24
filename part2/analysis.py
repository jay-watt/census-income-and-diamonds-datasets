import matplotlib.pyplot as plt
from common.analysis import summarise
from common.config import PLOTS_DIR
from common.utils import create_subplot_layout, plot_boxplot


def plot_class_distribution(class_name, class_data):
    axes = create_subplot_layout(2)

    # skew, kurt = plot_histogram(class_name, class_data, axes[0])
    plot_boxplot(class_name, class_data, axes[1])

    plt.savefig(f'{PLOTS_DIR}/part1_class_distribution.png')


def analyse(data):
    print('Initially analysing data...\n')
    class_name = summarise(data)
    plot_class_distribution(class_name, data[class_name])
    print('Initial analysis complete!\n')
