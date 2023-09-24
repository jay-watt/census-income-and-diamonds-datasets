import pandas as pd
from common.analysis import summarise
from common.utils import display_table


def display_class_distribution(class_name, class_data):
    class_counts = class_data.value_counts()
    class_percentage = class_data.value_counts(normalize=True) * 100
    distribution = pd.DataFrame(
        {'count': class_counts, 'percentage': class_percentage}
    )
    distribution['percentage'] = distribution['percentage'].apply(
        lambda x: f'{x:.2f}'
    )
    distribution['count'] = distribution['count'].apply(lambda x: f'{x:,.0f}')
    distribution.index.name = class_name.capitalize()
    display_table('Class Distribution', distribution)


def analyse(data):
    print('Initially analysing data...\n')
    class_name = summarise(data)
    display_class_distribution(class_name, data[class_name])
    print('Initial analysis complete!\n')
    return class_name
