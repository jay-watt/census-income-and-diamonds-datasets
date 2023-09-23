DATA_PATH = '/home/j/machine-learning/a4/Part1/data/original/diamonds.csv'
SEED = 309

# Plots
UNITS = {
    'price': 'USD',
    'carat': 'ct',
    'depth': '% width',
    'table': '% width',
    'x': 'mm',
    'y': 'mm',
    'z': 'mm',
    'volume': 'mm3',
}
PLOTS_PATH = '../Reports/plots'

# Distribution
SKEW_THRESHOLD = 1
KURT_THRESHOLD = 5

ORDINAL_MAPPINGS = {
    'cut': {
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Premium': 4,
        'Ideal': 5,
    },
    'color': {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7},
    'clarity': {
        'I1': 1,
        'SI2': 2,
        'SI1': 3,
        'VS2': 4,
        'VS1': 5,
        'VVS2': 6,
        'VVS1': 7,
        'IF': 8,
    },
}
