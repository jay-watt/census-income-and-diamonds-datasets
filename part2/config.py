DATA_DIR = '/home/j/machine-learning/a4/Part2/data/original'
DATA_FILENAMES = {'train': 'adult.data', 'test': 'adult.test'}
FEATURES = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
]
CLASS = 'income'
CLEANED_DATA = {'train': 'cleaned/train.csv', 'test': 'cleaned/test.csv'}
SEED = 309

# Plots
UNITS = {}
PLOTS_DIR = '../Reports/plots'

# Distribution
SKEW_THRESHOLD = 1
KURT_THRESHOLD = 5

# Preprocessing
CORRELATION_THRESHOLD = 0.5

# Modelling
MODELS = []
