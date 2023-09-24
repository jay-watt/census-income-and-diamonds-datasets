DATA_DIR = '/home/j/machine-learning/a4/Part2/data/original'
DATA_FILENAMES = {'train': 'adult.data', 'test': 'adult.test'}
COLUMN_NAMES = [
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
    'income',
]

MAPPINGS = {
    'education': {
        'Preschool': 0,
        '1st-4th': 1,
        '5th-6th': 2,
        '7th-8th': 3,
        '9th': 4,
        '10th': 5,
        '11th': 6,
        '12th': 7,
        'HS-grad': 8,
        'Some-college': 9,
        'Assoc-voc': 10,
        'Assoc-acdm': 11,
        'Bachelors': 12,
        'Masters': 13,
        'Doctorate': 14,
        'Prof-school': 15,
    },
    'marital-status': {
        'Never-married': 0,
        'Married-civ-spouse': 1,
        'Married-spouse-absent': 2,
        'Separated': 3,
        'Divorced': 4,
        'Widowed': 5,
        'Married-AF-spouse': 6,
    },
    'relationship': {
        'Not-in-family': 0,
        'Unmarried': 1,
        'Own-child': 2,
        'Other-relative': 3,
        'Husband': 4,
        'Wife': 5,
    },
    'sex': {'Male': 0, 'Female': 1},
}

CHI2_THRESHOLD = 100
