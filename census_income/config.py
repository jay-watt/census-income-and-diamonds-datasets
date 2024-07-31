DATA_DIR = '/home/j/machine-learning/a4/Part2/data/original'
DATA_FILENAMES = {'training': 'adult.data', 'test': 'adult.test'}
PLOTS_FILENAME = 'part2'

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

MAX_CATEGORIES = 10

MAPPINGS = {
    'education': {
        "Preschool": 0,
        "1st-4th": 1,
        "5th-6th": 2,
        "7th-8th": 3,
        "9th": 4,
        "10th": 5,
        "11th": 6,
        "12th": 7,
        "HS-grad": 8,
        "Some-college": 9,
        "Assoc-voc": 10,
        "Assoc-acdm": 11,
        "Bachelors": 12,
        "Masters": 13,
        "Doctorate": 14,
        "Prof-school": 15
    }
}

VAR_THRESHOLD = 0.01
RFE_SAMPLE_FRACTION = 0.3
