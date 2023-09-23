from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

DATA_DIR = '/home/j/machine-learning/a4/Part1/data/'
ORIGINAL_DATA = 'original/diamonds.csv'
CLEANED_DATA = {'train': 'cleaned/train.csv', 'test': 'cleaned/train.csv'}
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
PLOTS_DIR = '../Reports/plots'

# Distribution
SKEW_THRESHOLD = 1
KURT_THRESHOLD = 5

# Preprocessing
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
CORRELATION_THRESHOLD = 0.5

# Modelling
MODELS = [
    ('Linear Regression', LinearRegression()),
    ('K-Neighbors Regression', KNeighborsRegressor()),
    ('Ridge Regression', Ridge(random_state=SEED)),
    ('Decision Tree Regression', DecisionTreeRegressor(random_state=SEED)),
    ('Random Forest Regression', RandomForestRegressor(random_state=SEED)),
    (
        'Gradient Boosting Regression',
        GradientBoostingRegressor(random_state=SEED),
    ),
    ('SGD Regression', SGDRegressor(random_state=SEED)),
    ('Support Vector Regression', SVR()),
    ('Linear SVR', LinearSVR(random_state=SEED)),
    ('Multi-layer Perceptron Regression', MLPRegressor(random_state=SEED)),
]
