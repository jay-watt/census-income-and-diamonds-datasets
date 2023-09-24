DATA_FILENAME = 'diamonds'
PLOTS_FILENAME = 'part1'

MAPPINGS = {
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

SVR_MAX_ITER = 50000
MLP_MAX_ITER = 5000
