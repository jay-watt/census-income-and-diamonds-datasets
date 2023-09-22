import pandas as pd
from sklearn.discriminant_analysis import StandardScaler


# Preprocessing
def encode(data):
    ordinal_mappings = {
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
    for feature in ordinal_mappings:
        data[feature] = (
            data[feature].map(ordinal_mappings[feature]).astype(int)
        )
    return data


def remove_outliers(class_name, data):
    initial_instances = data.shape[0]
    for feature in data.drop(columns=[class_name]):
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Defining bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter data
        data = data[
            (data[feature] >= lower_bound) & (data[feature] <= upper_bound)
        ]
    instances_removed = initial_instances - data.shape[0]
    print(
        f"Number of rows removed due to outliers:\t{instances_removed:,.0f}\n"
    )
    return data


def scale(class_name, data):
    class_data = data[class_name].reset_index(drop=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.drop(columns=[class_name]))
    column_names = data.columns[data.columns != class_name]
    scaled_data = pd.DataFrame(scaled, columns=column_names).reset_index(
        drop=True
    )
    return pd.concat([scaled_data, class_data], axis=1)


# Writing
def write_cleaned_data(data, data_name):
    data.reset_index(drop=True, inplace=True)
    data.index = data.index + 1
    data.to_csv(f'data/cleaned/{data_name}.csv', index=True)


# Preprocessing and EDA main function
def preprocess(class_name, data, data_name):
    encoded_data = encode(data)
    inlier_data = remove_outliers(class_name, encoded_data)
    scaled_data = scale(class_name, inlier_data)
    write_cleaned_data(scaled_data, data_name)
    return scaled_data
