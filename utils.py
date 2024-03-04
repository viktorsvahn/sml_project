"""Read data, create features and write csv-files where the train and
test data is separated.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(path="siren_data_train.csv"):
    df = pd.read_csv(path)
    return df


def create_new_features(df):
    # Calculate distance to closest siren
    df.loc[:, "distance"] = (
        (df.xcoor - df.near_x) ** 2 + (df.ycoor - df.near_y) ** 2
    ) ** 0.5

    df.loc[:, "distance_log"] = np.log(df.distance)

    # Make angle circular
    df.loc[:, "sin_angle"] = np.sin(2.0 * np.pi * df.near_angle)
    df.loc[:, "cos_angle"] = np.cos(2.0 * np.pi * df.near_angle)
    return df


def split_data(df):
    train, test = train_test_split(df, train_size=0.8, random_state=1)
    train.to_csv("siren_data_train_TRAIN.csv")
    test.to_csv("siren_data_train_TEST.csv")


def build_X_y():

    train = pd.read_csv("siren_data_train_TRAIN.csv", index_col=0)
    test = pd.read_csv("siren_data_train_TEST.csv", index_col=0)

    # Divide data sets
    y_col = "heard"

    X_train = train.drop(columns=[y_col])
    y_train = train.loc[:, y_col]
    X_test = test.drop(columns=[y_col])
    y_test = test.loc[:, y_col]

    return X_train, y_train, X_test, y_test
