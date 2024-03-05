from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from utils import read_data, create_new_features, build_X_y, split_data
from tqdm import tqdm


class NaiveThresholdModel:

    def __init__(self, col=None):
        self.col = col if col is None else "distance"
        self.threshold = None

    def fit(self, X, y):
        "Implemented as a dummy"
        self.threshold = 6000

    def predict(self, X):
        return (X.loc[:, self.col] < self.threshold).astype(int).values


def drop_default_columns(X_train, X_test):
    # Drop features that are redundant.
    # Coordinates are converted to distance and any effect from exact location is deemed to not
    # generalize
    # Distance isincluded as log(distance)
    # Near angle is included as sin(angle) and cos(angle)

    X_train_dropped = X_train.drop(
        columns=["distance", "near_x", "near_y", "xcoor", "ycoor", "near_angle"]
    ).copy()
    X_test_dropped = X_test.drop(
        columns=["distance", "near_x", "near_y", "xcoor", "ycoor", "near_angle"]
    ).copy()
    return X_train_dropped, X_test_dropped


def test_models(X_train_full, y_train_full, iterations=5):

    misclassification_rate_train = pd.DataFrame()
    misclassification_rate_test = pd.DataFrame()

    # Iteratively split the train data into a train and validation set
    for i in tqdm(range(iterations), "test_models"):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, train_size=0.8
        )

        clfs = {
            "AdaBoost": AdaBoostClassifier(),
            "Gradient boosting": GradientBoostingClassifier(),
            "Naive threshold": NaiveThresholdModel(col="distance"),
            # "Random forest": RandomForestClassifier(),
            # "Bagging": BaggingClassifier(),
            # "Decision tree": DecisionTreeClassifier(),
        }

        misclassification_rate = pd.DataFrame()
        for clf_name, clf in clfs.items():
            clf.fit(X_train, y_train)
            y_train_pred = clf.predict(X_train)
            misclassification_rate_train.loc[clf_name, i] = np.mean(
                y_train != y_train_pred
            )
            y_val_pred = clf.predict(X_val)
            misclassification_rate_test.loc[clf_name, i] = np.mean(y_val != y_val_pred)
    misclassification_rate = pd.DataFrame()
    misclassification_rate["train"] = misclassification_rate_train.mean(axis=1)
    misclassification_rate["validation"] = misclassification_rate_test.mean(axis=1)
    misclassification_rate.columns.name = "Data set"
    misclassification_rate.index.name = "Model"
    misclassification_rate = misclassification_rate.sort_values("validation")
    print(misclassification_rate)
    fig = px.bar(
        misclassification_rate,
        title="Misclassification rate for different models",
        barmode="group",
        labels={"Model": "", "value": "Misclassification error"},
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()


def test_features(X_train_full, y_train_full, iterations=5):

    misclassification_rate_train = pd.DataFrame()
    misclassification_rate_test = pd.DataFrame()

    # Iteratively split the train data into a train and validation set
    for i in tqdm(range(iterations), "test_features"):

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, train_size=0.8
        )
        X_train, X_val = drop_default_columns(X_train, X_val)
        # Loop and drop various features to find the best ones
        cols_to_drop_list = list(X_train.columns) + [
            ["cos_angle", "sin_angle", "building", "near_fid"]
        ]
        for cols_to_drop in cols_to_drop_list:
            # Create new X data sets with various combinations of features dropped

            X_train_dropped = X_train.drop(columns=cols_to_drop)
            X_val_dropped = X_val.drop(columns=cols_to_drop)

            clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50)
            clf.fit(X_train_dropped, y_train)
            y_train_pred = clf.predict(X_train_dropped)
            cols_to_drop_name = (
                cols_to_drop if isinstance(cols_to_drop, str) else "4 least important"
            )
            misclassification_rate_train.loc[cols_to_drop_name, i] = np.mean(
                y_train != y_train_pred
            )
            y_val_pred = clf.predict(X_val_dropped)
            misclassification_rate_test.loc[cols_to_drop_name, i] = np.mean(
                y_val != y_val_pred
            )
    misclassification_rate = pd.DataFrame()
    misclassification_rate["train"] = misclassification_rate_train.mean(axis=1)
    misclassification_rate["validation"] = misclassification_rate_test.mean(axis=1)
    misclassification_rate.columns.name = "Data set"
    misclassification_rate.index.name = "Dropped columns"
    misclassification_rate = misclassification_rate.sort_values("validation")
    fig = px.bar(
        misclassification_rate,
        title="Misclassification rate for different sets of dropped features",
        barmode="group",
        labels={"Dropped columns": "", "value": "Misclassification error"},
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=0.9, xanchor="left", x=0.05)
    )
    fig.show()
    pass


def test_n_estimators(X_train_full, y_train_full, iterations=5):

    misclassification_rate_train = pd.DataFrame()
    misclassification_rate_test = pd.DataFrame()

    # Iteratively split the train data into a train and validation set
    for i in tqdm(range(iterations), "test_n_estimators"):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, train_size=0.8
        )
        X_train, X_val = drop_default_columns(X_train, X_val)

        N = np.arange(1, 101, 2)
        for n_estimators in N:
            gb_clf = GradientBoostingClassifier(
                learning_rate=0.1, n_estimators=n_estimators
            )
            gb_clf.fit(X_train, y_train)
            y_train_pred = gb_clf.predict(X_train)
            misclassification_rate_train.loc[n_estimators, i] = np.mean(
                y_train != y_train_pred
            )
            y_val_pred = gb_clf.predict(X_val)
            misclassification_rate_test.loc[n_estimators, i] = np.mean(
                y_val != y_val_pred
            )
    misclassification_rate = pd.DataFrame()
    misclassification_rate["train"] = misclassification_rate_train.mean(axis=1)
    misclassification_rate["validation"] = misclassification_rate_test.mean(axis=1)
    fig = px.line(
        misclassification_rate, title="Misclassification rate vs number of estimators"
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()


def test_learning_rate(X_train_full, y_train_full, iterations=5):
    misclassification_rate_train = pd.DataFrame()
    misclassification_rate_test = pd.DataFrame()
    for i in tqdm(range(iterations), "test_learning_rate"):
        # Further split the train data into a train and validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, train_size=0.8
        )
        X_train, X_val = drop_default_columns(X_train, X_val)

        misclassification_rate = pd.DataFrame()
        for learning_rate in np.arange(0.3, 0.0, -0.025):
            gb_clf = GradientBoostingClassifier(
                learning_rate=learning_rate, n_estimators=60
            )
            gb_clf.fit(X_train, y_train)
            y_train_pred = gb_clf.predict(X_train)
            misclassification_rate_train.loc[learning_rate, i] = np.mean(
                y_train != y_train_pred
            )
            y_val_pred = gb_clf.predict(X_val)
            misclassification_rate_test.loc[learning_rate, i] = np.mean(
                y_val != y_val_pred
            )

    misclassification_rate = pd.DataFrame()
    misclassification_rate["train"] = misclassification_rate_train.mean(axis=1)
    misclassification_rate["validation"] = misclassification_rate_test.mean(axis=1)
    fig = px.line(
        misclassification_rate, title="Misclassification rate vs learning_rate"
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()


def test_n_estimators_and_learning_rate(X_train_full, y_train_full, iterations=50):

    learning_rate_list = np.arange(0.05, 0.20, 0.025).round(3)
    n_estimators_list = np.arange(20, 80, 2)
    cols = [learning_rate_list, range(iterations)]
    rows = n_estimators_list
    misclassification_rate = pd.DataFrame(
        columns=pd.MultiIndex.from_product(cols, names=["learning_rate", "i"]),
        index=rows,
    )
    misclassification_rate.index.name = "n_estimators"

    # Iteratively split the train data into a train and validation set
    for i in tqdm(range(iterations), "test_n_estimators_and_learning_rate"):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, train_size=0.8
        )
        X_train, X_val = drop_default_columns(X_train, X_val)

        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                gb_clf = GradientBoostingClassifier(
                    learning_rate=learning_rate, n_estimators=n_estimators
                )
                gb_clf.fit(X_train, y_train)
                y_val_pred = gb_clf.predict(X_val)
                misclassification_rate.loc[n_estimators, (learning_rate, i)] = np.mean(
                    y_val != y_val_pred
                )

    fig = px.line(
        misclassification_rate.T.groupby(level=0).mean().T,
        title="Misclassification rate vs number of estimators and learning rate",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0)
    )
    fig.show()
    pass


def tune_gb(X_train, y_train, X_test, y_test):

    X_train, X_test = drop_default_columns(X_train, X_test)
    X_train = X_train.drop(columns=["cos_angle", "sin_angle", "near_fid", "building"])
    X_test = X_test.drop(columns=["cos_angle", "sin_angle", "near_fid", "building"])

    gb_clf = GradientBoostingClassifier(
        n_iter_no_change=10, learning_rate=0.1, n_estimators=60
    )

    # Perform grid search
    param_grid = {
        "max_depth": [1, 2, 3, 4],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": [2, 3, 4, 5, 6],
        "subsample": [0.5, 0.75, 1],  # 1.0 seems best
        "min_weight_fraction_leaf": [0, 0.01, 0.02],
    }

    clf = GridSearchCV(gb_clf, param_grid, verbose=2, cv=25, n_jobs=11)
    clf.fit(X_train, y_train)
    print()
    print("Best estimator found using:")
    print(clf.best_estimator_)
    print("Mean CV-score for best estimator:")
    print(clf.best_score_)

    # Save model as a pickle file
    with open("gb_best_clf.pickle", "wb") as f:
        pickle.dump(gb_best_clf, f)


def evaluate_model(X_train, y_train, X_test, y_test):

    X_train, X_test = drop_default_columns(X_train, X_test)
    X_train = X_train.drop(columns=["cos_angle", "sin_angle", "near_fid", "building"])
    X_test = X_test.drop(columns=["cos_angle", "sin_angle", "near_fid", "building"])

    gb_best_clf = load_model()
    y_train_pred = gb_best_clf.predict(X_train)
    print(
        f"Best estimator misclassification rate on train data: \
            {np.mean(y_train!=y_train_pred):.3f}"
    )

    y_test_pred = gb_best_clf.predict(X_test)
    print(
        f"Best estimator misclassification rate on test data: \
            {np.mean(y_test!=y_test_pred):.3f}"
    )

    print("Found parameters:")
    for key, value in gb_best_clf.get_params().items():
        print(key, ":", value)
    conf_mat = pd.DataFrame(confusion_matrix(y_test, y_test_pred))
    conf_mat.index.name = "True"
    conf_mat.columns.name = "Pred"
    print("Confusion matrix")
    print(conf_mat)


def load_model(filename=None):
    filename = "gb_best_clf.pickle" if filename is None else filename
    with open(filename, "rb") as f:
        gb_best_clf = pickle.load(f)
    return gb_best_clf


def train_model_on_all_data(X_train, y_train, X_test, y_test):

    X_train, X_test = drop_default_columns(X_train, X_test)
    X_train = X_train.drop(columns=["cos_angle", "sin_angle", "near_fid", "building"])
    X_test = X_test.drop(columns=["cos_angle", "sin_angle", "near_fid", "building"])
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    gb_best_clf = load_model()
    gb_best_clf.fit(X, y)

    y_pred = gb_best_clf.predict(X)
    print(
        f"Best estimator misclassification rate on all data as train data: \
            {np.mean(y!=y_pred):.3f}"
    )

    # Save model as a pickle file
    with open("gb_best_clf_all_data.pickle", "wb") as f:
        pickle.dump(gb_best_clf, f)


def evaluate_test_data():
    X_test = read_data(path="test_without_labels.csv")
    X_test = create_new_features(X_test)
    _, X_test = drop_default_columns(X_test, X_test)
    X_test = X_test.drop(columns=["cos_angle", "sin_angle", "near_fid", "building"])
    clf = load_model("gb_best_clf_all_data.pickle")
    Y_test_pred = pd.DataFrame(clf.predict(X_test)).T
    Y_test_pred.to_csv("predictions.csv", index=False, header=None)


def main():

    # Read data
    df = read_data()

    # Pre-process, note that normalization is not needed for gradient boosted trees
    df = create_new_features(df)

    # Split data and save to csv
    split_data(df)
    X_train, y_train, X_test, y_test = build_X_y()

    # Rough first test
    test_models(X_train, y_train, iterations=50)
    # Ada boost generalizes relatively better but Log loss GB is still slightly better
    # on the val data set
    # Interestingly the two models report quite different importance of the features.
    # Eg while AdaBoost have near_fid as the next most important feature, it is deemed
    # practically useless by the log-loss GB model.

    test_n_estimators(X_train, y_train, iterations=50)
    # ~60 for learning_rate = 0.1

    test_learning_rate(X_train, y_train, iterations=50)
    # ~0.1 for n_estimators = 60

    test_n_estimators_and_learning_rate(X_train, y_train, iterations=25)

    test_features(X_train, y_train, iterations=50)
    # Results not fully clear but 50 iterations suggest excluding
    # ["cos_angle", "sin_angle", "near_fid", "building"].

    # Tune the gb-model using GridSearchCV
    tune_gb(X_train, y_train, X_test, y_test)

    # Evaluate the best found model on the train and test data set
    evaluate_model(X_train, y_train, X_test, y_test)

    # train the best found model, now using the full combined train and test dataset
    train_model_on_all_data(X_train, y_train, X_test, y_test)

    pass


if __name__ == "__main__":
    main()
    evaluate_test_data()
