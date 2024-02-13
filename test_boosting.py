from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from utils import read_data, create_new_features, build_X_y, split_data


class NaiveModel:

    def __init__(self, col=None):
        self,
        self.col = col if col is None else "distance"
        self.threshold = None

    def fit(self, X, y):
        "Minimize loss by finding the optimal threshold for self.col"

        # Find optimal threshold as first bin where the fraction is below 0.5
        bins = pd.cut(X.loc[:, self.col], bins=np.arange(0, 50000, 1000))
        sum_per_bin = y.groupby(bins, observed=False).sum()
        count_per_bin = y.groupby(bins, observed=False).count()
        fraction_per_bin = sum_per_bin / count_per_bin
        self.threshold = (fraction_per_bin < 0.5).idxmax().left

    def predict(self, X):
        return (X.loc[:, self.col] < self.threshold).astype(int).values


def test_raw_models():

    # Loop and drop various features to find the best ones
    misclassification_rate = pd.DataFrame()
    cols_to_drop_list = [
        [],
        ["building"],
        ["near_fid"],
        ["asleep"],
        ["building", "near_fid"],
        ["building", "asleep"],
        ["near_fid", "asleep"],
        ["building", "near_fid", "asleep"],
        ["cos_angle", "sin_angle"],
        ["cos_angle", "sin_angle", "building"],
        ["cos_angle", "sin_angle", "near_fid"],
        ["cos_angle", "sin_angle", "asleep"],
        ["cos_angle", "sin_angle", "building", "near_fid"],
        ["cos_angle", "sin_angle", "building", "asleep"],
        ["cos_angle", "sin_angle", "near_fid", "asleep"],
        ["cos_angle", "sin_angle", "building", "near_fid", "asleep"],
    ]
    for cols_to_drop in cols_to_drop_list:
        X_train, y_train, X_test, y_test = build_X_y()
        X_train = X_train.drop(
            columns=["distance", "near_x", "near_y", "xcoor", "ycoor", "near_angle"]
        )
        X_test = X_test.drop(
            columns=["distance", "near_x", "near_y", "xcoor", "ycoor", "near_angle"]
        )
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        clfs = {
            "AdaBoost": AdaBoostClassifier(),
            "Log-loss GB": GradientBoostingClassifier(),
            # "Naive (dist threshold=6000m)": NaiveModel(col="distance", threshold=6000),
        }

        for clf_name, clf in clfs.items():
            clf.fit(X_train, y_train)
            y_train_pred = clf.predict(X_train)
            misclassification_rate.loc[",".join(cols_to_drop), clf_name] = np.mean(
                y_train != y_train_pred
            )
    misclassification_rate.columns.name = "Model"
    misclassification_rate.index.name = "Dropped columns"
    misclassification_rate.loc[:, "Mean"] = misclassification_rate.mean(axis=1)
    print(misclassification_rate.sort_values("Mean"))


def tune_gb():
    X_train, y_train, X_test, y_test = build_X_y()
    X_train = X_train.drop(
        columns=["distance", "near_x", "near_y", "xcoor", "ycoor", "near_angle"]
    )
    X_test = X_test.drop(
        columns=["distance", "near_x", "near_y", "xcoor", "ycoor", "near_angle"]
    )

    gb_clf = GradientBoostingClassifier(
        n_iter_no_change=10, learning_rate=0.1, n_estimators=100
    )

    # Perform grid search
    param_grid = {
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_leaf": [1, 3, 5, 7, 9],
        "max_features": [6, 7, 8, 9, 10],
        # "subsample": [0.6, 0.7, 0.8, 0.9, 1],  # 1.0 seems best
        "min_weight_fraction_leaf": [0, 0.05, 0.1, 0.15, 0.2],
    }

    clf = GridSearchCV(gb_clf, param_grid, verbose=2)
    clf.fit(X_train, y_train)
    print()
    print("Best estimator found using:")
    print(clf.best_estimator_)
    print("Mean CV-score for best estimator:")
    print(clf.best_score_)

    # Train model with found paramters using lower learning rate and higher n_estimators
    gb_best_clf = clf.best_estimator_
    y_train_pred = gb_best_clf.predict(X_train)
    print(
        f"Best estimator misclassification rate on train data: {np.mean(y_train!=y_train_pred):.3f}"
    )

    y_test_pred = gb_best_clf.predict(X_test)
    print(
        f"Best estimator misclassification rate on test data: {np.mean(y_test!=y_test_pred):.3f}"
    )
    conf_mat = pd.DataFrame(confusion_matrix(y_test, y_test_pred))
    conf_mat.index.name = "True"
    conf_mat.columns.name = "Pred"
    print("Confusion matrix")
    print(conf_mat)

    with open("gb_best_clf.pickle", "wb") as f:
        pickle.dump(gb_best_clf, f)

    return gb_best_clf


def load_model(filename=None):
    filename = "gb_best_clf.pickle" if filename is None else filename
    with open(filename, "rb") as f:
        gb_best_clf = pickle.load(f)
    return gb_best_clf


def main():

    # Read data
    # df = read_data()

    # Pre-process, note that normalization is not needed for gradient boosted trees
    # df = create_new_features(df)

    # Split data and save to csv
    # split_data(df)

    # Rough first test
    test_raw_models()
    # Interestingly the two models report quite different importance of the features. Eg while AdaBoost have near_fid as the next most important feature, it is deemed practically useless by the log-loss GB model.

    # Tune
    tune_gb()
    # gb_best_clf = load_model()
    pass


if __name__ == "__main__":
    main()

# TODO:s
# Write on overleaf
# Make some plots of the training?
# Check if n_estimators is reasonable :-)
