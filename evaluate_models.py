from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import plotly.express as px
import pickle


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


def build_X_y(df):

    # Divide data sets
    x_cols = [
        "near_fid",
        "building",
        "noise",
        "in_vehicle",
        "asleep",
        "no_windows",
        "age",
        "distance",
        # "distance_log",
        # "sin_angle",
        # "cos_angle",
    ]
    y_col = "heard"
    train, test = train_test_split(df, train_size=0.8, random_state=1)
    X_train = train.loc[:, x_cols]
    y_train = train.loc[:, y_col]
    X_test = test.loc[:, x_cols]
    y_test = test.loc[:, y_col]

    return X_train, y_train, X_test, y_test


def test_raw_models(df):
    X_train, y_train, X_test, y_test = build_X_y(df)

    clfs = {
        "AdaBoost": AdaBoostClassifier(),
        "Log-loss GB": GradientBoostingClassifier(),
        # "Naive (dist threshold=6000m)": NaiveModel(col="distance", threshold=6000),
    }

    for clf_name, clf in clfs.items():
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        print(
            f"{clf_name} : Misclassification error on training data = {np.mean(y_train != y_train_pred):.3f}"
        )
        try:
            feat_imp = pd.Series(clf.feature_importances_, list(X_train)).sort_values(
                ascending=False
            )
            px.bar(feat_imp, title=f"Importance of Features {clf_name}").show()
            # Interesting to see that different features have a very diffferent importance in AdaBoost vs log-loss GB, see eg near_fid.
        except AttributeError:
            pass  # Feature importance not implemented on the NaiveModel


def tune_gb(df):
    X_train, y_train, X_test, y_test = build_X_y(df)

    gb_clf = GradientBoostingClassifier(
        n_iter_no_change=10, learning_rate=0.1, n_estimators=100
    )

    # Perform grid search
    param_grid = {
        "max_depth": [1, 2, 3, 4, 5],
        "min_samples_leaf": [1, 3, 5, 7, 9],
        "max_features": [2, 3, 4, 5, 6, 7, 8],
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

    # The below did not improve the score on the train data, but made it worse!
    # gb_best_clf.set_params(learning_rate=0.01, n_estimators=500, n_iter_no_change=None)
    # gb_best_clf.fit(X_train, y_train)
    # y_train_pred = gb_best_clf.predict(X_train)
    # print(
    #     f"Best estimator misclassification rate on train data with n_estimators=500 and learning_rate=0.01: {np.mean(y_train!=y_train_pred):.3f}"
    # )

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
    df = read_data()

    # Pre-process, note that normalization is not needed for gradient boosted trees
    df = create_new_features(df)

    test_raw_models(df)
    # Interestingly the two models report quite different importance of the features. Eg while AdaBoost have near_fid as the next most important feature, it is deemed practically useless by the log-loss GB model.

    gb_best_clf = tune_gb(df)
    # gb_best_clf = load_model()
    pass


if __name__ == "__main__":
    main()

# TODO:s
# Write on overleaf
# Make some plots of the training?
# Check if n_estimators is reasonable :-)
