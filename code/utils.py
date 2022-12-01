import os
import random
import pickle
import shap
import copy
import numpy as np
import pandas as pd

np.seterr(divide="ignore", invalid="ignore")
import matplotlib.pyplot as plt
from zipfile import ZipFile
from scipy.stats import uniform
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import clo_dynamic, v_relative
import zipfile
import re


def open_shapley_files(
    dep_var="thermal", metric="f1_micro", data="-1", algo="rdf", features=None
):
    """
    dep_var = "thermal"
    metric = "f1_micro"
    data = "-1"
    algo = "rdf"
    features = "logged_clo_met"
    """
    path_file = []  # stores the file names
    for root, dirs, files in os.walk("./models"):
        if algo in root:
            for file in files:
                if re.match(f".*shap_values.*{data}.*{dep_var}.*{metric}.*", file):
                    if dep_var == "thermal":
                        if re.match(f".*{features}$", root):
                            path_file.append(os.path.join(root, file))
                    else:
                        path_file.append(os.path.join(root, file))
                    break

    if len(path_file) == 0:
        raise Warning("No file available for this combination of inputs")
    elif len(path_file) > 1:
        raise Warning("There are too many files which contain the data")
    else:
        path_file = path_file[0]

    if "zip" in path_file:
        df = extract_zip_shaply(path_file)
    elif path_file != "":
        df = load_variable(path_file.replace(".pickle", ""))
    else:
        raise Warning("No SHAP values for this combination of inputs")

    return df


def extract_zip_shaply(path_file):
    with ZipFile(path_file, "r") as zip_archive:
        for f in zip_archive.namelist():
            zip_archive.extract(f, path=os.path.dirname(path_file))
            data = load_variable(path_file.replace(".pickle.zip", ""))
            os.remove(path_file.replace(".zip", ""))
            return data


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class DataFrameImputer(TransformerMixin):
    """Impute missing values. Cat columns with most frequent, numeric with median"""

    def fit(self, X, y=None):
        self.fill = pd.Series(
            [
                X[c].value_counts().index[0]
                if X[c].dtype == np.dtype("O")
                else X[c].median()
                for c in X
            ],
            index=X.columns,
        )

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def find_pcm(
    dataframe,
    numeric_features,
    categorical_features,
    metrics,
    data,
    stratified,
    model,
    k_folds,
    test_size,
    use_val,
    fig_name,
    seed,
):
    """Finds the personal model of each user based on CV.
    Assumes a column `userid` exists.

    Args:
        dataframe: A DataFrame with all data and labels as last column
        numeric_features: List of features that are numeric
        categorical_features: List of featurs that are categorical
        metrics: List of performance metrics (string) to be calculated and used
            as scoring functions for CV
        data: Integer of how many datapoints to be used as training data.
            If -1, all data is used
        model: String model to be used to analyse the data
        k_folds: Number (int) of folds for CV
        test_size: Number (int) that indicates the number of test set size
        use_val: Boolean to indicate whether performance metrics are reported
            based on expected values (False) or based on train-test split (True)
        fig_name: String to be used as name for any figure/plot
        seed: Int to be used for random_state in the models later on

    Returns:
        user_pcm: Dictionary with the the metric as the key and another
            dictionary as the value. In this nested dictionary, the key is the
            userid and the values are the actual model
        user_pcm_metric: Dictionary with metrics as the key and another
            dictionary as the value. In this nested dictionary, the key is the
            userid and the values are the actual model performances.
        user_pcm_explainer: Dictionary with metrics as the key and another
            dictionary as the value. In this nested dictionary, the key is the
            user_id and the values are the SHAP explainers
        user_pcm_shap_values: Dictionary with metrics as the key and another
            dictionary as the value. In this nested dictionary, the key is the
            user_id and the values are the SHAP values
    """
    df = dataframe.copy()

    # variables to hold users' models, metrics, and Shap information
    user_pcm, user_pcm_metric, user_pcm_explainer, user_pcm_shap_values = (
        {},
        {},
        {},
        {},
    )
    for metric in metrics:
        (
            user_pcm[metric],
            user_pcm_metric[metric],
            user_pcm_explainer[metric],
            user_pcm_shap_values[metric],
        ) = ({}, {}, {}, {})

    # for every user, do CV
    for user in df["userid"].unique():
        df_user = df.copy()
        df_user = df_user[df_user["userid"] == user]
        df_user = df_user.drop(["userid"], axis=1)
        if "userid" in categorical_features:
            categorical_features.remove("userid")

        print(f"User: {user}")

        # PCM
        (
            model_user,
            model_user_metric,
            _,
            user_explainer,
            user_shap_values,
        ) = train_model(
            dataframe=df_user,
            data=data,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            metrics=metrics,
            stratified=stratified,
            model=model,
            k_folds=k_folds,
            test_size=test_size,
            use_val=use_val,
            trained_model=None,
            fig_name=f"models/{fig_name}_{user}",
            seed=seed,
        )
        for metric in metrics:
            # edge case for incremental data training
            if (
                model_user is None
                and model_user_metric is None
                and user_explainer is None
                and user_shap_values is None
            ):
                user_pcm[metric][user] = None
                user_pcm_metric[metric][user] = None
                user_pcm_explainer[metric][user] = None
                user_pcm_shap_values[metric][user] = None
            else:
                user_pcm[metric][user] = model_user[metric]
                user_pcm_metric[metric][user] = model_user_metric[metric]
                user_pcm_explainer[metric][user] = user_explainer[metric]
                user_pcm_shap_values[metric][user] = user_shap_values[metric]

        # end metric for loop

    # end user for loop

    return user_pcm, user_pcm_metric, user_pcm_explainer, user_pcm_shap_values


def choose_tree_depth(clf, x, y, kf, fig_name="", scorer="f1_micro", save_fig=False):
    """Choose the optimal depth of a tree model"""

    depths = list(range(8, 11))
    cv_scores = []

    for d in depths:
        clf_depth = clf.set_params(max_depth=d)

        if scorer == "cohen_kappa":
            scorer = make_scorer(cohen_kappa_score)

        scores = cross_val_score(clf_depth, x, y, cv=kf, scoring=scorer)
        cv_scores.append(scores.mean())

    # changing to misclassification error and determining best depth
    error = [1 - x for x in cv_scores]
    optimal_depth = depths[error.index(min(error))]

    if save_fig:
        plt.figure(figsize=(12, 10))
        plt.plot(depths, error)
        plt.xlabel("Tree Depth", fontsize=20)
        plt.ylabel("Misclassification Error", fontsize=20)
        plt.savefig(f"{fig_name}_depth.png")
        plt.close()

    return optimal_depth, max(cv_scores)


def cv_model_param(x, y, model, parameters, kf, seed, scorer="f1_micro"):
    """Choose the best combination of parameters for a given model"""

    if scorer == "cohen_kappa":
        scorer = make_scorer(cohen_kappa_score)

    random_search = RandomizedSearchCV(
        model,
        parameters,
        n_iter=1,
        cv=kf,
        scoring=scorer,
        random_state=seed,
    )
    random_search.fit(x, y)

    return random_search.best_estimator_, random_search.best_score_


def train_model(
    dataframe,
    data,
    numeric_features,
    categorical_features,
    metrics,
    stratified=False,
    model="rdf",
    k_folds=5,
    test_size=100,
    use_val=False,
    trained_model=None,
    fig_name="",
    seed=42,
):
    """
    Finds best set of param with K-fold CV and returns trained model and accuracy
    Assumes the label is the last column.

    Args:
        dataframe: A DataFrame with all data and labels as last column
        data: Integer of how many datapoints to be used as training data
            used for training. If -1, all data is used
        numeric_features: List of features that are numeric
        categorical_features: List of features that are categorical
        metrics: List of performance metrics (string) to be calculated and used
            as scoring functions for CV
        stratified: Boolean to whether or not used stratified CV
        model: String model to be used to analyse the data
        k_folds: Number (int) of folds for CV
        test_size: Number (int) that indicates the number of test set size
        use_val: Boolean to indicate whether performance metrics are reported
            based on expected values (False) or based on train-test split (True)
        trained_model: Object of already trained model to fine-tune. None otherwise
        fig_name: String for figure name. Figures depends on the model

    Returns:
        clf_cv: Dictionary with the trained model (value) for each metric (key)
        model_scores: Dictionary with the model score (value) for each metric (key)
        class_report: Dictionary with the classification report (value) for
            each metric (key)
        explainer: Shap object for future plotting and evaluation
        shap_values: List of arrays with the shap_values for each label value
    """
    df = dataframe.copy()

    # imputing missing values
    df = DataFrameImputer().fit_transform(df)

    # preprocess data according to the model
    if model == "mlp":
        # numeric features scaling
        dict_le = {}
        for feature in numeric_features:
            dict_le[feature] = StandardScaler().fit(df[feature].values.reshape(-1, 1))
            df[feature] = dict_le[feature].transform(df[feature].values.reshape(-1, 1))

        # one hot encoding for categorical
        for feature in categorical_features:
            dict_le[feature] = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(
                df[feature].values.reshape(-1, 1)
            )
            df[feature] = dict_le[feature].transform(df[feature].values.reshape(-1, 1))

    elif model in ("lr", "svm"):
        # numeric features scaling
        dict_le = {}
        for feature in numeric_features:
            dict_le[feature] = StandardScaler().fit(df[feature].values.reshape(-1, 1))
            df[feature] = dict_le[feature].transform(df[feature].values.reshape(-1, 1))

        for feature in categorical_features:
            dict_le[feature] = LabelEncoder().fit(df[feature])
            df[feature] = dict_le[feature].transform(df[feature])

    else:
        # categorical features encoding
        dict_le = {}
        for feature in categorical_features:
            dict_le[feature] = LabelEncoder().fit(df[feature])
            df[feature] = dict_le[feature].transform(df[feature])

    # initialize the model
    if model == "rdf":
        parameters = {
            "n_estimators": [
                500
            ],  # [int(x) for x in np.linspace(start=100, stop=600, num=100)],
            "min_samples_split": [2],  # [2, 5, 10],
            "min_samples_leaf": [2],  # [1, 2, 4],
        }
        clf = RandomForestClassifier(
            class_weight="balanced", criterion="gini", random_state=seed
        )

    elif model == "xgb":
        # xgb over gbm: https://datascience.stackexchange.com/questions/16904/gbm-vs-xgboost-key-differences
        parameters = {
            "n_estimators": [int(x) for x in np.linspace(start=100, stop=600, num=20)],
            "max_depth": [2, 4, 6, 8, 10],
        }
        clf = XGBClassifier(random_state=seed)
        # for incremental learning see:
        # https://stackoverflow.com/questions/38079853/how-can-i-implement-incremental-training-for-xgboost

    elif model == "lr":
        parameters = {
            "C": uniform(loc=0, scale=10),
            "fit_intercept": [True, False],
        }
        clf = LogisticRegression(random_state=seed)

    elif model == "svm":
        parameters = {
            "C": uniform(loc=0, scale=10),
            "kernel": ["rbf", "sigmoid"],
        }
        clf = SVC(random_state=seed)

    elif model == "kn":
        parameters = {
            "n_neighbors": [1, 2, 3, 5, 7, 8, 9, 10],
            "weights": ["uniform", "distance"],
        }
        clf = KNeighborsClassifier(p=2, n_neighbors=1)

    elif model == "gnb":
        parameters = {
            "var_smoothing": [1e-9],
        }
        clf = GaussianNB()

    elif model == "mlp":
        # Architecture based on:
        # Luo, M., Xie, J., Yan, Y., Ke, Z., Yu, P., Wang, Z., & Zhang, J. (2020). Comparing machine learning algorithms in predicting thermal sensation using ASHRAE Comfort Database II. Energy and Buildings, 210, 109776. https://doi.org/10.1016/j.enbuild.2020.109776
        parameters = {
            "hidden_layer_sizes": [(13, 13)],  # 2 hidden layers of 13 neurons,
        }
        clf = MLPClassifier(activation="relu", random_state=seed)

    # create folds objects
    if stratified:
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    else:
        kf = KFold(n_splits=k_folds, shuffle=True)

    # model hyperparameter tuning
    clf_cv, cv_score = {}, {}
    idx_timestamp = df.columns.get_loc("timestamp")

    if use_val:
        X = np.array(df.iloc[:, 0 : df.shape[1] - 1])  # minus 1 for label
        y = np.array(df.iloc[:, -1]).astype(int)  # sanity check in case is a float

        X_cv, X_test, y_cv, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        # remove helper column "timestamp"
        X_test = np.delete(X_test, idx_timestamp, axis=1)
        X_test = np.array(
            X_test, dtype=float
        )  # sanity check due to timestamp being part of np.array

        # use only "data" datapoints for training, -1 means no subsets
        if data != -1 and X_cv.shape[0] - data > 0:
            # aux col to sort label
            y_cv = np.stack((y_cv, X_cv[:, idx_timestamp]), axis=1)

            X_cv = X_cv[X_cv[:, idx_timestamp].argsort()]  # sort based on timestamp
            y_cv = y_cv[y_cv[:, -1].argsort()]  # sort based on timestamp

            y_cv = y_cv[:data, :]
            X_cv = X_cv[:data, :]

            y_cv = np.delete(y_cv, -1, axis=1)  # get rid of aux col
            y_cv = np.array(
                y_cv, dtype=float
            ).flatten()  # sanity check due to timestamp being part of np.array

        elif data != -1 and X_cv.shape[0] - data < 0:
            print(f"Current user have {X_cv.shape[0]} datapoints  (< {data})")
            return None, None, None, None, None

        # remove helper column "timestamp"
        X_cv = np.delete(X_cv, idx_timestamp, axis=1)
        X_cv = np.array(
            X_cv, dtype=float
        )  # sanity check due to timestamp being part of np.array

        # find params with all metrics as scorers
        if trained_model is None:
            # no previously trained model, start from scratch
            for metric in metrics:
                # for PCM, if `y_cv` only has 1 label, some models won't work
                one_class = False
                if len(np.unique(y_cv)) == 1:
                    one_class = True
                    clf_cv[metric] = None
                    cv_score[metric] = None
                else:
                    clf_cv[metric], cv_score[metric] = cv_model_param(
                        X_cv, y_cv, clf, parameters, kf, seed, metric
                    )
        else:
            # use the trained model instead
            clf_cv = copy.deepcopy(trained_model)

    else:  # find params without test set (use expectation values)
        # use only "data" datapoints for training, -1 means no subsets
        df = df.sort("timestamp")
        df = df.drop(["timestamp"], axis=1)
        if data != -1 and df.shape[0] - data > 0:
            df = df.head(data)
        elif data != -1 and df.shape[0] - data < 0:
            print(f"Current user doesn't have more than {data} datapoints")
            return None, None, None, None, None

        X = np.array(df.iloc[:, 0 : df.shape[1] - 1])  # minus 1 for label
        y = np.array(df.iloc[:, -1]).astype(int)  # sanity check in case is a float

        if trained_model is None:
            # no previously trained model, start from scratch
            for metric in metrics:
                clf_cv[metric], cv_score[metric] = cv_model_param(
                    X, y, clf, parameters, kf, seed, metric
                )
        else:
            # use the trained model instead
            clf_cv = copy.deepcopy(trained_model)

    # find depth for rdf and update model
    if model == "rdf" and trained_model is None and not one_class:
        for metric in metrics:
            optimal_depth, cv_score[metric] = (
                choose_tree_depth(clf_cv[metric], X_cv, y_cv, kf, fig_name, metric)
                if use_val
                else choose_tree_depth(clf_cv[metric], X, y, kf, fig_name, metric)
            )
            clf_cv[metric] = clf_cv[metric].set_params(max_depth=optimal_depth)

    # model prediction
    model_scores, class_report, explainer, shap_values = {}, {}, {}, {}
    if use_val:
        for metric in metrics:
            if one_class:
                # `y_cv` only has 1 label, model could only predict that label
                y_pred = np.full(X_test.shape[0], np.unique(y_cv))
            else:
                if trained_model is None:
                    # fit model with tuned hyperparam to the entire CV split
                    clf_cv[metric].fit(X_cv, y_cv)
                else:
                    # incremental
                    if model == "xgb":
                        # Based on https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
                        clf_cv[metric].fit(X_cv, y_cv, clf_cv[metric])
                    else:
                        clf_cv[metric].partial_fit(X_cv, y_cv)

                y_pred = clf_cv[metric].predict(X_test)

            model_scores[metric], class_report[metric] = clf_metrics(
                y_test, y_pred, metric
            )

            # SHAP value calculation
            # can't properly fit a model with only one class, skip shap calc
            if not one_class:
                # https://www.kaggle.com/dansbecker/shap-values (example)
                # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/explainers/Permutation.html (docs)
                # shap values explanation:
                # https://christophm.github.io/interpretable-ml-book/shap.html
                # use Kernel SHAP to explain test set predictions
                feature_names = list(df.columns)
                feature_names.remove("timestamp")
                feature_names = feature_names[0:-1]

                X_test_df = pd.DataFrame(X_test, columns=feature_names)
                explainer[metric] = shap.explainers.Permutation(
                    clf_cv[metric].predict, X_test_df
                )

                shap_values[metric] = explainer[metric](X_test_df)

            else:
                explainer[metric] = None
                shap_values[metric] = None

    else:  # no test set means the average CV score will be the model score
        if trained_model is not None:
            print(
                f"use_val = {use_val} and trained_model = {trained_model} are not compatible"
            )
            return None, None, None, None, None

        for metric in metrics:
            # cohen_kappa scoring is slightly different
            if metric == "cohen_kappa":
                metric = make_scorer(cohen_kappa_score)

            model_scores[metric] = cross_val_score(
                clf_cv[metric], X, y, scoring=metric, cv=k_folds
            )  # average CV score
            class_report[metric] = ""

            # no test set, shap values can't be calculated
            explainer[metric] = None
            shap_values[metric] = -1

    return clf_cv, model_scores, class_report, explainer, shap_values


def clf_metrics(test_labels, pred_labels, metric):
    """Compute the confusion matrix and score a given metric."""

    if metric == "accuracy":
        score = accuracy_score(test_labels, pred_labels)
    elif metric == "balanced_accuracy":
        score = balanced_accuracy_score(test_labels, pred_labels)
    elif metric == "f1_micro":
        score = f1_score(test_labels, pred_labels, average="micro", zero_division=0)
    elif metric == "f1_macro":
        score = f1_score(test_labels, pred_labels, average="macro", zero_division=0)
    elif metric == "f1_weighted":
        score = f1_score(test_labels, pred_labels, average="weighted", zero_division=0)
    elif metric == "cohen_kappa":  # [-1, 1]
        score = cohen_kappa_score(test_labels, pred_labels)

    # classification report
    class_report = classification_report(
        test_labels, pred_labels, output_dict=True, zero_division=0
    )

    return score, class_report


def save_variable(file_name, variable):
    with open(f"{file_name}.pickle", "wb") as f:
        pickle.dump(variable, f)


def load_variable(file_name):
    with open(f"{file_name}.pickle", "rb") as f:
        return pickle.load(f)


def simplified_pmv_model(data):
    data = data[["rh-env", "t-env", "clothing", "met", "thermal"]].copy()
    data["met"] = data["met"].map(
        {
            "Sitting": 1.1,
            "Resting": 0.8,
            "Standing": 1.4,
            "Exercising": 3,
        }
    )
    data["clothing"] = data["clothing"].map(
        {
            "Very light": 0.3,
            "Light": 0.5,
            "Medium": 0.7,
            "Heavy": 1,
        }
    )

    arr_pmv_grouped = []
    arr_pmv = []
    for _, row in data.iterrows():
        val = pmv(
            row["t-env"],
            row["t-env"],
            v_relative(0.1, row["met"]),
            row["rh-env"],
            row["met"],
            clo_dynamic(row["clothing"], row["met"]),
        )
        if val < -1.5:
            arr_pmv_grouped.append("Warmer")
        elif -1.5 <= val <= 1.5:
            arr_pmv_grouped.append("No Change")
        else:
            arr_pmv_grouped.append("Cooler")

        arr_pmv.append(val)

    data["PMV"] = arr_pmv
    data["PMV_grouped"] = arr_pmv_grouped

    return data["PMV_grouped"]
