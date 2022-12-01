import os
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns


class Configuration:
    def __init__(self):

        self.map_cozie = {
            "comfort": {10: "Yes", 9: "No"},
            "indoorOutdoor": {9: "Outdoor", 11: "Indoor"},
            "change": {11: "Yes", 10: "No"},
            "location": {8: "Portable", 9: "Work", 10: "Other", 11: "Home"},
            "thermal": {9: "Warmer", 10: "No Change", 11: "Cooler"},
            "clothing": {8: "Very light", 9: "Light", 10: "Medium", 11: "Heavy"},
            "met": {8: "Resting", 9: "Sitting", 10: "Standing", 11: "Exercising"},
            "air-vel": {10: "Yes", 11: "No"},
        }

        self.inv_map_cozie = {}
        for key in self.map_cozie:
            self.inv_map_cozie[key] = {v: k for k, v in self.map_cozie[key].items()}

        self.time_zone = "Asia/Singapore"

        self.data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
        self.tables_dir = os.path.join(
            os.path.dirname(os.getcwd()), "manuscript/src/tables"
        )
        self.fig_dir = os.path.join(
            os.path.dirname(os.getcwd()), "manuscript/src/figures"
        )
        self.var_dir = os.path.join(
            os.path.dirname(os.getcwd()), "manuscript/src/variables"
        )
        self.models_results_dir = os.path.join(
            os.path.dirname(os.getcwd()), "code/models"
        )
        self.tc_colors = [sns.color_palette("Paired")[i] for i in [0, 2, 4]]
        self.categorical_colors = [(0.8, 0.8, 0.8)] + [
            sns.color_palette("Paired")[i] for i in [6, 8, 10]
        ]

        # variable types
        num, cat, ind = "numeric", "categorical", "independent"

        # listed in order of complexity to measure, False means we are not going to use it
        self.features = {
            "independent": {"clothing": ind, "thermal": ind, "met": ind},
            "environmental": {
                "t-out": num,
                "hr-out": num,
                "rh-out": False,
                "t-env": num,
                "hr-env": num,
                "air-vel": False,  # not available in real scenario
                "rh-env": False,  # I am using humidity ratio instead
                "Light": False,  # lots of missing data
                "CO2": False,  # lots of missing data
                "Noise": False,  # lots of missing data
            },
            "environmental-engineered": {
                "ema-6-t-out": num,
                "slope-6-t-out": num,
                "ema-6-hr-out": num,
                "slope-6-hr-out": num,
                "ema-48-t-out": num,
                "slope-48-t-out": num,
                "ema-48-hr-out": num,
                "slope-48-hr-out": num,
                "slope-20-t-env": num,
                "slope-20-hr-env": False,  # too many missing data
                "ema-20-hr-env": False,  # too many missing data
                "ema-20-t-env": num,
                # this could be the cutoff point to split them in two
                "slope-60-t-env": False,  # too many missing data
                "ema-60-t-env": False,  # too many missing data
                "slope-60-hr-env": False,  # too many missing data
                "ema-60-hr-env": False,  # too many missing data
            },
            "personal": {
                "Sex": cat,
                "Education": cat,
                "Age": num,
                "Health": cat,
                "BMI": num,
                "userid": False,  # should be set to true only if two users have the same above inputs
            },
            "smartwatch": {
                "location": cat,
                "restingHR": False,
                "HR": num,
                "Steps": False,
                "BMR": False,
                "t-skin": num,
                "t-nb": num,
                "heartRate": False,  # I am using the one from Fitbit API
            },
            "smartwatch-engineered": {
                "ema-20-HR": False,  # too many missing data
                "slope-20-HR": False,  # too many missing data
                "ema-60-HR": False,  # too many missing data
                "slope-60-HR": False,  # too many missing data
                "skin-nb": False,
                # for the moment since I am using t-skin and t-nb as input
                "ema-7-t-skin": num,
                "slope-7-t-skin": num,
                "ema-7-t-nb": num,
                "slope-7-t-nb": num,
                # this could be the cutoff point to split them in two
                "ema-20-t-skin": num,
                "slope-20-t-skin": num,
                "ema-20-t-nb": num,
                "slope-20-t-nb": num,
            },
            "other": {
                "weekend": cat,
                "hour": num,
                "weekday": num,
            },
            "clo_met": {"clothing": cat, "met": cat},
        }

        # scoring metrics for model selection and evaluation
        self.metrics = [
            # "balanced_accuracy",
            "f1_micro",
            "f1_macro",
            # "f1_weighted",
            "cohen_kappa",
            # "accuracy",
        ]

        self.incremental_data_chunks = [
            42,
            126,
            210,
            294,
            378,
            462,
            546,
            630,
            714,
            798,
            882,
            966,
            1050,
        ]

        # define config files for simulations
        __default_config = dict(  # experiment wide
            {
                "iterations": 10,  # debugging value 2, default 100
                "data": "incremental",  # options are 'all' or 'incremental'
                "met_clo_pred": False,  # If False use the logged from RHRn otherwise use clo and met models
                # modeling
                "test_size": 100,
                "k_folds": 5,
                "stratified": True,
                "use_val": True,
            },
        )

        __default_pcm_clo = {
            **__default_config,
            "independent": "clothing",
            "feature_sets": [
                "environmental",
                "smartwatch",
                "other",
            ],
        }

        __default_pcm_met = {
            **__default_config,
            "independent": "met",
            "feature_sets": [
                "environmental",
                "smartwatch",
                "other",
            ],
        }

        __default_pcm_tp_logged_clo_met = {
            **__default_config,
            "independent": "thermal",
            "feature_sets": [
                "environmental",
                "smartwatch",
                "other",
                "clo_met",
            ],
        }

        __default_pcm_tp_logged_clo_met_engineered = {
            **__default_config,
            "independent": "thermal",
            "feature_sets": [
                "environmental",
                "environmental-engineered",
                "smartwatch",
                "smartwatch-engineered",
                "other",
                "clo_met",
            ],
        }

        __default_pcm_tp_no_clo_met = {
            **__default_config,
            "independent": "thermal",
            "feature_sets": [
                "environmental",
                "smartwatch",
                "other",
            ],
        }

        __ml_models = [
            "rdf",
            "xgb",
            "lr",
            "svm",
            "kn",
            "gnb",
            "mlp",
        ]

        self.ml_models = __ml_models.copy()

        self.models_config = {}

        # default configuration
        self.models_config.update(
            {
                "pcm_pmv": {
                    **__default_config,
                }
            }
        )

        # personal clothing model all data at once
        self.models_config.update(
            {
                f"pcm_clo_{model}": {
                    **__default_pcm_clo,
                    "model": model,
                }
                for model in __ml_models
            }
        )

        # personal met model all data at once
        self.models_config.update(
            {
                f"pcm_met_{model}": {
                    **__default_pcm_met,
                    "model": model,
                }
                for model in __ml_models
            }
        )

        # personal comfort models with logged clo and met all data at once
        self.models_config.update(
            {
                f"pcm_tp_{model}_logged_clo_met": {
                    **__default_pcm_tp_logged_clo_met,
                    "model": model,
                }
                for model in __ml_models
            }
        )

        # personal comfort models with logged clo met, and engineered vars all data at once
        self.models_config.update(
            {
                f"pcm_tp_{model}_logged_clo_met_engineered": {
                    **__default_pcm_tp_logged_clo_met_engineered,
                    "model": model,
                }
                for model in __ml_models
            }
        )

        # personal comfort models with no clo or met all data at once
        self.models_config.update(
            {
                f"pcm_tp_{model}_no_clo_met": {
                    **__default_pcm_tp_no_clo_met,
                    "model": model,
                }
                for model in __ml_models
            }
        )

        # Configurations
        #   - [ind var and model type] - [dataset] - [features] - [NOTEs]
        #   [x] clo pcm - incremental           - no engineered variables
        #   [-] clo group model + incr clo pcm  - no engineered variables
        #   [x] met pcm - incremental           - no engineered variables
        #   [-] met group model + incr met pcm  - no engineered variables
        #   [x] tp pcm - all data at once       - without clo and met
        #   [x] tp pcm - all data at once       - with clo and met
        #   [x] tp pcm - all data at once       - with clo, met and engineered
        #   [-] tp pcm - all data at once       - with predicted clo and met all data at once models
        #   [-] tp pcm - all data at once       - with predicted clo, met all data at once models and engineered
        #   [-] tp pcm - all data at once       - without clo and met but with engineered
        #   [-] AFTER the above we will have an idea on how important are clothing and activity as input
        #   [-] tp pcm - incr                   - with predicted clo, met incr models       - ?? check based on previous results
        #   [-] tp pcm - incr                   - with predicted clo, met incr models and engineered - ?? check based on previous results
        #   [-] clo, met group model + incr tp pcm    - with predicted clo, met incr models       - ?? check based on previous results
        #   [-] clo, met group model + incr tp pcm    - with predicted clo, met incr models and engineered - ?? check based on previous results
        #   [-] tp group model + incr tp pcm    - with predicted clo, met incr models       -  ?? check based on previous results
        #   [-] tp group model + incr tp pcm    - with predicted clo, met incr models and engineered - ?? check based on previous results
        #   [-] ashrae 55 + group model + incr pcm

    def recursive_items(self, dictionary):
        for k, v in dictionary.items():
            if type(v) is dict:
                yield from self.recursive_items(v)
            else:
                yield k, v

    def get_variables_by_feature_type(self, features_dict, feature_type):
        data = []
        for key, value in self.recursive_items(features_dict):
            if value == feature_type:
                data.append(key)

        return data

    def import_user_data(self):

        return pd.read_pickle(os.path.join(self.data_dir, "df_user.pkl"))

    def import_cozie_env(self, filter_data=False, filter_col=False):
        """
        This function imports the final dataset which contains all measured variables.

        :param filter_data: if true excludes data exercising, outdoors, transitory,
        t_nb > t_sk -1.
        :param filter_col: drop unnecessary columns such as t-ubi, t-net, etc.
        :return: cozie full dataset
        """

        df = pd.read_pickle(
            os.path.join(self.data_dir, "df_cozie_env.pkl.zip"), compression="zip"
        )

        if filter_data:

            # remove data from those users who did not wear the watch properly
            df["skin-nb"] = df[["t-skin", "t-nb"]].diff(axis=1)["t-nb"]
            threshold = df.groupby(["userid"])["skin-nb"].quantile(0.95).median()

            df = df[df["skin-nb"] < threshold]

            # remove the data from transitory
            df = df[df["change"] == "No"]

            df = df[df["met"] != "Exercising"]

            df = df[df["indoorOutdoor"] == "Indoor"]

            # I am defining the min response speed as the 0.25 quantile of the 1%
            # quantile of all the answers
            threshold_min_time = (
                df.groupby("userid")["responseSpeed"].quantile(0.01).quantile(0.25)
            )

            df = df[df["responseSpeed"] > threshold_min_time]

            df = df[df["location"] != "Other"]

        if filter_col:

            df["timestamp"] = pd.to_datetime(df.unix, unit="s")

            df = df[
                [
                    "BMR",
                    "air-vel",
                    "clothing",
                    "heartRate",
                    "location",
                    "met",
                    "restingHR",
                    "thermal",
                    "userid",
                    "t-skin",
                    "ema-7-t-skin",
                    "slope-7-t-skin",
                    "ema-20-t-skin",
                    "slope-20-t-skin",
                    "t-nb",
                    "ema-7-t-nb",
                    "slope-7-t-nb",
                    "ema-20-t-nb",
                    "slope-20-t-nb",
                    "CO2",
                    "Noise",
                    "Light",
                    "t-env",
                    "rh-env",
                    "hr-env",
                    "slope-20-t-env",
                    "slope-60-t-env",
                    "ema-20-t-env",
                    "ema-60-t-env",
                    "slope-20-hr-env",
                    "slope-60-hr-env",
                    "ema-20-hr-env",
                    "ema-60-hr-env",
                    "hour",
                    "weekday",
                    "weekend",
                    "t-out",
                    "rh-out",
                    "hr-out",
                    "ema-6-t-out",
                    "slope-6-t-out",
                    "ema-48-t-out",
                    "slope-48-t-out",
                    "ema-6-hr-out",
                    "slope-6-hr-out",
                    "ema-48-hr-out",
                    "slope-48-hr-out",
                    "HR",
                    "Steps",
                    "ema-20-HR",
                    "slope-20-HR",
                    "ema-60-HR",
                    "slope-60-HR",
                    "skin-nb",
                    "BMI",
                    "Sex",
                    "Education",
                    "Health",
                    "Age",
                    "timestamp",
                ]
            ]

        return df

    def import_cozie(self):

        return pd.read_pickle(
            os.path.join(self.data_dir, "df_cozie.pkl.zip"), compression="zip"
        )

    def color_map_tc(self):

        return LinearSegmentedColormap.from_list(
            "thermal_comfort", self.tc_colors, N=len(self.tc_colors)
        )

    def color_map_cat(self):

        if "cat_colormap" not in [cm for cm in plt.colormaps()]:
            cmap = LinearSegmentedColormap.from_list(
                "categorical", self.categorical_colors, N=len(self.categorical_colors)
            )
            matplotlib.cm.register_cmap("cat_colormap", cmap)


def init_logger(log_file_location, name="main", limit_log_file_size=True):
    # logger
    log_formatter = logging.Formatter(
        "%(levelname)s line: %(lineno)d; msg = %(message)s"
    )
    if limit_log_file_size:
        my_handler = RotatingFileHandler(
            log_file_location,
            mode="a",
            maxBytes=5 * 1024 * 1024,
            backupCount=1,
            encoding=None,
            delay=0,
        )
    else:
        my_handler = logging.FileHandler(log_file_location)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.INFO)
    app_log = logging.getLogger(name)
    if not app_log.hasHandlers():
        app_log.setLevel(logging.INFO)
        app_log.addHandler(my_handler)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(log_formatter)
        app_log.addHandler(ch)

    return app_log


if __name__ == "__main__":
    self = Configuration()
    from pprint import pprint

    pprint(self.models_config)
