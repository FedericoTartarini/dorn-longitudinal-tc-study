import warnings
import os
from collections import defaultdict

from configuration import Configuration, init_logger
from utils import save_variable, set_seed, clf_metrics, simplified_pmv_model


def warn(*args, **kwargs):
    """Function to avoid printing warnings"""


warnings.warn = warn

if __name__ == "__main__":
    """Evaluate a simplified PMV model"""

    config_data_analysis = Configuration()

    log_file_location = os.path.join(os.getcwd(), "pmv.log")
    logger = init_logger(log_file_location=log_file_location)

    # specific data import for the DORN dataset
    df = config_data_analysis.import_cozie_env(filter_data=True, filter_col=True)

    # remap independent variable ("thermal")
    df["thermal"] = df["thermal"].map(config_data_analysis.inv_map_cozie["thermal"])
    model_config = config_data_analysis.models_config["pcm_pmv"]

    # empty dictionaries
    dict_pmv_list = {}
    for metric in config_data_analysis.metrics:
        dict_pmv_list[metric] = defaultdict(list)

    logger.info("Starting simplified pmv model calculations")
    for i in range(0, model_config["iterations"]):
        logger.info(f"Iteration {i} ...")

        set_seed(i)

        # move independent variable to the end
        df_ind_var = df.pop("thermal")
        df.loc[:, "thermal"] = df_ind_var

        # PMV for each user
        for user in df["userid"].unique():
            df_user = df.copy()
            df_user = df_user[df_user["userid"] == user]
            df_user = df_user.drop(["userid"], axis=1)

            # calculate pmv for each user and for each metric
            df_user_test = df_user.sample(n=model_config["test_size"], random_state=i)
            for metric in config_data_analysis.metrics:
                y_pred = simplified_pmv_model(df_user_test).map(
                    config_data_analysis.inv_map_cozie["thermal"]
                )
                y_test = df_user_test["thermal"]
                pmv_score, _ = clf_metrics(df_user_test["thermal"], y_pred, metric)
                # keep track of all iterations
                dict_pmv_list[metric][user].append(pmv_score)

        # end user for loop

    # end iterations for loop

    logger.info("Unfolding metrics for each user")
    for metric in config_data_analysis.metrics:
        logger.info(f"Saving {metric} results in models/")

        # save variables, create folder if needed
        model_option = "simplified_pmv"
        models_path = os.path.join(os.getcwd(), f"models/{model_option}/")
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        # e.g.`models/{model_option}/dict_pcm_{ind_var}_{metric}.pkl`
        save_variable(
            f"{models_path}/dict_pmv_metrics_-1_pmv_{metric}",
            dict_pmv_list[metric],
        )
