import warnings
import os
import sys

from collections import defaultdict
from utils import save_variable, set_seed, find_pcm
from configuration import Configuration, init_logger


def warn(*args, **kwargs):
    """Function to avoid printing warnings"""


warnings.warn = warn

# load experiment name from CLI
model_option = str(sys.argv[1])

if __name__ == "__main__":
    """Train and evaluate a PCM on specific target variables"""

    config_data_analysis = Configuration()

    model_config = {}
    if model_option not in config_data_analysis.models_config.keys():
        print(
            f"The option '{model_option}' is not supported. Kindly used any of the following:\n{list(config_data_analysis.models_config.keys())}"
        )
        sys.exit()
    else:
        model_config = config_data_analysis.models_config[model_option]

    log_file_location = os.path.join(os.getcwd(), "pcm.log")
    logger = init_logger(log_file_location=log_file_location)
    logger.info(
        f"Starting experiment '{model_option}' with the following feature sets:\n{model_config['feature_sets']}\nuse validation set: {model_config['use_val']}"
    )

    # load only required feature sets
    variables = {
        k: v
        for feature_set in model_config["feature_sets"]
        for k, v in config_data_analysis.features[feature_set].items()
    }
    categorical_features = [x for x in variables if variables[x] == "categorical"]
    numeric_features = [x for x in variables if variables[x] == "numeric"]
    independent_feature = config_data_analysis.models_config[model_option][
        "independent"
    ]

    logger.info(
        f"Categorical features:\n{categorical_features}\nNumerical features:\n{numeric_features}"
    )
    # load mode of training scheme, using all data or incremental
    if model_config["data"] == "all":
        data_list = [-1]
    elif model_config["data"] == "incremental":
        data_list = config_data_analysis.incremental_data_chunks

    # specific data import for the DORN dataset
    df = config_data_analysis.import_cozie_env(filter_data=True, filter_col=True)

    # using only selected features (feature_sets), timestamp, independent variable, and userid
    available_features = categorical_features.copy()
    available_features.extend(numeric_features)
    available_features.append("timestamp")
    available_features.append("userid")
    available_features.append(independent_feature)
    df = df[available_features]

    # remap independent variables
    df[independent_feature] = df[independent_feature].map(
        config_data_analysis.inv_map_cozie[independent_feature]
    )

    logger.info(
        f"PCM training using {model_config['model']} with independent variable {independent_feature}"
    )

    # create folder for model if it doesn't exist
    models_path = os.path.join(os.getcwd(), f"models/{model_option}/")
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    for data in data_list:
        # empty dictionaries for keeping track of results
        (
            dict_pcm_list,
            dict_pcm_metrics_list,
            dict_pcm_explainer_list,
            dict_pcm_shap_values_list,
        ) = ({}, {}, {}, {})
        for metric in config_data_analysis.metrics:
            dict_pcm_list[metric] = defaultdict(list)
            dict_pcm_metrics_list[metric] = defaultdict(list)
            dict_pcm_explainer_list[metric] = defaultdict(list)
            dict_pcm_shap_values_list[metric] = defaultdict(list)

        logger.info(f"using {'all' if data == -1 else data} data points for training")
        for i in range(0, model_config["iterations"]):
            logger.info(f"Iteration {i} ...")

            set_seed(i)

            # move independent variable to the end
            df_ind_var = df.pop(independent_feature)
            df.loc[:, independent_feature] = df_ind_var

            # for PCM use all the dataset without constant features
            (
                dict_pcm,
                dict_pcm_metrics,
                dict_pcm_explainer,
                dict_pcm_shap_values,
            ) = find_pcm(
                df,
                numeric_features,
                categorical_features,
                config_data_analysis.metrics,
                data,
                model_config["stratified"],
                model_config["model"],
                model_config["k_folds"],
                model_config["test_size"],
                model_config["use_val"],
                model_option,
                i,  # seed
            )
            # retrieve available scores
            for metric in config_data_analysis.metrics:
                # append values of each dictionary within the metric for each iteration
                for user, _ in dict_pcm[metric].items():
                    # actual metric value
                    dict_pcm_metrics_list[metric][user].append(
                        dict_pcm_metrics[metric][user]
                    )
                    # model
                    dict_pcm_list[metric][user].append(dict_pcm[metric][user])
                    # shap explainer
                    dict_pcm_explainer_list[metric][user].append(
                        dict_pcm_explainer[metric][user]
                    )
                    # shap values
                    dict_pcm_shap_values_list[metric][user].append(
                        dict_pcm_shap_values[metric][user]
                    )

        # end iterations for loop

        logger.info("Unfolding metrics for each user")
        for metric in config_data_analysis.metrics:
            logger.info(f"Saving {metric} results in models/")

            # finding model with best score
            for user, metrics_list in dict_pcm_metrics_list[metric].items():
                if not all(
                    metric_user is None
                    for metric_user in dict_pcm_metrics_list[metric][user]
                ):
                    best_model_idx = dict_pcm_metrics_list[metric][user].index(
                        max(dict_pcm_metrics_list[metric][user])
                    )
                    best_model = dict_pcm_list[metric][user][best_model_idx]
                    dict_pcm_list[metric][user] = best_model
                else:
                    dict_pcm_list[metric][user] = None

            # save variables
            models_path = os.path.join(os.getcwd(), f"models/{model_option}/")
            if not os.path.exists(models_path):
                os.makedirs(models_path)

            # model
            save_variable(
                f"{models_path}/dict_pcm_{data}_{independent_feature}_{metric}",
                dict_pcm_list[metric],
            )

            # metrics
            save_variable(
                f"{models_path}/dict_pcm_metrics_{data}_{independent_feature}_{metric}",
                dict_pcm_metrics_list[metric],
            )

            # explainer
            save_variable(
                f"{models_path}/dict_pcm_explainer_{data}_{independent_feature}_{metric}",
                dict_pcm_explainer_list[metric],
            )

            # shap_values
            save_variable(
                f"{models_path}/dict_pcm_shap_values_{data}_{independent_feature}_{metric}",
                dict_pcm_shap_values_list[metric],
            )

        # end metric for loop
        logger.info(
            f"Finished all iterations using {'all' if data == -1 else data} data points"
        )

    # end data_list for loop
    logger.info(f"Experiment {model_option} finished")
