import itertools

import matplotlib as mpl
import numpy as np
import sys
import plotly.express as px
import statsmodels.api as sm
import scipy.stats as stats

mpl.use("Qt5Agg")  # or can use 'TkAgg', whatever you have/prefer
import matplotlib.pyplot as plt

import dataframe_image as dfi
import math
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import os
from configuration import Configuration
from utils import load_variable, open_shapley_files
import json
import re
from datetime import date
from bioinfokit.analys import stat


def import_pickle_files(path, model_name, file_name):

    file = load_variable(os.path.join(path, model_name, file_name))

    file = json.dumps(file)
    df = pd.read_json(file)

    df = df.unstack().reset_index(level=0)

    df.columns = ["userid", "value"]

    return df


def import_results(path, metric_type="metric", force_reload=True, use_preloaded=True):
    """
    path=config.models_results_dir
    metric_type="metric"
    """

    file_path = os.path.join(config.data_dir, "df_accuracy.pkl.zip")

    if use_preloaded and not force_reload:
        return pd.read_pickle(file_path, compression="zip")

    # by default reload only once per day
    if os.path.isfile(file_path) and not force_reload:
        last_modified = os.stat(file_path).st_mtime
        last_modified_date = pd.Timestamp(last_modified, unit="s").date()
        today_date = date.today()
        if today_date != last_modified_date:
            force_reload = True
        else:
            return pd.read_pickle(file_path, compression="zip")

    if force_reload:
        df_results = pd.DataFrame()
        for root, dirs, files in os.walk(path):
            for file in files:
                if metric_type in file:
                    # if (metric_type in file) and ("pcm" in file):
                    # print(os.path.join(root, file))
                    model = root.split("\\")[-1]
                    try:
                        config_file = config.models_config[model]
                    except KeyError:
                        if "_pmv_" in file:
                            config_file = {
                                "model": "pmv",
                                "independent": "thermal",
                                "feature_sets": ["environmental", "clo_met"],
                            }
                        else:
                            # sys.exit(
                            #     "Check import since there is an issue in the file names"
                            # )
                            config_file = None
                    if config_file:
                        file_name = file.replace(".pickle", "")
                        df = import_pickle_files(path, model, file_name)
                        df["algorithm"] = config_file["model"]
                        for metric in config.metrics:
                            if metric in file_name:
                                df["metric"] = metric
                                file_name = file_name.replace(f"_{metric}", "")
                        df["independent"] = config_file["independent"]
                        features = str(config_file["feature_sets"])
                        features = "_".join(
                            [
                                x[:3]
                                for x in re.split(
                                    "', '|-|_", re.sub(r"\['|'\]", "", features)
                                )
                            ]
                        ).replace("_eng", "-eng")
                        df["features"] = features
                        df["number_features"] = len(features.split("_"))
                        df["data"] = file_name.split("_")[-2]

                        df_results = df_results.append(df)

        # save results to file
        df_results.to_pickle(
            os.path.join(config.data_dir, "df_accuracy.pkl.zip"), compression="zip"
        )

        # check that the data imported are correct and do not have issues
        df_check = (
            df_results.groupby(
                ["userid", "algorithm", "independent", "features", "data", "metric"]
            )["value"]
            .count()
            .reset_index()
        )

        # check algo used df_check.columns
        df_values_counts = df_check[~df_check["value"].isin([0, 100])]
        if df_values_counts.shape[0] != 0:
            print("missing results for the following combinations.")
            print(
                df_values_counts.groupby(["independent", "metric"])["value"]
                .count()
                .index
            )

        models = config.ml_models + ["pmv"]
        if sorted(list(df_check["algorithm"].unique())) != sorted(models):
            print("model analyzed: ", list(df_check["algorithm"].unique()))
            raise ValueError("The models do not match")

        metrics_not_matching = [
            x
            for x in df_check["metric"].unique()
            if x
            not in config.metrics + ["balanced_accuracy", "f1_weighted", "accuracy"]
        ]
        if metrics_not_matching != []:
            print("metrics analyzed: ", list(df_check["metric"].unique()))
            raise ValueError("The metrics do not match")

        if list(df_check["userid"].unique()) != list(range(1, 21, 1)):
            print("users analyzed: ", list(df_check["userid"].unique()))
            raise ValueError("Different number of users")

        if sorted(list(df_check["independent"].unique())) != sorted(
            ["clothing", "met", "thermal"]
        ):
            print("independent analyzed: ", list(df_check["independent"].unique()))
            raise ValueError("The independent do not match")

        if sorted(list(df_check["features"].unique())) != sorted(
            [
                "env_clo_met",
                "env_env-eng_sma_sma-eng_oth_clo_met",
                "env_sma_oth",
                "env_sma_oth_clo_met",
            ]
        ):
            print("features analyzed: ", list(df_check["features"].unique()))
            raise ValueError("The features do not match")

        data_not_matching = [
            x
            for x in df_check["data"].unique()
            if int(x) not in config.incremental_data_chunks + [-1]
        ]
        if data_not_matching != []:
            print("data analyzed: ", list(df_check["data"].unique()))
            raise ValueError(f"These {data_not_matching} increments do not match")

        return df_results


def fake_hue(df, plotting_col, grouping_col):
    _df = df.copy()
    _df["Fake_Hue"] = 0
    val = _df[[plotting_col, grouping_col, "Fake_Hue"]].iloc[0]
    val[0] = None
    val[2] = 1
    _df = _df.append(val)
    return _df


def tmp_env_survey():

    df = config.import_cozie_env(filter_data=True)

    df_indoor = df.loc[
        (df.indoorOutdoor == "Indoor") & (df.location.isin(["Work", "Home"])),
        ["userid", "t-env", "location"],
    ]

    # import weather singapore data
    df_w = pd.read_pickle(
        os.path.join(config.data_dir, "df_weather_sg.pkl.zip"), compression="zip"
    )

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 21, hspace=0, wspace=0)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0, 20])
    ax2.spines["left"].set_visible(False)

    sns.violinplot(
        x="userid",
        y="t-env",
        data=df_indoor,
        cut=0,
        scale="count",
        palette=[config.categorical_colors[0]],
        inner="quartile",
        linewidth=1,
        ax=ax1,
    )

    sns.violinplot(
        x="station_id",
        y="t-out",
        data=df_w,
        cut=0,
        scale="count",
        split=True,
        palette=[config.categorical_colors[2]],
        inner="quartile",
        linewidth=1,
        ax=ax2,
    )

    ax1.set(
        ylabel=r"Indoor air temperature ($t_i$) [°C]",
        xlabel=labels["subjects"],
        ylim=(21, 35),
    )
    ax2.set(ylabel="", xlabel="", ylim=(21, 35))
    plt.setp(ax2.get_yticklabels(), visible=False)
    sns.despine(left=True, bottom=True, right=True)
    ax1.grid(axis="y", alpha=0.3)
    ax2.grid(axis="y", alpha=0.3)
    # ax1.legend(
    #     bbox_to_anchor=(0, 0.9, 1, 0.2), loc="lower center", frameon=False, ncol=2
    # )

    figure_name = "tmp_env_survey.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
        dpi=300,
    )


def tmp_skin_nb_survey():

    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=21, hspace=0.05, wspace=0)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0, 20], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, :-1], sharex=ax1)
    ax4 = fig.add_subplot(gs[1, 20], sharey=ax3)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    axs = [ax1, ax3]

    surveys_completed = []

    for index, ax in enumerate(axs):

        df = config.import_cozie_env().copy()
        fig_letter = "a"

        if index == 1:
            fig_letter = "b"
            df = config.import_cozie_env(filter_data=True).copy()

        # I am counting all the survey that were included in the analysis
        df_count = df.groupby(["userid"])["userid"].count()

        df = df[["userid", "t-skin", "t-nb"]].copy().set_index("userid")

        df = df.stack().reset_index()

        df.columns = ["userid", "sensor", "value"]
        df["sensor"] = df["sensor"].map(
            {"t-skin": r"$t_{sk,w}$", "t-nb": r"$t_{nb,w}$"}
        )

        sns.violinplot(
            x="userid",
            y="value",
            data=df,
            cut=0,
            scale="count",
            split=True,
            hue="sensor",
            palette=config.categorical_colors,
            inner="quartile",
            linewidth=1,
            ax=ax,
        )

        # # uncomment this line to count the number of near body data collected
        # df_count = df[df["sensor"] == "t-nb"].groupby(["userid"])["sensor"].count()

        for ix, user in enumerate(df_count.index):
            ax.text(
                ix + 0.3,
                38,
                df_count[df_count.index == user].values[0],
                horizontalalignment="center",
                verticalalignment="top",
                rotation=90,
            )

        surveys_completed.append(df_count.values)

        ax.set(ylabel="Temperature [°C]", ylim=(20, 38))

        if ax.get_subplotspec().rowspan.start == 1:
            ax.set(
                xlabel=labels["subjects"],
            )
            ax.get_legend().remove()
        else:
            ax.set(
                xlabel="",
            )
            ax.legend(
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower center",
                frameon=False,
                ncol=2,
            )

        sns.despine(left=True, bottom=True, right=True)
        ax.grid(axis="y", alpha=0.3)

        ax.text(
            -0.065,
            1,
            fig_letter,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize="large",
            fontweight="demi",
        )

    axs = [ax2, ax4]

    for index, ax in enumerate(axs):

        df = config.import_cozie_env().copy()

        if index == 1:
            df = config.import_cozie_env(filter_data=True).copy()

        df = df[["userid", "t-skin", "t-nb"]].copy().set_index("userid")

        df = df.stack().reset_index()

        df.columns = ["userid", "sensor", "value"]

        df["userid"] = "all"

        sns.violinplot(
            x="userid",
            y="value",
            data=df,
            cut=0,
            scale="count",
            split=True,
            hue="sensor",
            palette=config.categorical_colors,
            inner="quartile",
            linewidth=1,
            ax=ax,
        )

        ax.set(xlabel="", ylabel="")

        sns.despine(left=True, bottom=True, right=True)
        ax.grid(axis="y", alpha=0.3)
        ax.get_legend().remove()

    figure_name = "tmp_skin_nb_survey.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
        dpi=300,
    )

    # calculate the percentage of surveys were excluded
    per_excl = pd.DataFrame(
        1 - surveys_completed[1] / surveys_completed[0], index=df_count.index
    )
    per_excl = per_excl[per_excl[0] > 0.10]


def summary_stats_weather_singapore():

    # import weather singapore data
    df = pd.read_pickle(
        os.path.join(config.data_dir, "df_weather_sg.pkl.zip"), compression="zip"
    )

    # plot data to check that there are not missing data
    plt.figure()
    df["t-out"].plot()
    plt.show()

    # summary stats
    df.describe()


def table_features():
    models_with_metrics = [x for x in config.models_config]
    feature_sets = [x for x in config.features if x != "independent"]

    models = []

    for model in models_with_metrics:
        ml_model = config.ml_models[0]
        if ml_model in model:
            features = config.models_config[model]["feature_sets"]
            model_name = model.replace(f"_{ml_model}", "")
            for feature in feature_sets:
                if feature in features:
                    models.append(
                        {"model": model_name, "feature": feature, "used": "x"}
                    )
                else:
                    models.append({"model": model_name, "feature": feature, "used": ""})

    df = pd.DataFrame(models)
    df = df.replace(to_replace=r"_", value=" ", regex=True)
    df = df.replace(to_replace=r" logged", value=r"", regex=True)
    df = df.replace(to_replace=r"engineered", value=r"eng", regex=True)
    df = df.replace(to_replace=r"clo", value=r"clo", regex=True)
    df = df.replace(to_replace=r"met", value=r"met", regex=True)
    df = df.replace(to_replace=r"smartwatch", value="wear", regex=True)
    df = df.replace(to_replace=r"environmental", value="env", regex=True)
    df = df.replace(to_replace=r"other", value="time", regex=True)
    df = df.set_index(["model", "feature"]).unstack("feature")
    df.columns = df.columns.get_level_values(1)
    df = df.drop(columns=["personal"])
    df.to_latex(
        os.path.join(config.tables_dir, "features.tex"),
        caption="Features used to train the respective model",
        label="tab:features",
        escape=False,
        column_format="lccccccccccccccc",
        index=True,
        header=["\\rotatebox{90}{" + c + "}" for c in df.columns],
    )

    with open(os.path.join(config.tables_dir, "features.tex"), "r+") as f:
        text = f.read().replace("\\begin{tabular}", "\\small \\begin{tabular}")
        text = text.replace(
            "\\bottomrule",
            "\\bottomrule\\noalign{\\vskip 1mm} "
            "\\multicolumn{7}{l}{(clo) -- Self-reported clothing; (met) -- self-reported activity, (env) -- environmental}\\\\ "
            "\\multicolumn{7}{l}{(wear) -- wearable, (eng) -- engineered}",
        )
        f.seek(0)
        f.write(text)
        f.close()


def table_participants():
    df = pd.read_pickle(
        os.path.join(config.data_dir, "df_info_subjects.pkl.zip"), compression="zip"
    )

    df[r"BMI (kg/m$^{2}$)"] = round(df.Weight / ((df.Height / 100) ** 2), 2)

    df[r"HSPS$^{*}$"] = round(
        df[[x for x in df.columns if "Q33" in x]].sum(axis=1) / 12 + 1, 1
    )

    df[r"SWLS$^{**}$"] = (
        df[[x for x in df.columns if "Q34" in x]]
        .replace(
            {
                "Strongly Disagree": 1,
                "Disagree": 2,
                "Slightly Disagree": 3,
                "Neither agree nor disagree": 4,
                "Slightly Agree": 5,
                "Agree": 6,
                "Strongly Agree": 7,
            }
        )
        .sum(axis=1)
    )

    life_sat_col = r"SWLS$^{**}$"

    satisfaction = []
    for score in df[life_sat_col]:
        if score > 30:
            satisfaction.append(f"Extremely satisfied, ({score})")
        elif score > 25:
            satisfaction.append(f"Satisfied, ({score})")
        elif score > 20:
            satisfaction.append(f"Slightly satisfied, ({score})")
        elif score == 20:
            satisfaction.append(f"Neutral, ({score})")
        elif score > 14:
            satisfaction.append(f"Slightly dissatisfied, ({score})")
        elif score > 9:
            satisfaction.append(f"Dissatisfied, ({score})")
        else:
            satisfaction.append(f"Extremely dissatisfied, ({score})")

    df[life_sat_col] = satisfaction

    # df["Education"] = df["Education"].replace(
    #     {
    #         "Doctoral degree": "PhD",
    #         "Master's degree": "MS",
    #         "High school graduate": "HSD",
    #         "Bachelor's degree": "BS",
    #     }
    # )

    df["Sex"] = df["Sex"].replace({"Male": "M", "Female": "F"})

    df = df.sort_values(subject_id_col)

    df[subject_id_col] = df[subject_id_col].astype(int)

    df = df[
        [
            subject_id_col,
            "Sex",
            "Age",
            "Education",
            r"BMI (kg/m$^{2}$)",
            r"HSPS$^{*}$",
            life_sat_col,
        ]
    ].rename(columns={subject_id_col: "ID"})

    df = df[["ID", "Sex", "Age", "Education", "BMI (kg/m$^{2}$)"]]

    df.to_latex(
        os.path.join(config.tables_dir, "subject_info.tex"),
        caption="Information about the subjects.",
        label="tab:subject_info",
        escape=False,
        column_format="cccccccccccccccc",
        index=False,
    )

    with open(os.path.join(config.tables_dir, "subject_info.tex"), "r+") as f:
        text = f.read().replace("\\begin{tabular}", "\small \\begin{tabular}")
        text = text.replace(
            "\\begin{table}",
            "\\begin{table}[h!]",
        )
        # text = text.replace(
        #     "\\bottomrule",
        #     "\\bottomrule \\multicolumn{7}{l}{* Highly Sensitive Person Scale. ** Satisfaction With Life Scale (SWLS) category and score}",
        # )
        f.seek(0)
        f.write(text)
        f.close()

    df.groupby("Sex").count()


def cozie_answers_distributions():
    df = config.import_cozie_env(filter_data=True)

    save_var_latex(
        "per_no_change",
        round(df[df["thermal"] == no_change_tpv_col].shape[0] / df.shape[0] * 100),
    )

    save_var_latex(
        "per_cooler",
        round(df[df["thermal"] == "Cooler"].shape[0] / df.shape[0] * 100),
    )

    save_var_latex(
        "per_sitting",
        round(df[df["met"] == "Sitting"].shape[0] / df.shape[0] * 100),
    )

    save_var_latex(
        "per_far_sensor",
        round(df[df["location"] == "Other"].shape[0] / df.shape[0] * 100),
    )

    save_var_latex(
        "per_air_movement",
        round(df[df["air-vel"] == "Yes"].shape[0] / df.shape[0] * 100),
    )

    save_var_latex(
        "per_clo_light",
        round(df[df["clothing"] == "Light"].shape[0] / df.shape[0] * 100),
    )

    # variables to plot, the order for the x-axis and the palette
    variables = {
        # "comfort": [None, config.categorical_colors, 1],
        "thermal": [None, config.tc_colors, 1, "Would you prefer to be?"],
        # "indoorOutdoor": [None, config.categorical_colors],
        "location": [None, config.categorical_colors, 3, "Are you near a sensor?"],
        "clothing": [
            list(config.map_cozie["clothing"].values()),
            config.categorical_colors,
            4,
            "What are you wearing?",
        ],
        "air-vel": [
            None,
            config.categorical_colors,
            5,
            "Can you perceive air movement around you?",
        ],
        "met": [
            list(config.map_cozie["met"].values())[:3],
            config.categorical_colors,
            6,
            "Activity last 10-min?",
        ],
        # "change": [None, config.categorical_colors],
    }

    fig, ax = plt.subplots(5, 1, figsize=(7, 5))

    for ix, variable in enumerate(variables.keys()):

        _ax = ax.reshape(-1)[ix]

        df_grouped = df.groupby([variable])["userid"].count()
        df_grouped = (df_grouped / df_grouped.sum() * 100).reset_index()

        if variables[variable][0] is not None:
            df_grouped[variable] = df_grouped[variable].astype("category")
            df_grouped[variable].cat.set_categories(
                variables[variable][0], inplace=True
            )
            df_grouped = df_grouped.sort_values([variable])

        df_grouped = (
            df_grouped.set_index(variable).rename(columns={"userid": variable}).T
        )
        df_grouped.plot.barh(
            rot=0, stacked=True, ax=_ax, color=variables[variable][1], legend=False
        )

        increment = 0
        for x, col in enumerate(df_grouped.columns):
            x = increment + df_grouped[col] / 2
            increment += df_grouped[col]
            _ax.text(
                x,
                0,
                f"{df_grouped[col].values[0]:.0f}%\n{col}",
                va="center",
                ha="center",
                size="9",
            )

        sns.despine(left=True, bottom=True, right=True)

        if ix == 4:
            _ax.set(xlabel=labels["percentage"])
        else:
            _ax.set_xticks([])

        _ax.set_title(
            label=f"Q{variables[variable][2]} - {variables[variable][3]}", y=0.75
        )
        _ax.set_yticks([])

    plt.tight_layout()
    figure_name = "ans_distribution.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
        dpi=300,
    )


def tpv_vs_scalars(plot_data_by_user=False):

    if plot_data_by_user:
        fig, axs = plt.subplots(nrows=4, ncols=5, sharey=True, sharex=True)

        df_all = config.import_cozie_env(filter_data=True)

        for ax, user in zip(axs.flat, df_all["userid"].unique()):

            df = df_all[df_all["userid"] == user]

            cat_sorted = list(config.map_cozie["thermal"].values())
            cat_sorted.reverse()

            df["thermal"] = df["thermal"].astype("category")
            df["thermal"].cat.set_categories(cat_sorted, inplace=True)
            df = df.sort_values(["thermal"])

            kde = sns.kdeplot(
                data=df,
                x="t-env",
                hue="thermal",
                ax=ax,
                palette=config.tc_colors,
            )

            ax.set_title(str(user))
            leg = kde.legend_
            leg.remove()

        plt.tight_layout()

        figure_name = "tpv_vs_env_users.png"
        print(f"saved figure: {figure_name}")
        plt.savefig(
            os.path.join(
                config.fig_dir,
                figure_name,
            ),
            pad_inches=0,
        )

    df_all = config.import_cozie_env(filter_data=True)

    variables = {
        "t-skin": r"$t_{sk}$",
        "t-env": r"$t_{env}$",
        "heartRate": r"HR",
        "t-nb": r"$t_{nb}$",
    }

    for var in variables:
        df_all = remove_outliers(_df=df_all, _var=var)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(variables),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [5, 1]},
    )

    axs = axs.flat

    for ix, var in enumerate(variables.keys()):
        kde = sns.kdeplot(
            data=df_all,
            x=var,
            hue="thermal",
            ax=axs[ix],
            cut=0,
            palette=config.tc_colors,
        )

        _ = df_all.copy()
        _["Density"] = 1
        _ = fake_hue(_, var, "Density")
        ax = sns.violinplot(
            data=_,
            y="Density",
            x=var,
            ax=axs[ix + len(variables)],
            cut=0,
            color=config.categorical_colors[0],
            hue="Fake_Hue",
            split=True,
            orient="h",
            inner="quartile",
        )

        ax.legend().remove()
        ax.get_yaxis().set_ticks([])
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))

        if ix == 0:
            h = kde.legend_.legendHandles

        kde.legend_.remove()

    fig.legend(
        h,
        ["Cooler", no_change_tpv_col, "Warmer"],
        loc=9,
        frameon=False,
        borderaxespad=0,
        ncol=3,
    )
    sns.despine(fig, left=True, bottom=True, right=True, top=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    figure_name = "tpv_vs_scalars.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
        pad_inches=0,
    )


def var_by_userid(variable="met", save_fig=False, show_percentages=False):

    df = config.import_cozie_env(filter_data=True)

    _df = df.groupby(["userid", variable])[variable].count().unstack(variable)
    _df.reset_index(inplace=True)

    df_total = _df[_df.columns[1:]].sum(axis=1)
    df_rel = _df[_df.columns[1:]].div(df_total, 0) * 100
    df_rel["userid"] = _df["userid"]

    c_map = plt.cm.get_cmap("cat_colormap")
    if variable == "thermal":
        c_map = config.color_map_tc()

    # plot a Stacked Bar Chart using matplotlib
    df_rel.plot(
        x="userid",
        kind="barh",
        stacked=True,
        mark_right=True,
        cmap=c_map,
        width=0.95,
        figsize=(7, 5),
    )

    plt.legend(
        bbox_to_anchor=(0.5, 1.02),
        loc="center",
        borderaxespad=0,
        ncol=df[variable].unique().__len__(),
        frameon=False,
    )
    sns.despine(left=True, bottom=True, right=True, top=True)

    plt.xlabel(labels["percentage"])
    plt.ylabel(labels["subjects"])

    if show_percentages:
        # add percentages
        for index, row in df_rel.drop(["userid"], axis=1).iterrows():
            cum_sum = 0
            for ixe, el in enumerate(row):
                if el > 7:
                    plt.text(
                        cum_sum + el / 2,
                        index,
                        f"{int(round(el, 0))}%",
                        va="center",
                        ha="center",
                    )
                cum_sum += 0 if np.isnan(el) else el

    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(config.fig_dir, f"{variable}_by_userid.png"),
            pad_inches=0,
            dpi=300,
        )
    else:
        plt.show()


def tpv_by_cat(save_fig=False):

    df = config.import_cozie_env(filter_data=True)

    variables = ["clothing", "met"]

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 3))

    for ix, var in enumerate(variables):

        _df = df.groupby([var, "thermal"])["thermal"].count().unstack("thermal")
        _df.reset_index(inplace=True)

        sorter_index = dict(
            zip(
                list(config.map_cozie[var].values()),
                range(len(list(config.map_cozie[var].values()))),
            )
        )

        _df["sort"] = _df[var].map(sorter_index)
        _df = _df.sort_values("sort").drop(columns=["sort"]).reset_index(drop=True)

        df_total = _df[_df.columns[1:]].sum(axis=1)
        df_rel = _df[_df.columns[1:]].div(df_total, 0) * 100
        df_rel[var] = _df[var]

        # plot a Stacked Bar Chart using matplotlib
        hist = df_rel.plot(
            x=var,
            kind="bar",
            stacked=True,
            mark_right=True,
            cmap=config.color_map_tc(),
            width=0.95,
            ax=ax[ix],
            legend=False,
            rot=0,
        )

        # add percentages
        for index, row in df_rel.drop([var], axis=1).iterrows():
            cum_sum = 0
            for ixe, el in enumerate(row):
                if el > 7:
                    ax[ix].text(
                        index,
                        cum_sum + el / 2,
                        f"{int(round(el, 0))}%",
                        va="center",
                        ha="center",
                    )
                cum_sum += el

        if var == "clothing":
            ax[ix].set_xlabel("Q.5 Clothing")
        else:
            ax[ix].set_xlabel("Q.7 Metabolic rate")

        if ix == 0:
            ax[ix].set_ylabel(labels["percentage"])

        # add number of surveys per category
        for index, row in enumerate(df_total):
            print(row)
            ax[ix].text(
                index,
                104,
                f"{row}",
                va="center",
                ha="center",
            )

        if var == "clothing":
            ax[ix].set_xlabel("Q.5 Clothing")
        else:
            ax[ix].set_xlabel("Q.7 Metabolic rate")

        if ix == 0:
            ax[ix].set_ylabel(labels["percentage"])

    h, l = hist.get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc=9,
        frameon=False,
        borderaxespad=0,
        ncol=3,
    )
    sns.despine(left=True, bottom=True, right=True, top=True)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)

    if save_fig:
        plt.savefig(
            os.path.join(config.fig_dir, "tpv_by_cat.png"), pad_inches=0, dpi=300
        )
    else:
        plt.show()

    # save variables for latex
    save_var_latex(
        "heavy_clo_warmer",
        round(
            df.loc[(df["clothing"] == "Heavy") & (df["thermal"] == "Warmer")].shape[0]
            / df.loc[df["clothing"] == "Heavy"].shape[0]
            * 100
        ),
    )
    save_var_latex(
        "heavy_clo_no_change",
        round(
            df.loc[
                (df["clothing"] == "Heavy") & (df["thermal"] == no_change_tpv_col)
            ].shape[0]
            / df.loc[df["clothing"] == "Heavy"].shape[0]
            * 100
        ),
    )
    save_var_latex(
        "very_light_clo_no_change",
        round(
            df.loc[
                (df["clothing"] == "Very light") & (df["thermal"] == no_change_tpv_col)
            ].shape[0]
            / df.loc[df["clothing"] == "Very light"].shape[0]
            * 100
        ),
    )
    save_var_latex(
        "light_clo_no_change",
        round(
            df.loc[
                (df["clothing"] == "Light") & (df["thermal"] == no_change_tpv_col)
            ].shape[0]
            / df.loc[df["clothing"] == "Light"].shape[0]
            * 100
        ),
    )
    save_var_latex(
        "met_resting_no_change",
        round(
            df.loc[
                (df["met"] == "Resting") & (df["thermal"] == no_change_tpv_col)
            ].shape[0]
            / df.loc[df["met"] == "Resting"].shape[0]
            * 100
        ),
    )


def save_var_latex(key, value):
    import csv

    dict_var = {}

    file_path = os.path.join(config.var_dir, "mydata.dat")

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")


def variables_for_latex():

    # calculate how many surveys were completed
    df = config.import_cozie()

    save_var_latex("tot_surveys", df.shape[0])

    # calculate difference skin temperature and nb temperature
    df = config.import_cozie_env()

    save_var_latex(
        "per_surveys_exercising",
        round(df[df["met"] == "Exercising"].shape[0] / df.shape[0] * 100),
    )

    save_var_latex(
        "per_surveys_outdoor",
        round(df[df["indoorOutdoor"] == "Outdoor"].shape[0] / df.shape[0] * 100),
    )

    save_var_latex(
        "per_surveys_change",
        round(df[df["change"] == "Yes"].shape[0] / df.shape[0] * 100),
    )

    df_diff = df.groupby(["userid"])[["t-skin", "t-nb"]].mean().diff(axis=1)["t-nb"]
    save_var_latex("mean_diff_tnb_tsk", round(df_diff.mean(), 1))
    save_var_latex(
        "diff_tnb_tsk_user_10", round(df_diff[df_diff.index == "10"].values[0], 1)
    )

    # calculate stats tmp at which participants were exposed
    df = config.import_cozie_env(filter_data=True)

    save_var_latex("tot_surveys_filtered", df.shape[0])

    df_env = df[["t-env", "userid"]].groupby("userid")["t-env"].describe()
    df_env[df_env.index == "02"].round(1)
    df_env[df_env.index == "20"].round(1)
    df.groupby("userid")[["clothing", "met"]].describe()


def activity_hr_location():

    df = config.import_cozie_env()

    df = remove_outliers(df, "heartRate")

    df = sort_categorical(df, "met", list(config.map_cozie["met"].values()))

    fig, axs = plt.subplots(1, 2, sharey=True)

    sns.kdeplot(
        data=df,
        x="heartRate",
        hue="met",
        palette=config.categorical_colors,
        ax=axs[0],
        multiple="fill",
        legend=False,
    )

    axs[0].set(xlabel=labels["heart_rate"])

    df["loc_inOut"] = df["indoorOutdoor"].str[:3] + "_" + df["location"]

    bar_plot_stacked_density(
        df,
        x="loc_inOut",
        y="met",
        ax=axs[1],
        cmap="cat_colormap",
        n_col=2,
        legend=False,
        show_percentages=True,
    )

    h, l = axs[1].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc=9,
        frameon=False,
        borderaxespad=0,
        ncol=4,
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(os.path.join(config.fig_dir, "activity_hr_location.png"), pad_inches=0)


def clothing_vs_ind():

    df = config.import_cozie_env()

    x = "t-env"

    df = remove_outliers(df, x, threshold_outliers=0.025)

    df = sort_categorical(df, "clothing", list(config.map_cozie["clothing"].values()))

    fig, axs = plt.subplots(1, 2, sharey=True)

    sns.kdeplot(
        data=df,
        x=x,
        hue="clothing",
        palette=config.categorical_colors,
        ax=axs[0],
        multiple="fill",
        legend=False,
    )

    # axs[0].set(xlabel="Heart rate [beats per minute]")

    df["loc_inOut"] = df["indoorOutdoor"].str[:3] + "_" + df["location"]

    bar_plot_stacked_density(
        df,
        x="loc_inOut",
        y="clothing",
        ax=axs[1],
        cmap="cat_colormap",
        n_col=2,
        legend=False,
        show_percentages=True,
    )

    h, l = axs[1].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc=9,
        frameon=False,
        borderaxespad=0,
        ncol=4,
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(os.path.join(config.fig_dir, "clothing_vs_ind.png"), pad_inches=0)


def remove_outliers(_df, _var, threshold_outliers=0.01):
    q_min, q_max = (
        _df[_var].quantile(threshold_outliers),
        _df[_var].quantile(1 - threshold_outliers),
    )
    return _df[(_df[_var] > q_min) & (_df[_var] < q_max)]


def sort_categorical(df, variable, order):
    df[variable] = df[variable].astype("category")
    df[variable].cat.set_categories(order, inplace=True)
    return df.sort_values([variable])


def distribution_answers_over_time(variable="thermal", save_fig=False):

    df = config.import_cozie_env(filter_data=True)

    var = "clothing"
    user = "04"

    df_user = df[df["userid"] == user]

    colors = ["tab:red", "tab:green", "tab:gray", "tab:orange"]

    df_user.reset_index(inplace=True)
    df_user["grouping"] = pd.cut(
        df_user.index,
        [0, *config.incremental_data_chunks],
        right=False,
        labels=config.incremental_data_chunks,
    )

    df_grouped = df_user.groupby(["grouping", var])["time"].count().unstack()

    f, ax = plt.subplots()

    n_cols = len(df_grouped.columns)
    for ix, col in enumerate(df_grouped.columns):

        data = df_grouped[df_grouped.columns[: (n_cols - ix)]].sum(axis=1)
        sns.barplot(
            x=data.index,
            y=data,
            ax=ax,
            color=colors[ix],
            label=df_grouped.columns[n_cols - ix - 1],
        )

    ax.legend()
    plt.title(f"{var} - user: {user}")
    plt.tight_layout()


def accuracy_incremental_by_model_metric(ind, score, algo, features=None):
    """
    ind="thermal"
    score="f1_micro"
    algo="svm"
    features="env_sma_oth" # None, "env_sma_oth"
    """

    df_results_models = import_results(
        path=config.models_results_dir, force_reload=False
    )

    _df_plot = df_results_models[
        (df_results_models["algorithm"] == algo)
        & (df_results_models["independent"] == ind)
        & (df_results_models["metric"] == score)
    ]
    _df_plot["data"] = _df_plot["data"].astype("float").values
    _df_plot = _df_plot.sort_values(["userid", "data"])

    if features:
        _df_plot = _df_plot[_df_plot["features"] == features]

    _fig, _ax = plt.subplots(
        2,
        2,
        gridspec_kw={"height_ratios": [1, 7], "hspace": 0, "wspace": 0},
        constrained_layout=True,
        figsize=(7, 4.5),
    )

    _ax = _ax.flat
    df_user = _df_plot[_df_plot["data"] != -1]
    mean = df_user.groupby("data")["value"].mean().values
    std = df_user.groupby("data")["value"].std().values
    x = df_user.data.unique()
    _ax[2].plot(x[:8], mean[:8], c="#000", lw=3)
    _ax[2].fill_between(
        x[:8], mean[:8] + std[:8] / 2, mean[:8] - std[:8] / 2, alpha=0.5
    )

    markers = itertools.cycle([(i, j, 0) for i in range(2, 10) for j in range(1, 3)])
    for ix, user in enumerate(sorted(df_user["userid"].unique())):
        df_u = df_user[df_user["userid"] == user]
        mean = df_u.groupby("data")["value"].median()
        x = df_u.data.sort_values().unique()
        _ax[2].plot(
            x,
            mean,
            c=sns.color_palette("tab20", 20)[ix],
            marker=next(markers),
            alpha=0.75,
        )

    sns.barplot(
        data=df_user.groupby("data")["value"].count().reset_index() / 100,
        x="data",
        y="value",
        color="gray",
        ax=_ax[0],
    )

    df_user = _df_plot[_df_plot["data"] == -1]
    round(df_user.groupby("userid")["value"].std().std(), 3)
    round(df_user.query("userid == 7")["value"].mean(), 2)
    sns.boxenplot(
        data=df_user,
        x="userid",
        y="value",
        ax=_ax[3],
        palette=sns.color_palette("tab20", 20),
        linewidth=0.5,
        showfliers=False,
    )

    for ax in _ax:
        ax.label_outer()
    _ax[3].set_xticklabels(_ax[3].get_xticklabels(), rotation=90)

    sns.despine(left=True, bottom=True, right=True, top=True)
    _ax[3].set(xlabel=labels["subjects"])
    _ax[2].set(xlabel=labels["data_points"], ylabel=labels["accuracy"])
    _ax[0].set(xlim=(-0.5, 10.5), ylabel="# People")
    _ax[1].set_xticks([])
    _ax[1].set_yticks([])
    [_, _, y_min, y_max] = _ax[2].axis()
    _ax[3].set(ylim=(y_min, y_max))

    for ax, text in zip(_ax[2:], ["a", "b"]):
        ax.text(
            -0.15,
            0.97,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize="large",
            fontweight="demi",
        )

    figure_name = f"{ind}_{score}_{algo}.png"
    if features:
        figure_name = f"{ind}_{score}_{algo}_{features}.png"

    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
        pad_inches=0,
        dpi=300,
    )


def check_status_simulations(independent=None, treemap=False, icicle=True):
    df = import_results(path=config.models_results_dir, force_reload=False)

    df_runs = (
        df.groupby(["independent", "algorithm", "features", "metric", "data"])["value"]
        .count()
        .reset_index()
    )
    df_runs = df_runs[df_runs["data"].isin(["42", "-1"])]
    df_runs.loc[df_runs["data"] == "42", "data"] = "incremental"
    df_runs.loc[df_runs["data"] == "-1", "data"] = "allData"

    df = df_runs
    if independent:
        df = df_runs[df_runs["independent"] == independent]

    if treemap:
        fig = px.treemap(
            df,
            path=["independent", "features", "data", "algorithm", "metric"],
            values="value",
            color="value",
            color_continuous_scale="RdBu",
        )
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.show()
        fig.write_image(os.path.join(config.fig_dir, f"treemap_{independent}.png"))

    if icicle:
        fig = px.icicle(
            df,
            path=["independent", "features", "data", "algorithm", "metric"],
            values="value",
        )
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.show()
        fig.write_image(os.path.join(config.fig_dir, f"icicle_{independent}.png"))


def compare_different_model_results_by_feature(ind, algo, score, data="-1"):
    """
    ind = "clothing"
    algo = "svm"
    score = "f1_micro"
    data="-1"
    """
    df = import_results(path=config.models_results_dir, force_reload=False)

    _df_plot = df[
        (df["independent"] == ind)
        & (df["data"] == data)
        & (df["algorithm"] == algo)
        & (df["metric"] == score)
    ]
    _df_plot.groupby("algorithm")["algorithm"].count()
    _df_plot = _df_plot.sort_values(["userid", "number_features"])
    _fig, ax = plt.subplots(constrained_layout=True)
    sns.boxenplot(
        x="userid",
        y="value",
        data=_df_plot,
        hue="features",
        ax=ax,
        linewidth=0.5,
        showfliers=False,
    )
    overall_stats_model = _df_plot["value"].describe()
    ax.fill_between(
        ax.get_xlim(),
        overall_stats_model[["25%"]],
        overall_stats_model[["75%"]],
        color="lightsteelblue",
        alpha=0.5,
    )
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.set(xlim=ax.get_xlim())

    plt.legend(
        bbox_to_anchor=(0.5, 1.02),
        loc="center",
        borderaxespad=0,
        ncol=3,
        frameon=False,
    )

    plt.savefig(
        os.path.join(
            config.fig_dir, f"boxen_comp_metric_by_feature_{ind}_{algo}_{score}.png"
        ),
        pad_inches=0,
        dpi=300,
    )


def compare_different_model_results(ind, score="f1_micro", y="value", data="-1"):
    """
    ind = "thermal"
    score = "f1_micro"
    y="value"
    data="-1"
    """
    df = import_results(path=config.models_results_dir, force_reload=False)

    _df_plot = df[(df["independent"] == ind) & (df["data"] == data)]
    # _df_plot.groupby(["metric", "algorithm"])["algorithm"].count()
    if ind == "thermal":
        _fig = plt.figure()

        gs = _fig.add_gridspec(1, 7, hspace=0, wspace=0)
        ax1 = _fig.add_subplot(gs[0, :-1])
        ax2 = _fig.add_subplot(gs[0, 6])
        ax2.spines["left"].set_visible(False)
        df_ml = _df_plot[
            (_df_plot["metric"] == score) & (_df_plot["algorithm"] != "pmv")
        ]
        df_ml["features"] = df_ml["features"].replace(
            {
                "env_env-eng_sma_sma-eng_oth_clo_met": "env, time, wear, clo-met, env-hist, wear-hist",
                "env_sma_oth": "env, time, wear",
                "env_sma_oth_clo_met": "env, time, wear, clo-met",
            }
        )
        median = df_ml[
            df_ml["algorithm"].isin(["xgb", "rdf", "mlp", "kn", "lr", "svm"])
        ][y].median()
        save_var_latex("median_f1_score_no_gnb", median)
        df_ml["algorithm"] = df_ml["algorithm"].str.upper()
        sns.boxplot(
            data=df_ml.sort_values("features"),
            x="algorithm",
            hue="features",
            y=y,
            ax=ax1,
            linewidth=0.5,
            showfliers=False,
        )
        df_gnb = df_ml[
            (df_ml["algorithm"] == "gnb")
            & (df_ml["features"] == "env, time, wear, clo-met, env-hist, wear-hist")
        ]
        df_ml.groupby(["algorithm", "features"])[y].mean().sort_values()
        ax1.set(
            ylim=(0, 1.01),
            xlim=(-0.5, 6.5),
            ylabel=labels["accuracy"],
            xlabel="Supervised machine learning algorithm",
        )
        _df_pmv = _df_plot[
            (_df_plot["metric"] == score) & (_df_plot["algorithm"] == "pmv")
        ]
        _df_pmv["algorithm"] = _df_pmv["algorithm"].str.upper()
        sns.boxplot(
            data=_df_pmv,
            x="algorithm",
            y=y,
            ax=ax2,
            color="lightsteelblue",
            width=0.267,
            linewidth=0.5,
            showfliers=False,
        )
        pmv_scores = _df_pmv["value"].describe()
        ax1.fill_between(
            [-0.5, 7.5],
            pmv_scores[["25%"]],
            pmv_scores[["75%"]],
            color="lightsteelblue",
            alpha=0.5,
        )

        # t-test gnb results vs pmv
        stats.ttest_ind(df_gnb[y], _df_pmv[y])

        ax2.set(ylabel="", xlabel="", ylim=(0, 1.01))
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_yticks([])
        plt.tight_layout()
        ax1.legend(frameon=False)
    else:
        _fig, _ax = plt.subplots(constrained_layout=True)
        sns.boxplot(
            data=_df_plot,
            x="algorithm",
            hue="metric",
            y=y,
            linewidth=0.5,
        )
        median = _df_plot[
            (_df_plot["metric"] == score)
            & (_df_plot["algorithm"].isin(["xgb", "rdf", "mlp", "kn", "lr", "svm"]))
        ][y].median()
        save_var_latex(f"median_f1_score_no_gnb_{ind}", round(median, 2))
        _ax.set(ylim=(-0.25, 1.01), ylabel=labels["accuracy"])
        _ax.axhline(0, -0.5, 5.5, color="lightgray", linestyle="--")
        _ax.legend(
            bbox_to_anchor=(0.5, 0.05),
            loc="center",
            borderaxespad=0,
            ncol=3,
            frameon=False,
        )
    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.show()

    figure_name = f"boxen_comp_metrics_{ind}.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
        pad_inches=0,
        dpi=300,
    )

    # test if the prediction scores across the model have a significant difference
    _df_plot = _df_plot[_df_plot["metric"] == score]
    print(
        _df_plot.groupby("algorithm")[y].describe().sort_values("50%", ascending=False)
    )
    top_performing = (
        _df_plot.groupby("algorithm")[y].median().sort_values(ascending=False).index[:5]
    )
    _df_plot = _df_plot[_df_plot["algorithm"].isin(top_performing)]
    _fig, _ax = plt.subplots(constrained_layout=True)
    sns.boxenplot(data=_df_plot, x="algorithm", y=y)
    data = {}
    for type in _df_plot.algorithm.unique():
        data[type] = list(_df_plot[_df_plot["algorithm"] == type][y].values)
    print(
        stats.kruskal(
            data[top_performing[0]],
            data[top_performing[1]],
            data[top_performing[2]],
            data[top_performing[3]],
            data[top_performing[4]],
        )
    )
    # res = stat()
    # res.anova_stat(df=_df_plot, res_var=y, anova_model=f"{y} ~ C(algorithm)")
    # print(res.anova_summary)
    # res.tukey_hsd(
    #     df=_df_plot, res_var=y, xfac_var="algorithm", anova_model=f"{y} ~ C(algorithm)"
    # )
    # res.tukey_summary
    # res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
    # sm.qqplot(res.anova_std_residuals, line="45")
    # plt.xlabel("Theoretical Quantiles")
    # plt.ylabel("Standardized Residuals")
    # plt.show()
    #
    # # histogram
    # plt.hist(res.anova_model_out.resid, bins="auto", histtype="bar", ec="k")
    # plt.xlabel("Residuals")
    # plt.ylabel("Frequency")
    # plt.show()


def accuracy_ml_models_clo_met(score="f1_micro", y="value", data="-1"):
    """
    score = "f1_micro"
    y="value"
    data="-1"
    """
    df = import_results(path=config.models_results_dir, force_reload=False)

    df_data = df[
        (df["independent"].isin(["clothing", "met"]))
        & (df["data"] == data)
        & (df["metric"] == score)
    ]

    # test if the prediction scores across the model have a significant difference
    df_data = df_data.groupby(["independent", "algorithm"])[y].describe().round(2)
    f = {x: "{:.2f}" for x in df_data.columns}
    df_styled = df_data.reset_index().style.format(f).background_gradient().hide_index()

    figure_name = f"table_accuracy_ml_models_clo_met.png"
    print(f"saved figure: {figure_name}")

    dfi.export(
        df_styled,
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
    )


def accuracy_ml_models_thermal(score="f1_micro", y="value", data="-1"):
    """
    score = "f1_micro"
    y="value"
    data="-1"
    """
    df = import_results(path=config.models_results_dir, force_reload=False)

    _df_plot = df[(df["independent"] == "thermal") & (df["data"] == data)]
    df_ml = _df_plot[(_df_plot["metric"] == score) & (_df_plot["algorithm"] != "pmv")]
    df_data = df_ml.groupby(["algorithm", "features"])[y].describe()

    # test if the prediction scores across the model have a significant difference
    f = {x: "{:.2f}" for x in df_data.columns}
    df_styled = df_data.reset_index().style.format(f).background_gradient().hide_index()

    figure_name = f"table_f1_micro_ml_models_thermal.png"
    print(f"saved figure: {figure_name}")

    dfi.export(
        df_styled,
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
    )

    # only select best performing models xgb, lr, svm, rdf, lr,
    df_anova = df_ml[df_ml["algorithm"].isin(["xgb", "rdf", "mlp", "lr", "svm"])]

    data = {}
    for type in ["xgb", "rdf", "mlp", "lr", "svm"]:
        for feature in df_anova["features"].unique():
            data[f"{type}_{feature}"] = list(
                df_anova[
                    (df_anova["algorithm"] == type) & (df_anova["features"] == feature)
                ][y].values
            )
    for key in data:
        print(key)
        print(len(data[key]))

    print(
        stats.kruskal(
            data["xgb_env_env-eng_sma_sma-eng_oth_clo_met"],
            data["xgb_env_sma_oth"],
            data["xgb_env_sma_oth_clo_met"],
            data["rdf_env_env-eng_sma_sma-eng_oth_clo_met"],
            data["rdf_env_sma_oth"],
            data["rdf_env_sma_oth_clo_met"],
            data["mlp_env_env-eng_sma_sma-eng_oth_clo_met"],
            data["mlp_env_sma_oth"],
            data["mlp_env_sma_oth_clo_met"],
            data["lr_env_env-eng_sma_sma-eng_oth_clo_met"],
            data["lr_env_sma_oth"],
            data["lr_env_sma_oth_clo_met"],
            data["svm_env_env-eng_sma_sma-eng_oth_clo_met"],
            data["svm_env_sma_oth"],
            data["svm_env_sma_oth_clo_met"],
        )
    )


def import_process_shapely_files(
    dep_var="clothing",
    metric="f1_micro",
    data="-1",
    algo="rdf",
    features=None,
):

    """
    dep_var = "clothing"
    metric = "f1_micro"
    data = "-1"
    algo = "rdf"
    features = "logged_clo_met"
    user = "all"
    """

    df = open_shapley_files(
        dep_var=dep_var, metric=metric, data=data, algo=algo, features=features
    )

    n_features = df["01"][1][1].data.shape[0]
    shapely_values = np.empty((0, n_features))
    feature_data = np.empty((0, n_features))
    user_array = []
    base_values = []
    for user in df:  # for each user
        for ix, run in enumerate(df[user]):  # for each run, total number of runs is 100
            try:
                shapely_values = np.append(
                    shapely_values, df[user][ix][:].values, axis=0
                )
                feature_data = np.append(feature_data, df[user][ix][:].data, axis=0)
                base_values = base_values + list(df[user][ix][:].base_values)
                user_array = user_array + [user] * df[user][ix][:].values.shape[0]
            except TypeError:
                print(f"No data for user {user} and run {ix}")

    df_shap = pd.DataFrame(shapely_values)
    df_data = pd.DataFrame(feature_data)
    df_shap.columns = df["01"][1].feature_names
    df_data.columns = df["01"][1].feature_names
    index = pd.MultiIndex.from_tuples(
        list(zip(user_array, base_values)), names=["user", "base_val"]
    )
    df_shap.index = index
    df_shap = df_shap.stack().reset_index()
    df_shap.columns = ["user", "base_val", "feature", "value"]
    df_data.index = user_array
    df_data = df_data.stack().reset_index()
    df_data.columns = ["user", "feature", "data"]
    df_shap["data"] = df_data["data"]

    return df_shap


def shapely_bar_plot(df, model_info, ax=None, order_feat=None, cmap=None):
    """
    df=df_shap_model
            model_info={
                "dep_var": "thermal",
                "metric": "f1_micro",
                "data": "-1",
                "algo": model,
                "features": "no_clo_met",
                # {None, 'no_clo_met', 'logged_clo_met_engineered', 'logged_clo_met'}
            }
            ax=ax
            order_feat=order
            cmap=cmap
    """

    var_names = {
        "t-env": r"$t_{i}$",
        "t-nb": r"$t_{nb,w}$",
        "HR": "HR",
        "t-skin": r"$t_{sk,w}$",
        "hr-env": r"$W_{i}$",
        "hour": "hour",
        "t-out": r"$t_{out}$",
        "weekday": "weekday",
        "location": "location",
        "hr-out": "$W_{out}$]",
        "weekend": "weekend",
    }
    df_shap_abs = df.copy()
    df_shap_abs["value"] = df_shap_abs["value"].abs()
    if order_feat is not None:
        order = df_shap_abs.groupby("feature")["value"].describe()
        order = order.reindex(order_feat)
    else:
        order = (
            df_shap_abs.groupby("feature")["value"]
            .describe()
            .sort_values("mean", ascending=False)
        )
    if not ax:
        fig, ax = plt.subplots(
            constrained_layout=True,
        )
    if cmap:
        plt.bar(order.index.map(var_names), order["mean"], color=cmap)
    else:
        plt.bar(
            order.index.map(var_names), order["mean"], color=sns.color_palette("Paired")
        )

    for item in ax.get_xticklabels():
        item.set_rotation(90)
    sns.despine(left=True, bottom=True, right=True, top=True)
    ax.set(ylabel="mean(|SHAP value|)", xlabel="")

    if not ax:
        plt.show()
        figure_name = f"shapely_bar_{model_info['dep_var']}_{model_info['metric']}_{model_info['data']}_{model_info['algo']}_{model_info['features']}.png"
        print(f"saved figure: {figure_name}")
        plt.savefig(
            os.path.join(
                config.fig_dir,
                figure_name,
            ),
        )

    return order.index


def shapely_bar_all(df):
    fig = plt.figure(figsize=(7, 10), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=2, hspace=0.05, wspace=0.05, figure=fig)

    ax = fig.add_subplot(gs[3, :])
    # env = {x[0]:x[1] for x in zip(['t-env', 'hr-env', 't-out', 'hr-out'], sns.color_palette("Greys", 6)[1:5][::-1])}
    # wear = {x[0]:x[1] for x in zip(['t-nb', 'HR', 't-skin', 'location'], sns.color_palette("Purples", 6)[1:5][::-1])}
    # time = {x[0]:x[1] for x in zip(['hour', 'weekday', 'weekend'], sns.color_palette("Oranges", 5)[1:4][::-1])}
    # cmap = {**env, **wear, **time}
    # cmap = list(order.map(cmap))
    cmap = [
        (0.35912341407151094, 0.35912341407151094, 0.35912341407151094),
        (0.440722798923491, 0.36772010765090346, 0.6653902345251825),
        (0.5513264129181085, 0.537916186082276, 0.7524490580545944),
        (0.6878892733564014, 0.6835832372164552, 0.829834678969627),
        (0.5085736255286428, 0.5085736255286428, 0.5085736255286428),
        (0.9137254901960784, 0.3686274509803921, 0.050980392156862744),
        (0.6770011534025375, 0.6770011534025375, 0.6770011534025375),
        (0.9914186851211073, 0.550726643598616, 0.23277201076509035),
        (0.8207612456747405, 0.8218992695117262, 0.9044982698961938),
        (0.819115724721261, 0.819115724721261, 0.819115724721261),
        (0.9921568627450981, 0.726797385620915, 0.49150326797385624),
    ]

    order = shapely_bar_plot(
        df=df,
        model_info={
            "dep_var": "thermal",
            "metric": "f1_micro",
            "data": "-1",
            "algo": "all",
            "features": "no_clo_met",
            # {None, 'no_clo_met', 'logged_clo_met_engineered', 'logged_clo_met'}
        },
        ax=ax,
        cmap=cmap,
    )
    ax.set(
        ylim=(0, 0.12),
    )

    ax_pre = None
    for ix, model in enumerate(["rdf", "xgb", "lr", "svm", "kn", "mlp"]):
        row_index = int(ix / 2)
        col_index = ix % 2
        if col_index == 1:
            ax = fig.add_subplot(gs[row_index, col_index], sharey=ax_pre)
        elif col_index == 0 and row_index != 0:
            ax = fig.add_subplot(gs[row_index, col_index], sharex=ax_pre)
        else:
            ax = fig.add_subplot(gs[row_index, col_index])
        print(row_index, col_index, model)
        df_shap_model = df[df["ml_model"] == model]
        df_shap_model["ml_model"] == df_shap_model["ml_model"].str.upper()
        shapely_bar_plot(
            df=df_shap_model,
            model_info={
                "dep_var": "thermal",
                "metric": "f1_micro",
                "data": "-1",
                "algo": model,
                "features": "no_clo_met",
                # {None, 'no_clo_met', 'logged_clo_met_engineered', 'logged_clo_met'}
            },
            ax=ax,
            order_feat=order,
            cmap=cmap,
        )
        ax_pre = ax
        if row_index < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
            # ax.set(
            #     xlabel="",
            # )
        if col_index == 1:
            ax.set(
                ylabel="",
            )
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set(
            ylim=(0, 0.12),
        )
        ax.annotate(
            model.upper(),
            (0.8, 0.8),
            xycoords="axes fraction",
            va="center",
            ha="center",
        )

    plt.tight_layout()

    plt.show()
    figure_name = f"shapely_all.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
        dpi=300,
    )


def shapely_bar_violin(df, model_info):

    # filter out some features
    df_shap = df.copy()

    fig = plt.figure()
    grid = plt.GridSpec(2, 2)
    df_shap_abs = df_shap.copy()
    df_shap_abs["value"] = df_shap_abs["value"].abs()
    order = (
        df_shap_abs.groupby("feature")["value"].mean().sort_values(ascending=False)[:5]
    )
    stats_main_feat = (
        df_shap_abs[df_shap_abs["feature"] == order.index[0]]
        .describe()
        .to_dict()["value"]
    )
    df_shap_abs = df_shap_abs[df_shap_abs["feature"].isin(order.index)]
    ax0 = fig.add_subplot(grid[0, 0])
    sns.barplot(
        x="feature",
        y="value",
        data=df_shap_abs,
        order=order.index,
        ax=ax0,
        errcolor="lightgray",
    )
    ax0.text(
        0.75,
        0.75,
        "all data",
        ha="center",
        va="center",
        transform=ax0.transAxes,
    )
    ax0.set(ylabel="mean(|SHAP value|)", xlabel="")

    ax1 = fig.add_subplot(grid[0, 1])
    # plt.close("all")
    sns.violinplot(
        x="feature",
        y="value",
        data=df_shap,
        showfliers=False,
        cut=0,
        order=order.index,
        scale="width",
        linewidth=1,
        ax=ax1,
    )
    ax1.axhline(y=0)
    ax1.set(ylim=(-0.15, 0.15))
    ax1.set(ylabel=shap_label, xlabel="")

    # boxenplot for each user for the most important feature
    ax2 = fig.add_subplot(grid[1, :])
    x_lim = (-0.5, 17.5)
    ax2.axhline(stats_main_feat["mean"], c="gray")
    ax2.fill_between(
        x_lim,
        stats_main_feat["mean"] - stats_main_feat["std"],
        stats_main_feat["mean"] + stats_main_feat["std"],
        color="lightgray",
        alpha=0.5,
    )
    sns.violinplot(
        x="user",
        y="value",
        data=df_shap[df_shap["feature"] == order.index[0]],
        linewidth=0.5,
        ax=ax2,
        showfliers=False,
        cut=0,
        scale="width",
    )
    ax2.set(
        xlim=x_lim,
        ylim=(-1, 1),
        ylabel=f"SHAP value - {order.index[0]}",
        xlabel=labels["subjects"],
    )
    ax2.text(16, stats_main_feat["mean"] + 0.9 * stats_main_feat["std"], "SD", va="top")
    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.tight_layout()
    figure_name = f"shapely_violin_{model_info['dep_var']}_{model_info['metric']}_{model_info['data']}_{model_info['algo']}_{model_info['features']}.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
    )


def shapely_values_scatter_distribution(
    df, model_info, variable_name="t-env", user_id="02"
):

    fig, ax = plt.subplots(
        2,
        1,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [5, 1]},
    )
    df_plot = df[(df["feature"] == variable_name) & (df["user"] == user_id)]
    sns.scatterplot(
        data=df_plot,
        x="data",
        y="value",
        hue="base_val",
        palette="deep",
        legend=False,
        ax=ax[0],
    )
    sns.despine()
    ax[0].set(ylabel=shap_label, xlabel=f"{variable_name}")

    sns.histplot(df_plot, x="base_val", ax=ax[1])

    figure_name = f"shapely_scatter_{user_id}_{variable_name}_{model_info['dep_var']}_{model_info['metric']}_{model_info['data']}_{model_info['algo']}_{model_info['features']}.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
    )


def shapely_values_kdeplot(df, model_info, variable_name="t-env", user_id="02"):

    fig, ax = plt.subplots()
    df_plot = df[(df["feature"] == variable_name) & (df["user"] == user_id)]
    sns.kdeplot(data=df_plot, x="data", y="value", fill=True, cmap="viridis")
    plt.tight_layout()
    sns.despine()
    ax.set(ylabel=shap_label, xlabel=f"{variable_name}")

    figure_name = f"shapely_kde_{user_id}_{variable_name}_{model_info['dep_var']}_{model_info['metric']}_{model_info['data']}_{model_info['algo']}_{model_info['features']}.png"
    print(f"saved figure: {figure_name}")
    plt.savefig(
        os.path.join(
            config.fig_dir,
            figure_name,
        ),
    )


def summary_html(open_html=False):
    # show all figures in generated
    fig_to_exclude = [
        "comforttool.png",
        "confusion_matrices_combined.png",
        "cozie_card.png",
        "ibuttons_fitbit.png",
        "linkedin_qr.png",
        "pythermalcomfort.png",
        "sinberbest.png",
        "tpv_vs_env_users.png",
        "clothing_vs_ind.png",
    ]

    file_path = os.path.join(os.path.dirname(config.fig_dir), "all_figures.md")

    with open(file_path) as f:
        content = f.readlines()

    fig_included = [
        re.match(r"^.*/(.*)\).*$", x).group(1) for x in content if "![" in x
    ]

    for root, dirs, figures in os.walk(config.fig_dir):
        for fig in figures:
            if (
                ("png" in fig)
                and (fig not in fig_to_exclude)
                and (fig not in fig_included)
            ):
                print(f"{fig} not included in the final HTML")

    # code to automatically generate the markdown file
    generate_markdown_programmatically = False
    if generate_markdown_programmatically:
        f = open_html(file_path, "w")
        f.write("---\ntitle: Output\n---\n\n")
        for line in fig_included:
            if "png" in line:
                f.write(f"### {line} \n")
                f.write(f"![](./figures/{line})" + "{width=30%}  \n \n")
            else:
                f.write(f"{line} \n \n")
        f.close()

    output_file_path = os.path.join(
        os.path.dirname(os.path.dirname(config.fig_dir)), "src", "all_figures.html"
    )
    start_command = ""
    if open_html:
        start_command = f'start "" "{output_file_path}"'
    os.system(
        f'cd "{os.path.dirname(config.fig_dir)}" & '
        f'pandoc -s "{file_path}" -c all_figures.css -o "{output_file_path}" & '
        f"{start_command}"
    )


def bar_plot_stacked_density(
    _df,
    x,
    y,
    ax,
    show_percentages=True,
    orient="v",
    cmap="Pastel1",
    n_col=3,
    legend=True,
):

    _df = _df.groupby([x, y])[y].count().unstack(y)

    _df = _df[_df > _df.sum().sum() * 0.0025].dropna(how="all")

    df_total = _df.sum(axis=1)
    df_rel = _df[_df.columns].div(df_total, 0)
    df_rel.columns = df_rel.columns.tolist()
    df_rel.reset_index(inplace=True)

    # plot a Stacked Bar Chart using matplotlib
    df_rel.plot(
        x=x,
        kind="barh" if orient == "h" else "bar",
        stacked=True,
        mark_right=True,
        cmap=cmap,
        width=0.95,
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    if legend:
        ax.legend(
            bbox_to_anchor=(0.5, 1.02),
            loc="center",
            borderaxespad=0,
            ncol=n_col,
            frameon=False,
        )
    else:
        ax.legend().remove()

    sns.despine(left=True, bottom=True, right=True, top=True)

    ax.set(xlabel="")

    if show_percentages:
        # add percentages
        for index, row in df_rel.drop([x], axis=1).iterrows():
            cum_sum = 0
            for ixe, el in enumerate(row):
                if not math.isnan(el):
                    x_text = index if orient == "v" else cum_sum + el / 2
                    y_text = index if orient != "v" else cum_sum + el / 2
                    if el > 0.07:
                        ax.text(
                            x_text,
                            y_text,
                            f"{round(el, 2)}",
                            va="center",
                            ha="center",
                        )
                    cum_sum += el


if __name__ == "__main__":

    plt.close("all")

    config = Configuration()
    config.color_map_cat()

    mpl.rcParams["figure.figsize"] = [7, 4]
    mpl.rcParams["image.cmap"] = plt.cm.get_cmap("cat_colormap")
    sns.set_palette(config.categorical_colors)

    subject_id_col = "Subject ID"
    no_change_tpv_col = "No Change"
    shap_label = "SHAP value"

    labels = {
        "percentage": "Percentage [%]",
        "subjects": "Participant ID",
        "heart_rate": "Heart rate [beats per minute]",
        "accuracy": "Prediction accuracy, F1-micro score",
        "data_points": "Number of training data points",
    }

    variables_for_latex()

if __name__ == "__plot__":

    tmp_skin_nb_survey()
    cozie_answers_distributions()
    clothing_vs_ind()
    activity_hr_location()
    tmp_env_survey()
    summary_stats_weather_singapore()
    tpv_vs_scalars()
    var_by_userid(variable="thermal", save_fig=True, show_percentages=True)
    var_by_userid(variable="clothing", save_fig=True, show_percentages=True)
    var_by_userid(variable="met", save_fig=True, show_percentages=True)
    tpv_by_cat(save_fig=True)
    accuracy_incremental_by_model_metric(ind="clothing", score="f1_micro", algo="svm")
    accuracy_incremental_by_model_metric(ind="met", score="f1_micro", algo="svm")
    accuracy_incremental_by_model_metric(
        ind="thermal", score="f1_micro", algo="svm", features="env_sma_oth"
    )
    accuracy_incremental_by_model_metric(
        ind="thermal", score="f1_micro", algo="lr", features="env_sma_oth"
    )
    # compare_different_model_results(ind="clothing")
    # compare_different_model_results(ind="met")
    compare_different_model_results(ind="thermal", score="f1_micro")
    # compare_different_model_results_by_feature(
    #     ind="thermal", algo="lr", score="f1_micro"
    # )

    df_shapely_all = pd.DataFrame()
    for model in config.ml_models:
        print(model)
        model_to_plot = {
            "dep_var": "thermal",
            "metric": "f1_micro",
            "data": "-1",
            "algo": model,
            "features": "no_clo_met",  # {None, 'no_clo_met', 'logged_clo_met_engineered', 'logged_clo_met'}
        }
        df_shapely = import_process_shapely_files(
            dep_var=model_to_plot["dep_var"],
            metric=model_to_plot["metric"],
            data=model_to_plot["data"],
            algo=model_to_plot["algo"],
            features=model_to_plot["features"],
        )

        shapely_bar_plot(df=df_shapely, model_info=model_to_plot)

        df_shapely["ml_model"] = model
        df_shapely_all = df_shapely_all.append(df_shapely)

    df_shapely_all.to_pickle(
        os.path.join(
            config.data_dir,
            "df_pkl_thermal_f1_micro_-1_all_models_no_clo_met.pkl.zip",
        ),
        compression="zip",
    )
    df_shapely_all = pd.read_pickle(
        os.path.join(
            config.data_dir,
            "df_pkl_thermal_f1_micro_-1_all_models_no_clo_met.pkl.zip",
        ),
        compression="zip",
    )
    shapely_bar_all(df=df_shapely_all)

    # shapely_bar_violin(df_shapely, model_to_plot)
    # shapely_values_scatter_distribution(df_shapely, model_to_plot)
    # shapely_values_kdeplot(df_shapely, model_to_plot)

    table_participants()

    table_features()

    accuracy_ml_models_clo_met()

    accuracy_ml_models_thermal()

    summary_html(open_html=False)

    check_status_simulations(independent="clothing")
    check_status_simulations(independent="met")
    check_status_simulations(independent="thermal")
