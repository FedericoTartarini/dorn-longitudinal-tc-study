"""Exploratory data analysis

I generate a couple of plots to see if there are any visual correlations in the data."""

import matplotlib as mpl

mpl.use("Qt5Agg")  # or can use 'TkAgg', whatever you have/prefer

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
from configuration import Configuration

colors = [
    (214 / 255, 39 / 255, 40 / 255),
    (44 / 255, 0.63, 44 / 255),
    (31 / 255, 119 / 255, 0.71),
]  # R -> G -> B


def sensor_data():
    """Plot sensors data grouped by user id"""

    for sensor in ["netatmo", "ubibot", "ib_skin", "ib_bag", "ib_air"]:

        df = pd.read_pickle(
            os.path.join(config.data_dir, f"df_{sensor}.pkl.zip"), compression="zip"
        ).sort_values("userid")

        # df.userid.unique()

        if (len(df.userid.unique()) != 20) and (
            sensor in ["netatmo", "ubibot", "ib_skin"]
        ):
            raise ValueError("Not all participants were included")

        variables_to_plot = ["Temperature"]
        if sensor == "ubibot":
            variables_to_plot = ["Temperature", "Humidity"]
        elif sensor == "netatmo":
            variables_to_plot = ["Temperature", "Humidity", "CO2", "Noise"]

        for variable in variables_to_plot:
            plt.subplots(constrained_layout=True)
            sns.violinplot(x="userid", y=variable, data=df)
            plt.title(sensor)
            plt.show()


def ans_distribution():
    """See distribution of answers for each question"""

    df = config.import_cozie()

    variables = [
        "air-vel",
        "change",
        "clothing",
        "comfort",
        "indoorOutdoor",
        "location",
        "met",
        "thermal",
    ]

    fig, ax = plt.subplots(2, 4, sharey=True, constrained_layout="True")

    sns.set_theme(style="whitegrid")

    for ix, variable in enumerate(variables):

        df_grouped = df.groupby([variable])["userid"].count()

        df_grouped = df_grouped / df_grouped.sum()

        sns.barplot(x=df_grouped.index, y=df_grouped.values, ax=ax.reshape(-1)[ix])

    plt.show()


def visualize_correlation():

    df_data = pd.read_pickle(
        os.path.join(config.data_dir, "df_cozie_env.pkl.zip"), compression="zip"
    )

    for user in df_data.userid.unique():

        df_user = df_data[df_data.userid == user]

        df_indoor_no_change = df_user[
            (df_user["change"] == "No") & (df_user["indoorOutdoor"] == "Indoor")
        ]

        x, y, c, z, shape, size = (
            "t-env",
            "t-skin",
            "thermal",
            "heartRate",
            "change",
            "clothing",
        )

        for key in [c, shape, size]:
            inv_map = {v: k for k, v in config.map_cozie[key].items()}
            df_indoor_no_change[key] = df_indoor_no_change[key].map(inv_map).values

        _ = df_indoor_no_change[[x, y, z, c, shape, size]].dropna()
        fig = px.scatter_3d(_, x=x, y=y, z=z, color=c, size=size)
        plot(fig)


def look_for_missing_data(save_fig=False):

    df_data = config.import_cozie_env()

    df = pd.DataFrame()

    for user in df_data.userid.unique():

        df_user = df_data[df_data.userid == user]

        df = df.append(df_user.count() / df_user.shape[0], ignore_index=True)

    df.index = df_data.userid.unique()

    # analyze which data are missing and in which percentage
    plt.subplots()
    sns.heatmap(
        df[
            [
                "CO2",
                "Light",
                "Noise",
                "rh-env",
                "hr-env",
                "t-env",
                "t-nb",
                "t-skin",
            ]
        ],
        # df[["clothing", "comfort", "indoorOutdoor", "location", "met", "userid", "BMR", "heartRate", "restingHR", "t-skin", "t-nb", "t-env", "rh-env"]],
        center=0.5,
        annot=True,
    )
    plt.title("Available data as percentage total")
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(config.fig_dir, "missing_data.png"), pad_inches=0, dpi=300
        )
    else:
        plt.show()

    # analyze where from users completed the surveys
    df_loc = df_data.groupby(["userid", "location"])["location"].count()
    df_loc = df_loc.groupby(level="userid").apply(lambda x: x / float(x.sum()))
    df_loc = df_loc.unstack(["location"])

    plt.figure()
    sns.heatmap(df_loc, center=0.5, annot=True)
    plt.title("Location where from people completed survey")
    plt.tight_layout()
    plt.show()


def hr_api_hr_fitbit():
    """plot heart rate from API vs the one logged by cozie"""

    df_data = config.import_cozie_env(filter_data=True)

    (df_data.heartRate - df_data.HR).plot.box()


def ema_vs_reading(variable="t-env"):
    """plot exponential moving average data vs actual reading"""

    df_data = config.import_cozie_env(filter_data=True)

    _df = df_data[["userid", f"slope-20-{variable}", f"slope-60-{variable}"]].set_index(
        "userid"
    )
    _df = _df.stack().reset_index(name=variable)

    plt.figure()
    sns.violinplot(
        x="userid",
        y=variable,
        split=True,
        hue="level_1",
        data=_df,
        inner="quartile",
        cut=0,
    )

    _df = df_data[["userid", f"{variable}", f"ema-20-{variable}"]].set_index("userid")
    _df = _df.stack().reset_index(name=variable)

    plt.figure()
    sns.violinplot(
        x="userid",
        y=variable,
        split=True,
        hue="level_1",
        data=_df,
        inner="quartile",
        cut=0,
    )

    _df = df_data[["userid", f"{variable}", f"ema-60-{variable}"]].set_index("userid")
    _df = _df.stack().reset_index(name=variable)

    plt.figure()
    sns.violinplot(
        x="userid",
        y=variable,
        split=True,
        hue="level_1",
        data=_df,
        inner="quartile",
        cut=0,
    )


def check_sensor_incorrectly_tagged():
    df = config.import_cozie_env()

    for user in df["userid"].unique():
        df_user = df[df.userid == user]

        fig, ax = plt.subplots()
        df_user[["t-bag", "t-net", "t-ubi"]].plot(ax=ax)
        df_user["location"].map(config.inv_map_cozie["location"]).plot(ax=ax)
        df_user[["t-skin", "t-nb"]].plot(ax=ax, linestyle=":")
        ax.scatter(x=df_user.index, y=df_user["t-env"], c="k")
        plt.show()


if __name__ == "__main__":

    config = Configuration()

    plt.close("all")

    # available datasets to explore = ["sensors", "cozie", "missing_data", "eng_features"]
    dataset_to_explore = ["missing_data"]

    for dataset in dataset_to_explore:

        if dataset == "sensors":
            sensor_data()

        if dataset == "cozie":
            ans_distribution()

        if dataset == "missing_data":

            # # 3d plot showing how features affect tsv, plot is not very clear
            # visualize_correlation()

            # visualize if there are missing data
            look_for_missing_data(save_fig=False)

        if dataset == "eng_features":

            # ema vs value measured when completing survey
            ema_vs_reading(variable="t-env")

        if dataset == "HR_API_vs_HR_Fitbit":

            # ema vs value measured when completing survey
            hr_api_hr_fitbit()
