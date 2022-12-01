"""
This function is used only to import and preprocess the data from Influx.

If you need to upload more iButton data to Influx use the function `import_ibutton_data`
which is located in Work and Projects\SinBerBEST\20 Cozie\Code

All the dataframe that are imported are then saved into the data directory so in future,
I will be able to access the data even if I do not longer have access to Influx.
"""
import os
from sqlalchemy import create_engine
import numpy as np
from influxdb import InfluxDBClient
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from configuration import Configuration, init_logger
import psychrolib
import matplotlib.pyplot as plt

import secret

psychrolib.SetUnitSystem(psychrolib.SI)
participant_id_col = "Participant ID"
start_survey_col = "Start Survey"


def import_google_sheet(file_name):
    """Import data in the Google Sheet and return a dataframe"""

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]

    # import credentials file
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials_google.json", scope
    )
    client = gspread.authorize(credentials)
    # open file
    sheet = client.open(file_name).sheet1
    # get all records
    data = sheet.get_all_records()

    return pd.DataFrame(data)


def calc_slope(x):
    slope = np.polyfit(range(len(x)), x, 1)[0]
    return slope


def resample_ema_slope(
    data, variable="Temperature", resample_interval="10T", windows=(2, 6), suffix=""
):
    data = data.copy().resample(resample_interval).mean()

    for window in windows:
        data[f"ema-{window}-{variable}{suffix}"] = (
            data[variable].ewm(span=window).mean()
        )
        data[f"slope-{window}-{variable}{suffix}"] = (
            data[variable].rolling(window).apply(calc_slope)
        )
        # drop all the ema values calculated when main variable was not there
        data.loc[
            data[f"slope-{window}-{variable}{suffix}"].isna(),
            f"ema-{window}-{variable}{suffix}",
        ] = np.nan

    return data


class ImportData:
    def __init__(self, import_user_data=False):
        self.influx_host = "lonepine-64d016d6.influxcloud.net"
        self.time_zone = config.time_zone
        self.cwd = os.getcwd()
        self.data_dir = config.data_dir

        # define client to connect to Influx
        self.influx_cl = InfluxDBClient(
            host=self.influx_host,
            port=8086,
            username=secret.influx_username,
            password=secret.influx_buds_password,
            database="testingDB",
            ssl=True,
            verify_ssl=True,
        )

        if import_user_data:
            self.import_users_data()
        else:
            self.df_users = config.import_user_data()

        self.participants = self.df_users[participant_id_col].unique()

        log_file_location = os.path.join(os.getcwd(), "import_data.log")
        self.logger = init_logger(log_file_location=log_file_location)

    def import_outdoor_weather_data_sg(self):
        """Import Singapore weather data, these data are queried from data.gov.sg and currently
        stored on Federico's laptop"""

        self.logger.info("importing singapore outdoor weather data")

        engine = create_engine(
            "postgresql://postgres:D8wHlR560cy@localhost:5432/weather_sg"
        )

        # I am querying Banyan Road data 1.256,103.679
        df = pd.read_sql_query(
            "Select * from observations where "
            "date(timestamp ) >= '2020-03-30' and "
            "date(timestamp ) <= '2021-01-01'",
            engine,
        )

        df.set_index("timestamp", inplace=True)

        df = df.copy().resample("1H").mean()

        # replacing missing data with data from previous day
        df[
            (df.index >= pd.Timestamp(2020, 6, 8))
            & (df.index < pd.Timestamp(2020, 6, 12))
        ] = df[
            (df.index >= pd.Timestamp(2020, 6, 4))
            & (df.index < pd.Timestamp(2020, 6, 8))
        ].values

        df[
            (df.index >= pd.Timestamp(2020, 7, 1, 10))
            & (df.index < pd.Timestamp(2020, 7, 2, 16))
        ] = df[
            (df.index >= pd.Timestamp(2020, 7, 3, 10))
            & (df.index < pd.Timestamp(2020, 7, 4, 16))
        ].values

        df = resample_ema_slope(
            df, variable="Tmp", resample_interval="1H", windows=(6, 48), suffix="-out"
        )

        df["HumidityRatio"] = psychrolib.GetHumRatioFromRelHum(
            df.Tmp.values, df.RH.values / 100, 101325
        )

        df = resample_ema_slope(
            df,
            variable="HumidityRatio",
            resample_interval="1H",
            windows=(6, 48),
            suffix="-out",
        )

        df["station_id"] = "SG"

        df = df.rename(
            columns={
                "Tmp": "t-out",
                "RH": "rh-out",
                "HumidityRatio": "hr-out",
                "ema-6-Tmp-out": "ema-6-t-out",
                "slope-6-Tmp-out": "slope-6-t-out",
                "ema-48-Tmp-out": "ema-48-t-out",
                "slope-48-Tmp-out": "slope-48-t-out",
                "ema-6-HumidityRatio-out": "ema-6-hr-out",
                "slope-6-HumidityRatio-out": "slope-6-hr-out",
                "ema-48-HumidityRatio-out": "ema-48-hr-out",
                "slope-48-HumidityRatio-out": "slope-48-hr-out",
            }
        )

        df.to_pickle(
            os.path.join(self.data_dir, "df_weather_sg.pkl.zip"), compression="zip"
        )

        self.logger.info("success")

    def influx_to_df(self, query):
        """Query Influx and return a dataframe"""
        try:
            result = self.influx_cl.query(query)
            df = pd.DataFrame(result[result.keys()[0]])

            df.index = pd.to_datetime(df.time)
            df.index = df.index.tz_convert(self.time_zone)
            return df.drop(columns=["time"])
        except IndexError:
            return pd.DataFrame()

    def import_users_data(self):
        """Import participants' information"""

        experiment_id = "dorn"
        file_name_on_boarding = f"{experiment_id}-on-boarding"
        file_equipment = "Equipment"

        # import data on-boarding questionnaire
        df_participants = import_google_sheet(file_name=file_name_on_boarding)
        df_participants.columns = [x.strip() for x in df_participants.columns]
        df_participants[start_survey_col] = pd.to_datetime(
            df_participants["Timestamp"], format="%m/%d/%Y %H:%M:%S"
        ).dt.tz_localize(self.time_zone)
        df_participants[participant_id_col] = df_participants[
            participant_id_col
        ].str.lower()

        df_equipment = import_google_sheet(file_name=file_equipment)
        df_equipment[participant_id_col] = [
            experiment_id + str(x) if x > 9 else f"{experiment_id}0{x}"
            for x in df_equipment["Id"]
        ]

        df_users = df_equipment.merge(
            df_participants[[participant_id_col, start_survey_col]],
            on=participant_id_col,
        )

        df_users = df_users[
            [
                "Id",
                "Fitbit_ID",
                "Netatmo_MAC",
                "Ubibot_Channel",
                "Ibutton_Skin",
                "Ibutton_air",
                "Ibutton_Bag",
                participant_id_col,
                start_survey_col,
            ]
        ]

        df_users.to_pickle(os.path.join(self.data_dir, "df_user.pkl"))

        self.df_users = df_users

    def import_ibutton_data(self):
        """Import all the ibutton data that are stored in Influx"""

        df = self.influx_to_df('SELECT * FROM "people"."autogen"."ibutton"')

        df_ib_skin = df[
            df.ibutton_id.isin(self.df_users.Ibutton_Skin.values)
        ].reset_index()

        self.df_users[participant_id_col] = self.df_users[participant_id_col].replace(
            {"dorn": ""}, regex=True
        )

        # dorn 02 gave his skin ibutton to dorn 03 on the 7th of october
        df_ib_skin.loc[
            (df_ib_skin.time > pd.Timestamp(2020, 10, 7).tz_localize(self.time_zone))
            & (df_ib_skin.ibutton_id == "520000000D2BDB53"),
            ["ibutton_id"],
        ] = "950000000D4DD653"

        df_ib_skin = df_ib_skin.merge(
            self.df_users[[participant_id_col, "Ibutton_Skin"]],
            left_on="ibutton_id",
            right_on="Ibutton_Skin",
        )[["Temperature", participant_id_col, "time"]].rename(
            columns={participant_id_col: "userid"}
        )

        df_ib_skin.index = pd.to_datetime(df_ib_skin["time"])

        df_skin = pd.DataFrame()
        for user in df_ib_skin["userid"].unique():
            df_user = df_ib_skin[df_ib_skin["userid"] == user]
            df_user = resample_ema_slope(
                df_user, resample_interval="3T", windows=(7, 20), suffix="-skin"
            )
            df_user["userid"] = user
            df_skin = df_skin.append(df_user)

        df_skin.to_pickle(
            os.path.join(self.data_dir, "df_ib_skin.pkl.zip"), compression="zip"
        )

        df_ib_air = df[
            df.ibutton_id.isin(self.df_users.Ibutton_air.values)
        ].reset_index()

        df_ib_air = df_ib_air.merge(
            self.df_users[[participant_id_col, "Ibutton_air"]],
            left_on="ibutton_id",
            right_on="Ibutton_air",
        )[["Temperature", participant_id_col, "time"]].rename(
            columns={participant_id_col: "userid"}
        )

        df_ib_air.index = pd.to_datetime(df_ib_air["time"])

        df_skin = pd.DataFrame()
        for user in df_ib_air["userid"].unique():
            df_user = df_ib_air[df_ib_air["userid"] == user]
            df_user = resample_ema_slope(
                df_user, resample_interval="3T", windows=(7, 20), suffix="-air"
            )
            df_user["userid"] = user
            df_skin = df_skin.append(df_user)

        df_skin.to_pickle(
            os.path.join(self.data_dir, "df_ib_air.pkl.zip"), compression="zip"
        )

        df_ib_bag = df[
            df.ibutton_id.isin(self.df_users.Ibutton_Bag.values)
        ].reset_index()

        df_ib_bag = df_ib_bag.merge(
            self.df_users[[participant_id_col, "Ibutton_Bag"]],
            left_on="ibutton_id",
            right_on="Ibutton_Bag",
        )[["Temperature", participant_id_col, "time"]].rename(
            columns={participant_id_col: "userid"}
        )

        df_ib_bag.index = pd.to_datetime(df_ib_bag["time"])

        df_skin = pd.DataFrame()
        for user in df_ib_bag["userid"].unique():
            df_user = df_ib_bag[df_ib_bag["userid"] == user]
            df_user = resample_ema_slope(
                df_user, resample_interval="3T", windows=(7, 20), suffix="-bag"
            )
            df_user["userid"] = user
            df_skin = df_skin.append(df_user)

        df_skin.to_pickle(
            os.path.join(self.data_dir, "df_ib_bag.pkl.zip"), compression="zip"
        )

    def import_cozie(self):

        # get the number of responses per user
        df_cozie = pd.DataFrame()

        for user_id in self.participants:

            start_date = (
                self.df_users.loc[
                    self.df_users[participant_id_col] == user_id, start_survey_col
                ]
                .values[0]
                .astype("uint64")
            )

            df = self.influx_to_df(
                f'SELECT * FROM "coziePublic"."autogen"."fitbit" where '
                f"userid='{user_id}' AND time > {start_date}"
            )

            if user_id == "dorn02":

                df.loc[
                    (df.location == 10)
                    & (df.indoorOutdoor == 11)
                    & (df.lat > 1.312)
                    & (df.lat < 1.314)
                    & (df.lon > 103.877)
                    & (df.lat < 103.879),
                    "location",
                ] = 11

                df.loc[
                    (df.location == 10) & (df.indoorOutdoor == 11) & (df.met == 8),
                    "location",
                ] = 11

                df.loc[
                    (df.location == 10)
                    & (df.indoorOutdoor == 11)
                    & (df.index.month < 6)
                    & (df.met == 9)
                    & (df.lat.isna()),
                    "location",
                ] = 11

            df_cozie = df_cozie.append(df)

        df_cozie = df_cozie[
            [
                "BMR",
                "air-vel",
                "bodyPresence",
                "change",
                "clothing",
                "comfort",
                "experimentid",
                "heartRate",
                "indoorOutdoor",
                "lat",
                "location",
                "lon",
                "met",
                "responseSpeed",
                "restingHR",
                "thermal",
                "userid",
                "voteLog",
            ]
        ]

        df_cozie["userid"] = df_cozie["userid"].replace({"dorn": ""}, regex=True)

        df_cozie.to_pickle(
            os.path.join(self.data_dir, "df_cozie.pkl.zip"), compression="zip"
        )

    def import_environmental_data(self):

        df_netatmo = pd.DataFrame()
        df_ubibot = pd.DataFrame()

        for user_id in self.participants:

            netatmo_mac = self.df_users[self.df_users[participant_id_col] == user_id][
                "Netatmo_MAC"
            ].values[0]
            ubibot_id = self.df_users[self.df_users[participant_id_col] == user_id][
                "Ubibot_Channel"
            ].values[0]
            start_date = (
                self.df_users.loc[
                    self.df_users[participant_id_col] == user_id, start_survey_col
                ]
                .values[0]
                .astype("uint64")
            )

            user_id = user_id.replace("dorn", "")

            # import netatmo data
            df = self.influx_to_df(
                f'SELECT "Temperature", "Humidity", "CO2", "Noise" '
                f'FROM "spaces"."autogen"."netatmo" '
                f"where station_id='{netatmo_mac}' "
                f"AND module='indoor'"
                f"AND time > {start_date}"
            )

            df["HumidityRatio"] = psychrolib.GetHumRatioFromRelHum(
                df.Temperature.values, df.Humidity.values / 100, 101325
            )

            df = resample_ema_slope(df, variable="Temperature")
            df = resample_ema_slope(df, variable="HumidityRatio")
            df["userid"] = user_id
            df_netatmo = df_netatmo.append(df)

            # import ubibot data
            df = self.influx_to_df(
                f'SELECT "Temperature", "Humidity", "Light" '
                f'FROM "spaces"."autogen"."ubibot" '
                f"where station_id='{ubibot_id}' "
                f"AND time > {start_date}"
            )

            df["HumidityRatio"] = psychrolib.GetHumRatioFromRelHum(
                df.Temperature.values, df.Humidity.values / 100, 101325
            )

            df = resample_ema_slope(df, variable="Temperature")
            df = resample_ema_slope(df, variable="HumidityRatio")
            df["userid"] = user_id
            df_ubibot = df_ubibot.append(df)

        df_netatmo.to_pickle(
            os.path.join(self.data_dir, "df_netatmo.pkl.zip"), compression="zip"
        )

        df_ubibot.to_pickle(
            os.path.join(self.data_dir, "df_ubibot.pkl.zip"), compression="zip"
        )

    def import_fitbit_api(self):

        df_fitbit = pd.DataFrame()

        for user_id in self.participants:

            start_date = (
                self.df_users.loc[
                    self.df_users[participant_id_col] == user_id, start_survey_col
                ]
                .values[0]
                .astype("uint64")
            )

            end_date = (
                pd.to_datetime(["2021-01-01 00:00:00"]).values[0].astype("uint64")
            )

            user_id = user_id.replace("dorn", "")

            df = self.influx_to_df(
                f'SELECT * FROM "people"."autogen"."fitbit_api" where '
                f"user_id='{user_id}' AND "
                f"time > {start_date} AND time < {end_date}"
            )

            df = resample_ema_slope(
                df, variable="HR", resample_interval="1T", windows=(20, 60)
            )

            df["userid"] = user_id

            df_fitbit = df_fitbit.append(df)

        df_fitbit.to_pickle(
            os.path.join(self.data_dir, "df_fitbit.pkl.zip"), compression="zip"
        )

    def import_qualtrics_data(self):
        path_file = r"C:\Users\sbbfti\Google Drive\Work and Projects\SinBerBEST\20 Cozie\Dorn\Surveys Data\qualtrics-survey.csv"

        df = pd.read_csv(path_file)[
            [
                "user_id",
                "Q26",
                "Q28",
                "Q32",
                "Q34",
                "Q36",
                "Q38",
                "Q40",
                "Q42",
                "Q44",
                "Q46",
                "Q48",
                "Q54_1",
                "Q54_2",
                "Q54_3",
                "Q54_4",
                "Q54_5",
                "Q33_1",
                "Q33_2",
                "Q33_3",
                "Q33_4",
                "Q33_5",
                "Q33_6",
                "Q33_7",
                "Q33_8",
                "Q33_9",
                "Q33_10",
                "Q33_11",
                "Q33_12",
                "Q34_1",
                "Q34_2",
                "Q34_3",
                "Q34_4",
                "Q34_5",
                "Q36_1",
                "Q36_2",
                "Q36_3",
                "Q36_4",
                "Q36_5",
                "Q36_6",
                "Q36_7",
                "Q36_8",
                "Q36_9",
                "Q36_10",
            ]
        ]

        df = df.dropna(subset=["user_id"])

        df["Age"] = 2020 - df.Q26.astype(int)

        df = df.rename(
            columns={
                "Q28": "Sex",
                "user_id": "Subject ID",
                "Q32": "Height",
                "Q34": "Weight",
                "Q36": "Education",
                "Q38": "Place of birth",
                "Q42": "Years in SG",
                "Q44": "postcode_home",
                "Q46": "postcode_work",
                "Q48": "Health",
                "Q54_1": "Cold hand",
                "Q54_2": "Exercise",
                "Q54_3": "Coffee consumption",
                "Q54_4": "Alcohol consumption",
                "Q54_5": "Smoke",
            }
        )

        map_q33 = {
            "Not at all": 0,
            "2": 1,
            "8": 2,
            "Moderately": 3,
            "11": 4,
            "15": 5,
            "Extremely": 6,
        }
        for col in [
            "Q33_1",
            "Q33_2",
            "Q33_3",
            "Q33_4",
            "Q33_5",
            "Q33_6",
            "Q33_7",
            "Q33_8",
            "Q33_9",
            "Q33_10",
            "Q33_11",
            "Q33_12",
        ]:
            df[col] = df[col].map(map_q33)

        df.Weight = df.Weight.str.split("kg", expand=True)[0].astype(int)
        df.Height = df.Height.str.split("cm", expand=True)[0].astype(int)

        df.to_pickle(
            os.path.join(self.data_dir, "df_info_subjects.pkl.zip"), compression="zip"
        )

    def combine_datasets(self):
        """Firstly I am importing all the datasets and merging them with df_cozie
        each variable collected during the study is added as new column to df_cozie

        Secondly, I am analyzing people responses and only keeping relevant variables.
        Such as tmp_netatmo only when participant was at home.

        Finally, I am saving the file in /data
        """

        df_cozie = pd.read_pickle(
            os.path.join(self.data_dir, "df_cozie.pkl.zip"),
            compression="zip",
        )

        df_cozie["unix"] = (df_cozie.index.astype(np.int64) // 10 ** 9).values
        df_cozie.reset_index(inplace=True)

        for sensor in ["ib_skin", "ib_bag", "ib_air", "netatmo", "ubibot"]:

            # rename columns
            if sensor == "ib_skin":
                rename_col = {
                    "Temperature": "t-skin",
                    "ema-7-Temperature-skin": "ema-7-t-skin",
                    "slope-7-Temperature-skin": "slope-7-t-skin",
                    "ema-20-Temperature-skin": "ema-20-t-skin",
                    "slope-20-Temperature-skin": "slope-20-t-skin",
                }
                tolerance = 60 * 3.5  # in seconds
            elif sensor == "ib_bag":
                rename_col = {"Temperature": "t-bag"}
                tolerance = 60 * 3.5
            elif sensor == "ib_air":
                rename_col = {
                    "Temperature": "t-nb",
                    "ema-7-Temperature-air": "ema-7-t-nb",
                    "slope-7-Temperature-air": "slope-7-t-nb",
                    "ema-20-Temperature-air": "ema-20-t-nb",
                    "slope-20-Temperature-air": "slope-20-t-nb",
                }
                tolerance = 60 * 3.5
            elif sensor == "netatmo":
                rename_col = {
                    "Temperature": "t-net",
                    "Humidity": "rh-net",
                    "HumidityRatio": "hr-net",
                    "ema-2-Temperature": "ema-20-net-t",
                    "ema-6-Temperature": "ema-60-net-t",
                    "slope-2-Temperature": "slope-20-net-t",
                    "slope-6-Temperature": "slope-60-net-t",
                    "ema-2-HumidityRatio": "ema-20-net-hr",
                    "ema-6-HumidityRatio": "ema-60-net-hr",
                    "slope-2-HumidityRatio": "slope-20-net-hr",
                    "slope-6-HumidityRatio": "slope-60-net-hr",
                }
                tolerance = 60 * 10.5
            else:  # ubibot case
                rename_col = {
                    "Temperature": "t-ubi",
                    "Humidity": "rh-ubi",
                    "HumidityRatio": "hr-ubi",
                    "ema-2-Temperature": "ema-20-ubi-t",
                    "ema-6-Temperature": "ema-60-ubi-t",
                    "slope-2-Temperature": "slope-20-ubi-t",
                    "slope-6-Temperature": "slope-60-ubi-t",
                    "ema-2-HumidityRatio": "ema-20-ubi-hr",
                    "ema-6-HumidityRatio": "ema-60-ubi-hr",
                    "slope-2-HumidityRatio": "slope-20-ubi-hr",
                    "slope-6-HumidityRatio": "slope-60-ubi-hr",
                }
                tolerance = 60 * 10.5

            df_sensor = pd.read_pickle(
                os.path.join(self.data_dir, f"df_{sensor}.pkl.zip"), compression="zip"
            )

            df_sensor["unix"] = (df_sensor.index.astype(np.int64) // 10 ** 9).values
            try:
                df_sensor = df_sensor.drop(columns=["time"])
            except KeyError:
                pass

            df_all = pd.DataFrame()

            for user in df_cozie.userid.unique():

                _df_c = df_cozie[df_cozie.userid == user]
                _df_s = df_sensor[df_sensor.userid == user]

                df_all = df_all.append(
                    pd.merge_asof(
                        _df_c,
                        _df_s.drop(columns=["userid"]),
                        on="unix",
                        tolerance=int(tolerance),
                    )
                )

            df_all = df_all.rename(columns=rename_col)

            df_cozie = df_all.copy()

        df_cozie.set_index("time", inplace=True)

        # now that all data have been added to df_cozie I am only keeping useful data
        df_cozie["t-env"] = np.nan
        df_cozie["rh-env"] = np.nan
        df_cozie["hr-env"] = np.nan
        df_cozie["slope-20-t-env"] = np.nan
        df_cozie["slope-60-t-env"] = np.nan
        df_cozie["ema-20-t-env"] = np.nan
        df_cozie["ema-60-t-env"] = np.nan
        df_cozie["slope-20-hr-env"] = np.nan
        df_cozie["slope-60-hr-env"] = np.nan
        df_cozie["ema-20-hr-env"] = np.nan
        df_cozie["ema-60-hr-env"] = np.nan

        for location in [8, 9, 11]:
            shared_columns_to_update = [
                "t-env",
                "rh-env",
                "hr-env",
                "ema-20-t-env",
                "ema-60-t-env",
                "slope-20-t-env",
                "slope-60-t-env",
                "ema-20-hr-env",
                "ema-60-hr-env",
                "slope-20-hr-env",
                "slope-60-hr-env",
            ]
            if location == 11:  # home
                reference_col = [
                    "t-net",
                    "rh-net",
                    "hr-net",
                    "ema-20-net-t",
                    "ema-60-net-t",
                    "slope-20-net-t",
                    "slope-60-net-t",
                    "ema-20-net-hr",
                    "ema-60-net-hr",
                    "slope-20-net-hr",
                    "slope-60-net-hr",
                ]
            elif location == 9:  # work
                reference_col = [
                    "t-ubi",
                    "rh-ubi",
                    "hr-ubi",
                    "ema-20-ubi-t",
                    "ema-60-ubi-t",
                    "slope-20-ubi-t",
                    "slope-60-ubi-t",
                    "ema-20-ubi-hr",
                    "ema-60-ubi-hr",
                    "slope-20-ubi-hr",
                    "slope-60-ubi-hr",
                ]
            elif location == 8:  # portable
                reference_col = [
                    "t-bag",
                    "ema-7-Temperature-bag",
                    "ema-20-Temperature-bag",
                    "slope-7-Temperature-bag",
                    "slope-20-Temperature-bag",
                ]
                shared_columns_to_update = [
                    "t-env",
                    "ema-20-t-env",
                    "ema-60-t-env",
                    "slope-20-t-env",
                    "slope-60-t-env",
                ]

            df_cozie.loc[
                df_cozie.location == location, shared_columns_to_update
            ] = df_cozie.loc[df_cozie.location == location, reference_col].values

            if location == 9:

                df_cozie.loc[df_cozie.location != location, "Light"] = None

            if location == 11:

                df_cozie.loc[df_cozie.location != location, ["CO2", "Noise"]] = None

        for key in config.map_cozie:
            df_cozie[key] = df_cozie[key].map(config.map_cozie[key]).values

        df_cozie["hour"] = df_cozie.index.hour
        df_cozie["weekday"] = df_cozie.index.weekday
        df_cozie["weekend"] = [1 if x >= 5 else 0 for x in df_cozie["weekday"]]
        df_cozie.reset_index(inplace=True)

        df_weather = pd.read_pickle(
            os.path.join(self.data_dir, "df_weather_sg.pkl.zip"), compression="zip"
        )

        df_weather["unix"] = df_weather.index.astype(np.int64) // 10 ** 9
        df_cozie = pd.merge_asof(
            df_cozie.sort_values("unix"),
            df_weather[
                [
                    "unix",
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
                ]
            ],
            on="unix",
            tolerance=3600,
        )

        # import fitbit data
        df_fitbit = pd.read_pickle(
            os.path.join(self.data_dir, "df_fitbit.pkl.zip"), compression="zip"
        )
        df_fitbit["unix"] = df_fitbit.index.astype(np.int64) // 10 ** 9
        df_all = pd.DataFrame()

        for user in df_cozie.userid.unique():

            _df_c = df_cozie[df_cozie.userid == user]
            _df_s = df_fitbit[df_fitbit.userid == user]

            df_all = df_all.append(
                pd.merge_asof(
                    _df_c.sort_values("unix"),
                    _df_s.drop(columns=["userid"]),
                    on="unix",
                    tolerance=60,
                )
            )

        # df_all.loc[df_all["HR"].isna(), ["userid", "unix"]].groupby("userid").count()
        # df_all.loc[df_all["heartRate"].isna(), ["userid", "unix"]].groupby("userid").count()

        # if Fitbit API data was not available I am using the Cozie HR value
        df_all.loc[df_all["HR"].isna(), "HR"] = df_all.loc[
            df_all["HR"].isna(), "heartRate"
        ].values

        df_all.rename(
            columns={
                "restingHR": "restingHeartRate",
                "HR": "HeartRate",
                "ema-20-HR": "ema-20-HeartRate",
                "slope-20-HR": "slope-20-HeartRate",
                "ema-60-HR": "ema-60-HR",
                "slope-60-HR": "slope-60-HeartRate",
            }
        )

        df_cozie = df_all.copy()

        # import qualtrics data
        df_qualtrics = pd.read_pickle(
            os.path.join(self.data_dir, "df_info_subjects.pkl.zip"), compression="zip"
        )

        df_qualtrics["userid"] = [
            f"0{x}" if x < 10 else f"{x}"
            for x in df_qualtrics["Subject ID"].astype("int")
        ]
        df_qualtrics["BMI"] = (
            df_qualtrics["Weight"] / (df_qualtrics["Height"] / 100) ** 2
        )

        df_cozie = df_cozie.merge(
            df_qualtrics[["BMI", "Sex", "userid", "Education", "Health", "Age"]],
            on="userid",
        )

        # Attempt to infer better dtypes for object columns
        df_cozie = df_cozie.infer_objects()
        df_cozie = df_cozie.sort_values("userid").copy()

        df_cozie.to_pickle(
            os.path.join(self.data_dir, "df_cozie_env.pkl.zip"), compression="zip"
        )


if __name__ == "__main__":

    config = Configuration()

    self = ImportData(import_user_data=True)

    # available to import ["out_weather", "ibutton", "cozie", "ind_env", "fitbit", "qualtrics"]
    dataset_to_import = [""]

    for dataset in dataset_to_import:

        if dataset == "out_weather":
            self.import_outdoor_weather_data_sg()

        if dataset == "ibutton":
            self.import_ibutton_data()

        if dataset == "cozie":
            self.import_cozie()

        if dataset == "ind_env":
            self.import_environmental_data()

        if dataset == "fitbit":
            self.import_fitbit_api()

        if dataset == "qualtrics":
            self.import_qualtrics_data()

    self.combine_datasets()
