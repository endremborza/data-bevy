from dataclasses import dataclass
from datetime import datetime

import datazimmer as dz
import pandas as pd
from colassigner import ColAssigner, get_all_cols


class NoStops(Exception):
    pass


@dataclass
class DaySetup:
    work_start: int
    work_end: int
    home_arrive: int
    home_depart: int


class Coordinates(dz.CompositeTypeBase):
    lat = float
    lon = float


class Interval(dz.CompositeTypeBase):
    start = datetime
    end = datetime


class PingFeatures(dz.TableFeaturesBase):
    loc = Coordinates
    datetime = datetime
    device_id = str


class StopFeatures(dz.TableFeaturesBase):
    device_id = str
    destination_label = str
    stay_number = int
    n_events = int
    interval = Interval
    center = Coordinates
    is_home = bool
    is_work = bool
    info = str


class Labeler(ColAssigner):
    def __init__(self, model, day: DaySetup) -> None:
        self.model = model
        self.day = day

    def ts(self, df):
        return df[PingFeatures.datetime].view(int) / 10**9

    def hour(self, df):
        return df[PingFeatures.datetime].dt.hour

    def destination_label(self, df):
        arr = df.loc[:, [PingFeatures.loc.lat, PingFeatures.loc.lon, Labeler.ts]].values
        try:
            return self.model.fit_predict(arr).astype(str)
        except Exception as e:
            assert "No stop events found" in str(e)
            raise NoStops("hopefully")

    def stay_number(self, df):
        return (
            df[Labeler.destination_label] != df[Labeler.destination_label].shift(1)
        ).cumsum()

    def is_worktime(self, df):
        return (df[Labeler.hour] >= self.day.work_start) & (
            df[Labeler.hour] <= self.day.work_end
        )

    def is_hometime(self, df):
        return (df[Labeler.hour] >= self.day.home_arrive) | (
            df[Labeler.hour] <= self.day.home_depart
        )


def proc_device_pings(ping_df, model, day: DaySetup):
    return (
        ping_df.sort_values(PingFeatures.datetime)
        .pipe(Labeler(model, day))
        .pipe(_gb_stop)
    )


def _gb_stop(labeled_df):
    dt_col = PingFeatures.datetime
    return (
        labeled_df.groupby([Labeler.stay_number, Labeler.destination_label])
        .agg(
            **{
                StopFeatures.n_events: pd.NamedAgg(dt_col, "count"),
                StopFeatures.interval.start: pd.NamedAgg(dt_col, "first"),
                StopFeatures.interval.end: pd.NamedAgg(dt_col, "last"),
                StopFeatures.center.lon: pd.NamedAgg(PingFeatures.loc.lon, "mean"),
                StopFeatures.center.lat: pd.NamedAgg(PingFeatures.loc.lat, "mean"),
                "home_rate": pd.NamedAgg(Labeler.is_hometime, "mean"),
                "work_rate": pd.NamedAgg(Labeler.is_worktime, "mean"),
            }
        )
        .reset_index()
        .assign(
            **{
                "dur": lambda df: (
                    df[StopFeatures.interval.end] - df[StopFeatures.interval.start]
                ).dt.total_seconds()
                * (df[StopFeatures.destination_label] != "-1"),
                StopFeatures.is_work: lambda df: _is_maxw(df, "work_rate"),
                StopFeatures.is_home: lambda df: _is_maxw(df, "home_rate"),
                StopFeatures.info: "N/A",
                StopFeatures.device_id: "0",
            }
        )
        .loc[:, get_all_cols(StopFeatures)]
    )


def _is_maxw(df, rate_col):
    gb_cols = ["_week", StopFeatures.destination_label]
    wdf = df.assign(
        _week=df[StopFeatures.interval.start].dt.isocalendar().week,
        target=df["dur"] * df[rate_col],
    )
    wsums = wdf.groupby(gb_cols)["target"].sum()
    wmaxs = wsums.groupby("_week").transform("max")
    return (wsums == wmaxs).reindex(wdf[gb_cols]).values
