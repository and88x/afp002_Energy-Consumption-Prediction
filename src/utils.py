from typing import Tuple, Type
import numpy as np


def split_dataset(df, split_date: str) -> Tuple:
    """Docstring"""
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()
    return (df_train, df_test)


def create_features(df) -> None:
    """Docstring"""
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["hour"] = df.index.hour
    df["weekday"] = df.index.day_name()
    df["weekofyear"] = df.index.isocalendar().week.astype("int64")
    df["dayofmonth"] = df.index.day
    # df['date'] = df.index.date

    circular_feature(df, feature="month")
    circular_feature(df, feature="dayofweek")
    circular_feature(df, feature="dayofyear")
    circular_feature(df, feature="quarter")
    circular_feature(df, feature="hour")
    circular_feature(df, feature="weekofyear")
    circular_feature(df, feature="dayofmonth")


def circular_feature(df, feature: str) -> None:
    """Docstring"""
    maximum = max(df[feature])
    df[f"{feature}_sin"] = np.sin(2 * np.pi * df[feature] / maximum)
    df[f"{feature}_cos"] = np.cos(2 * np.pi * df[feature] / maximum)
