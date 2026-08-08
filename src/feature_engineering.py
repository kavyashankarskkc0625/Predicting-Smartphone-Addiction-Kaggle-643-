"""
Feature Engineering Module

This module contains reusable feature engineering
functions for the Smartphone Addiction Kaggle competition.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features.
    """

    df = df.copy()

    # ======================================================
    # Feature Groups
    # ======================================================

    # Time Features

    # Ratio Features

    # Behaviour Features

    # Interaction Features

    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features.
    """

    df = df.copy()

    df["total_entertainment_hours"] = (
        df["social_media_hours"] +
        df["gaming_hours"]
    )

    df["weekend_difference"] = (
        df["weekend_screen_time"] -
        df["daily_screen_time_hours"]
    )


    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    """

    df = df.copy()

    df = create_time_features(df)
    df = create_ratio_features(df)
    df = create_behavior_features(df)

    return df

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create relative usage and behavioral ratio features.
    """

    df = df.copy()

    screen_time = df["daily_screen_time_hours"]

    df["entertainment_ratio"] = (
        df["total_entertainment_hours"] / screen_time
    )

    df["social_media_ratio"] = (
        df["social_media_hours"] / screen_time
    )

    df["gaming_ratio"] = (
        df["gaming_hours"] / screen_time
    )

    df["work_study_ratio"] = (
        df["work_study_hours"] / screen_time
    )

    return df

def create_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral relationship features.
    """

    df = df.copy()

    work_study = df["work_study_hours"].replace(0, float("nan"))

    df["leisure_to_work_ratio"] = (
        df["total_entertainment_hours"] / work_study
    )

    df["screen_time_waking_ratio"] = (
        df["daily_screen_time_hours"] /
        (24 - df["sleep_hours"])
    )

    df["non_screen_time"] = (
        24
        - df["sleep_hours"]
        - df["daily_screen_time_hours"]
    )

    return df