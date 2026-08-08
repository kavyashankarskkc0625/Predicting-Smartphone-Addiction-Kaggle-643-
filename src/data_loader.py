"""
Data Loading Module

Handles loading of training, testing, and sample submission data.
"""

import pandas as pd

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    SAMPLE_SUBMISSION_FILE,
)


def load_data():
    """
    Load training, testing, and sample submission datasets.

    Returns
    -------
    tuple
        train_df, test_df, sample_submission
    """

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)

    return train_df, test_df, sample_submission