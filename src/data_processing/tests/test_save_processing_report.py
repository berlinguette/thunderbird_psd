import os
from pathlib import Path

import pandas as pd
from data_processing.saving.io import save_results

ROOT = Path("./data_processing/tests")

COLUMN_LABELS = ["counts", "counts / sec", "FOM"]


def test_save_new_results():
    # Arrange: Prepare a new dataframe to be saved
    destination = ROOT / "io_test_files/report_new.csv"

    df = pd.DataFrame([(420, 512, 690)], columns=COLUMN_LABELS, index=["20220906-0002"])

    # Act: Save results
    save_results(df, destination)

    # Assert: Check if df is correctly saved
    new_df = pd.read_csv(destination, index_col=0)
    assert new_df.equals(df)

    # Post: Reset directory so that test can be rerun
    os.remove(destination)


def test_modify_results():
    # Arrange: Prepare dataframe to be modified
    unchanged_row_vals = 122.0, 155.0, 184.0
    unchanged_row_label = "20220906-0115"

    df_original = pd.DataFrame(
        [(420.0, 512.0, 690.0), (unchanged_row_vals)],
        columns=COLUMN_LABELS,
        index=["20220906-0001", unchanged_row_label],
    )

    # Save dataframe
    destination = ROOT / "io_test_files/report_stock.csv"
    save_results(df_original, destination)

    # Create new dataframe that modifies existing entry
    # and also creates a new entry
    df_new = pd.DataFrame(
        [(112.0, 612.0, 332.0), (176.0, 591.0, 132.0)],
        columns=COLUMN_LABELS,
        index=["20220906-0001", "20220906-0420"],
    )

    # Act: Modify old dataframe and prepare for comparison
    save_results(df_new, destination)

    df_modified = pd.read_csv(destination, index_col=0)
    df_unchanged_row = pd.DataFrame(
        [(unchanged_row_vals)], columns=COLUMN_LABELS, index=[unchanged_row_label]
    )

    # Add unchanged row to new dataframe
    df_comparator = pd.concat([df_new, df_unchanged_row])
    df_comparator = df_comparator.sort_index()

    # Assert: Check if df is correctly modified with new values
    assert df_modified.equals(df_comparator)

    # Post: Reset directory so that test can be rerun
    os.remove(destination)
