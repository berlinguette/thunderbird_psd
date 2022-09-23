import os
import shutil
from pathlib import Path

import tomli
from data_processing.reporting.reporting import generate_report
from data_processing.saving.io import save_cleaning_report

ROOT = Path("./data_processing/tests")


def test_new_save_report():
    """Test to check if new reports are made correctly"""
    # Arrange: Set up variables
    uid = "20220922-0001"
    signal_stats = generate_report(*list(range(6)))
    destination = ROOT / "io_test_files/report_new.toml"

    # Act: Save report
    save_cleaning_report(uid, signal_stats, destination)

    # Assert: Check if new file has the same dictionary
    with open(destination, "rb") as f:
        saved_report = tomli.load(f)

    assert saved_report[uid] == signal_stats

    # Post: Reset directory so that test can be rerun
    os.remove(destination)


def test_modify_saved_report():
    """Test to check if existing files are modified correctly"""
    # Arrage: get original report
    original_path = ROOT / "io_test_files/report_stock.toml"
    modified_path = ROOT / "io_test_files/report_modified.toml"

    shutil.copyfile(original_path, modified_path)  # Makes a dupe

    with open(original_path, "rb") as f:
        original_report = tomli.load(f)

    uid = list(original_report.keys())[0]
    signal_stats = generate_report(*list(range(6)))

    # Act: Modify with new values
    save_cleaning_report(uid, signal_stats, modified_path)

    # Assert:
    with open(modified_path, "rb") as f:
        modified_report = tomli.load(f)

    # Check if keys are the same
    assert modified_report.keys() == original_report.keys()

    # Check if stats was modified
    assert signal_stats == modified_report[uid]

    # Post: Reset directory so test can be rerun
    os.remove(modified_path)
