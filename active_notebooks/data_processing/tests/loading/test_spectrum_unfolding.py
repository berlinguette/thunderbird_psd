from pathlib import Path
from unittest.mock import PropertyMock

import data_processing.loading.spectrum_unfolding as spec_unfold
import pytest
from data_processing.loading.spectrum_unfolding import load_neutron_response_matrix
from pyfakefs.fake_filesystem import FakeFilesystem

test_dir = "test_dir"


def test_good_path(fs: FakeFilesystem):
    contents1 = """
 source_e (MeV)            NPS   det_pulse (MeVee)       det_cell     corr_count            PSD       time (s)  cells_history


              1              9           0.0546242              1             no       0.290346    9.57902e-09          4  1 
              1             21            0.099207              1             no       0.287368    1.10722e-08          4  1 
              1             37           0.0627974              1             no       0.341804    9.62823e-09          4  1 
              1             44           0.0310387              1             no       0.243887    1.01032e-08          4  1 
              1             64           0.0647629              1             no       0.341947    9.20444e-09          4  1 
"""
    contents2 = """
 source_e (MeV)            NPS   det_pulse (MeVee)       det_cell     corr_count            PSD       time (s)  cells_history


              2              9            0.124175              1             no       0.327161    6.83285e-09          4  1 
              2             21            0.320092              1             no       0.389883      8.599e-09          4  1 
              2             37            0.221753              1             no       0.387224    7.06528e-09          4  1 
              2             44           0.0291878              1             no       0.194259    7.40726e-09          4  1 
              2             64            0.217934              1             no       0.388198      6.594e-09          4  1 
"""
    fs.create_file("test_dir/output_1.txt", contents=contents1)
    fs.create_file("test_dir/output_2.txt", contents=contents2)
    R = load_neutron_response_matrix(Path(test_dir))
    assert R.counts.shape == (1400, 2)
    assert all([x == y for x, y in zip(list(R.y_midpoints), [1, 2])])
    E_sums = R.counts.sum(axis=0)
    assert all(E_sums == (9 + 21 + 37 + 44 + 64))


def test_no_files(fs: FakeFilesystem):
    fs.create_dir("test_dir")
    with pytest.raises(ValueError) as excinfo:
        load_neutron_response_matrix(Path(test_dir))
    assert "data files" in str(excinfo.value)


def test_bad_index_type(fs: FakeFilesystem, mocker):
    contents1 = """
 source_e (MeV)            NPS   det_pulse (MeVee)       det_cell     corr_count            PSD       time (s)  cells_history


              1              9           0.0546242              1             no       0.290346    9.57902e-09          4  1 
              1             21            0.099207              1             no       0.287368    1.10722e-08          4  1 
              1             37           0.0627974              1             no       0.341804    9.62823e-09          4  1 
              1             44           0.0310387              1             no       0.243887    1.01032e-08          4  1 
              1             64           0.0647629              1             no       0.341947    9.20444e-09          4  1 
"""
    contents2 = """
 source_e (MeV)            NPS   det_pulse (MeVee)       det_cell     corr_count            PSD       time (s)  cells_history


              2              9            0.124175              1             no       0.327161    6.83285e-09          4  1 
              2             21            0.320092              1             no       0.389883      8.599e-09          4  1 
              2             37            0.221753              1             no       0.387224    7.06528e-09          4  1 
              2             44           0.0291878              1             no       0.194259    7.40726e-09          4  1 
              2             64            0.217934              1             no       0.388198      6.594e-09          4  1 
"""
    fs.create_file("test_dir/output_1.txt", contents=contents1)
    fs.create_file("test_dir/output_2.txt", contents=contents2)
    mock_index = mocker.patch.object(
        spec_unfold.pd.DataFrame, "index", new_callable=PropertyMock
    )
    mock_index.return_value = None
    with pytest.raises(RuntimeError):
        load_neutron_response_matrix(Path(test_dir))
