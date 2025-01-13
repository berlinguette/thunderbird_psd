from pathlib import Path
from unittest.mock import PropertyMock

import data_processing.loading.spectrum_unfolding as spec_unfold
import numpy as np
import pytest
from data_processing.loading.spectrum_unfolding import (
    _infer_format,
    _load_R_from_fwf,
    _load_R_from_npy,
    load_neutron_response_matrix,
)
from pyfakefs.fake_filesystem import FakeFilesystem

test_dir = "test_dir"

class TestLoadNeutronResponseMatrix:
    def test_fwf(self, fs: FakeFilesystem):
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
        
        R = load_neutron_response_matrix(Path(test_dir), format="fwf")
        assert R.shape == (1400, 2)
        assert all([x == y for x, y in zip(list(R.midpoints[1]), [1, 2])])
    
    def test_npy(self, fs: FakeFilesystem):
        contents1 = np.random.rand(10)
        contents2 = np.random.rand(10)
        fs.create_dir(test_dir)
        np.save("test_dir/neutron_1.000_MeV.csv.npy", contents1)
        np.save("test_dir/neutron_2.000_MeV.csv.npy", contents2)
        
        R = load_neutron_response_matrix(Path(test_dir), format="npy")
        assert R.shape == (1400, 2)
        assert all([x == y for x, y in zip(list(R.midpoints[1]), [1, 2])])
    
    def test_fwf_inferred(self, fs: FakeFilesystem):
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
        assert R.shape == (1400, 2)
        assert all([x == y for x, y in zip(list(R.midpoints[1]), [1, 2])])
    
    def test_npy_inferred(self, fs: FakeFilesystem):
        contents1 = np.random.rand(10)
        contents2 = np.random.rand(10)
        fs.create_dir(test_dir)
        np.save("test_dir/neutron_1.000_MeV.csv.npy", contents1)
        np.save("test_dir/neutron_2.000_MeV.csv.npy", contents2)
        
        R = load_neutron_response_matrix(Path(test_dir))
        assert R.shape == (1400, 2)
        assert all([x == y for x, y in zip(list(R.midpoints[1]), [1, 2])])
    
    def test_no_files(self, fs: FakeFilesystem):
        fs.create_dir(test_dir)
        with pytest.raises(ValueError) as excinfo:
            load_neutron_response_matrix(Path(test_dir), format="fwf")
        assert "no data files" in str(excinfo.value).lower()
        with pytest.raises(ValueError) as excinfo:
            load_neutron_response_matrix(Path(test_dir), format="npy")
        assert "no data files" in str(excinfo.value).lower()
    
    def test_cannot_infer(self, fs: FakeFilesystem):
        fs.create_file("test_dir/test1.gif")
        fs.create_file("test_dir/test2.gif")
        with pytest.raises(ValueError) as excinfo:
            load_neutron_response_matrix(Path(test_dir))
        assert "cannot" in str(excinfo.value).lower()


class TestLoadRFromFWF:
    def test_good_path(self, fs: FakeFilesystem):
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
        # R = load_neutron_response_matrix(Path(test_dir))
        files = [file for file in Path(test_dir).iterdir() if file.is_file()]
        R = _load_R_from_fwf(files)
        assert R.shape == (1400, 2)
        assert all([x == y for x, y in zip(list(R.midpoints[1]), [1, 2])])
        E_sums = R.counts.sum(axis=0)
        assert all(E_sums == (9 + 21 + 37 + 44 + 64))

    def test_no_files(self, fs: FakeFilesystem):
        fs.create_dir("test_dir")
        with pytest.raises(ValueError) as excinfo:
            # load_neutron_response_matrix(Path(test_dir))
            files = [file for file in Path(test_dir).iterdir() if file.is_file()]
            _load_R_from_fwf(files)
        assert "data files" in str(excinfo.value)

    def test_bad_index_type(self, fs: FakeFilesystem, mocker):
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
            files = [file for file in Path(test_dir).iterdir() if file.is_file()]
            _load_R_from_fwf(files)
            # load_neutron_response_matrix(Path(test_dir))


class TestLoadRFromNPY:
    def test_good_path(self, fs: FakeFilesystem):
        contents1 = np.random.rand(10)
        contents2 = np.random.rand(10)
        fs.create_dir(test_dir)
        np.save("test_dir/neutron_1.000_MeV.csv.npy", contents1)
        np.save("test_dir/neutron_2.000_MeV.csv.npy", contents2)
        
        files = [file for file in Path(test_dir).iterdir() if file.is_file()]
        R = _load_R_from_npy(files)
        assert R.shape == (1400, 2)
        assert all([x == y for x, y in zip(list(R.midpoints[1]), [1, 2])])
        E_sums = R.counts.sum(axis=0)
        assert all(E_sums == 10)
    
    def test_no_files(self, fs: FakeFilesystem):
        fs.create_dir(test_dir)
        with pytest.raises(ValueError) as excinfo:
            # load_neutron_response_matrix(Path(test_dir))
            files = [file for file in Path(test_dir).iterdir() if file.is_file()]
            _load_R_from_npy(files)
        assert "data files" in str(excinfo.value)


class TestGuessFormat:
    def test_infer_fwf(self, fs: FakeFilesystem):
        fs.create_file("test_dir/output_1.txt")
        fs.create_file("test_dir/output_2.txt")
        files = [file for file in Path(test_dir).iterdir() if file.is_file()]
        guess = _infer_format(files)
        assert guess == "fwf"
    
    def test_infer_npy(self, fs: FakeFilesystem):
        fs.create_file("test_dir/neutron_1.000_MeV.csv.npy")
        fs.create_file("test_dir/neutron_2.000_MeV.csv.npy")
        files = [file for file in Path(test_dir).iterdir() if file.is_file()]
        guess = _infer_format(files)
        assert guess == "npy"
    
    def test_cannot_infer(self, fs: FakeFilesystem):
        fs.create_file("test_dir/test1.gif")
        fs.create_file("test_dir/test2.gif")
        files = [file for file in Path(test_dir).iterdir() if file.is_file()]
        with pytest.raises(ValueError) as excinfo:
            _infer_format(files)
        assert "cannot" in str(excinfo.value).lower()
