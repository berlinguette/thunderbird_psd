from math import pow

import numpy as np
import pytest
from data_processing.processing.spectrum_unfolding import (
    NDHistogram,
    _are_r_dimensions_close_enough,
    _is_n_compatible,
    _is_phi_compatible,
    _nan_divide,
    clean_data,
    cut_low_l,
    next_phi,
    r_dot,
    stopping_criteria,
    unfold_spectrum,
    weight_factor,
)


@pytest.fixture
def _array_generator():
    def _generate_array(start: int, stop: int | None = None) -> np.ndarray:
        if stop is None:
            return np.array(range(start))
        else:
            return np.array(range(start, stop))

    return _generate_array


@pytest.fixture
def _array_generator_2d():
    def _generate_array_2d(
        start: tuple[int, int], stop: tuple[int, int] | None = None
    ) -> np.ndarray:
        start_x, start_y = start
        if stop is None:
            x_range = range(start_x)
            y_range = range(start_y)
        else:
            stop_x, stop_y = stop
            x_range = range(start_x, stop_x)
            y_range = range(start_y, stop_y)
        data = [[x + y for y in y_range] for x in x_range]
        return np.array(data)

    return _generate_array_2d


class TestNDHistogram:
    def test_inits(self):
        counts = np.ones((4, 3))
        midpoints = [np.ones(4), np.ones(3)]
        histogram = NDHistogram(counts, midpoints)
        assert (counts == histogram.counts).all()
        assert all(
            [
                (expected == actual).all()
                for expected, actual in zip(midpoints, histogram.midpoints)
            ]
        )

    @pytest.mark.parametrize(
        "midpoints,exc_text_list",
        [
            ([np.ones(3)], "dimensions"),
            ([np.ones(4), np.ones(3), np.ones(5)], ["dimensions"]),
            ([np.ones((4, 1)), np.ones(3)], ["1-dimensional"]),
            ([np.ones((1, 4)), np.ones(3)], ["1-dimensional"]),
            ([np.ones(4), np.ones((3, 1))], ["1-dimensional"]),
            ([np.ones(4), np.ones((1, 3))], ["1-dimensional"]),
            ([np.ones(3), np.ones(3)], ["size", "axis 0"]),
            ([np.ones(4), np.ones(4)], ["size", "axis 1"]),
        ],
    )
    def test_midpoints_bad(self, midpoints, exc_text_list):
        counts = np.ones((4, 3))
        with pytest.raises(ValueError) as excinfo:
            NDHistogram(counts, midpoints)
        for exc_text in exc_text_list:
            assert exc_text in str(excinfo.value)

    def test_midpoints_wrong_dimensions(self):
        counts = np.ones((4, 3))
        midpoints = [np.ones(3)]
        with pytest.raises(ValueError) as excinfo:
            NDHistogram(counts, midpoints)
        assert "dimensions" in str(excinfo.value)

    def test_midpoints_not_1D(self):
        counts = np.ones((4, 3))
        midpoints = [np.ones((4, 1)), np.ones(3)]
        with pytest.raises(ValueError) as excinfo:
            NDHistogram(counts, midpoints)
        assert "1-dimensional" in str(excinfo.value)

    def test_midpoints_wrong_length(self):
        counts = np.ones((4, 3))
        midpoints = [np.ones(4), np.ones(4)]
        with pytest.raises(ValueError) as excinfo:
            NDHistogram(counts, midpoints)
        assert "size" in str(excinfo.value)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            np.array([[6.0, 7.0], [8.0, 9.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[6.0, 7.0 / 2.0], [8.0 / 3.0, 9.0 / 4.0]]),
        ),
        (
            np.array([[3.0, 3.5], [11.7, 19.5]]),
            np.array([[0.5, 1.0], [1.5, 2.0]]),
            np.array([[3.0 / 0.5, 3.5], [11.7 / 1.5, 19.5 / 2.0]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 9.0]]),
            np.array([[1.0, 2.0], [3.0, 0.0]]),
            np.array([[6.0, 7.0 / 2.0], [8.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 0.0]]),
            np.array([[1.0, 2.0], [3.0, 0.0]]),
            np.array([[6.0, 7.0 / 2.0], [8.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[0.0, 7.0], [8.0, 9.0]]),
            np.array([[0.0, 2.0], [3.0, 0.0]]),
            np.array([[np.nan, 7.0 / 2.0], [8.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 9.0]]),
            np.array([[3.0, 4.0]]),
            np.array([[6.0 / 3.0, 7.0 / 4.0], [8.0 / 3.0, 9.0 / 4.0]]),
        ),
        (
            np.array([[3.0, 3.5], [11.7, 19.5]]),
            np.array([[1.5, 2.0]]),
            np.array([[3.0 / 1.5, 3.5 / 2.0], [11.7 / 1.5, 19.5 / 2.0]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 9.0]]),
            np.array([[3.0, 0.0]]),
            np.array([[6.0 / 3.0, np.nan], [8.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 0.0]]),
            np.array([[3.0, 0.0]]),
            np.array([[6.0 / 3.0, np.nan], [8.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[0.0, 7.0], [8.0, 9.0]]),
            np.array([[0.0, 2.0]]),
            np.array([[np.nan, 7.0 / 2.0], [np.nan, 9.0 / 2.0]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 9.0]]),
            np.array([[2.0], [4.0]]),
            np.array([[6.0 / 2.0, 7.0 / 2.0], [8.0 / 4.0, 9.0 / 4.0]]),
        ),
        (
            np.array([[3.0, 3.5], [11.7, 19.5]]),
            np.array([[0.5], [1.5]]),
            np.array([[3.0 / 0.5, 3.5 / 0.5], [11.7 / 1.5, 19.5 / 1.5]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 9.0]]),
            np.array([[2.0], [0.0]]),
            np.array([[6.0 / 2.0, 7.0 / 2.0], [np.nan, np.nan]]),
        ),
        (
            np.array([[6.0, 7.0], [8.0, 0.0]]),
            np.array([[2.0], [0.0]]),
            np.array([[6.0 / 2.0, 7.0 / 2.0], [np.nan, np.nan]]),
        ),
        (
            np.array([[0.0, 7.0], [8.0, 9.0]]),
            np.array([[0.0], [0.0]]),
            np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        ),
        (
            np.array([[6.0, 7.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[6.0, 7.0 / 2.0], [6.0 / 3.0, 7.0 / 4.0]]),
        ),
        (
            np.array([[11.7, 19.5]]),
            np.array([[0.5, 1.0], [1.5, 2.0]]),
            np.array([[11.7 / 0.5, 19.5], [11.7 / 1.5, 19.5 / 2.0]]),
        ),
        (
            np.array([[8.0, 9.0]]),
            np.array([[1.0, 2.0], [3.0, 0.0]]),
            np.array([[8.0, 9.0 / 2.0], [8.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[8.0, 0.0]]),
            np.array([[1.0, 2.0], [3.0, 0.0]]),
            np.array([[8.0, 0.0], [8.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[0.0, 7.0]]),
            np.array([[0.0, 2.0], [3.0, 0.0]]),
            np.array([[np.nan, 7.0 / 2.0], [0.0, np.nan]]),
        ),
        (
            np.array([[7.0], [9.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[7.0, 7.0 / 2.0], [9.0 / 3.0, 9.0 / 4.0]]),
        ),
        (
            np.array([[11.7, 19.5]]),
            np.array([[0.5, 1.0], [1.5, 2.0]]),
            np.array([[11.7 / 0.5, 19.5], [11.7 / 1.5, 19.5 / 2.0]]),
        ),
        (
            np.array([[7.0], [9.0]]),
            np.array([[1.0, 2.0], [3.0, 0.0]]),
            np.array([[7.0, 7.0 / 2.0], [9.0 / 3.0, np.nan]]),
        ),
        (
            np.array([[7.0], [0.0]]),
            np.array([[1.0, 2.0], [3.0, 0.0]]),
            np.array([[7.0, 7.0 / 2.0], [0.0, np.nan]]),
        ),
        (
            np.array([[0.0], [9.0]]),
            np.array([[0.0, 2.0], [3.0, 0.0]]),
            np.array([[np.nan, 0.0], [9.0 / 3.0, np.nan]]),
        ),
    ],
)
def test_nan_divide(a: np.ndarray, b: np.ndarray, expected: np.ndarray):
    with np.testing.assert_no_warnings():
        result = _nan_divide(a, b)
    assert result.shape == expected.shape
    is_same = (result == expected) | (np.isnan(result))
    print(result)
    print(is_same)
    assert is_same.all()


class TestIsNCompatible:
    @pytest.mark.parametrize("N_n", [1, 3])
    def test_is_compatible(self, _array_generator, N_n):
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), _array_generator(3)])
        n = NDHistogram(np.ones((4, N_n)), [_array_generator(4), _array_generator(N_n)])
        compatible, reason = _is_n_compatible(r, n)
        assert compatible
        assert reason == ""
        
    @pytest.mark.parametrize("N_n", [1, 3])
    def test_mids0_close(self, _array_generator, N_n):
        r_mids0 = _array_generator(4)
        n_mids0 = r_mids0 + 1e-9
        assert not (r_mids0 == n_mids0).all()
        assert np.isclose(r_mids0, n_mids0).all()
        
        r = NDHistogram(np.ones((4, 3)), [r_mids0, _array_generator(3)])
        n = NDHistogram(np.ones((4, N_n)), [n_mids0, _array_generator(N_n)])
        compatible, reason = _is_n_compatible(r, n)
        assert compatible
        assert reason == ""
        assert (r.midpoints[0] == r.midpoints[0]).all()
        
    def test_mids1_close(self, _array_generator):
        r_mids1 = _array_generator(3)
        n_mids1 = r_mids1 + 1e-9
        assert not (r_mids1 == n_mids1).all()
        assert np.isclose(r_mids1, n_mids1).all()
        
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), r_mids1])
        n = NDHistogram(np.ones((4, 3)), [_array_generator(4), n_mids1])
        compatible, reason = _is_n_compatible(r, n)
        assert compatible
        assert reason == ""
        assert (r.midpoints[1] == r.midpoints[1]).all()
        
        n = NDHistogram(np.ones((4, 1)), [_array_generator(4), _array_generator(1)])
        compatible, reason = _is_n_compatible(r, n)
        assert compatible
        assert reason == ""

    @pytest.mark.parametrize(
        "N_m,N_n", [(5, 1), (3, 1), (5, 3), (3, 3), (4, 2), (4, 4)]
    )
    def test_no_shape_match(self, _array_generator, N_m, N_n):
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), _array_generator(3)])
        n = NDHistogram(
            np.ones((N_m, N_n)), [_array_generator(N_m), _array_generator(N_n)]
        )
        compatible, reason = _is_n_compatible(r, n)
        assert not compatible
        assert "Shapes" in reason

    @pytest.mark.parametrize("start,end", [(1, 5), (2, 6)])
    def test_no_axis_0_mids_match(self, _array_generator, start, end):
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), _array_generator(3)])
        n = NDHistogram(
            np.ones((end - start, 3)),
            [_array_generator(start, end), _array_generator(3)],
        )
        compatible, reason = _is_n_compatible(r, n)
        assert not compatible
        assert "Midpoints" in reason
        assert "axis 0" in reason

    @pytest.mark.parametrize("start,end", [(1, 4), (2, 5)])
    def test_no_axis_1_mids_match(self, _array_generator, start, end):
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), _array_generator(3)])
        n = NDHistogram(
            np.ones((4, end - start)),
            [_array_generator(4), _array_generator(start, end)],
        )
        compatible, reason = _is_n_compatible(r, n)
        assert not compatible
        assert "Midpoints" in reason
        assert "axis 1" in reason


class TestIsPhiCompatible:
    def test_is_compatible(self, _array_generator):
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), _array_generator(3)])
        phi = NDHistogram(np.ones((1, 3)), [_array_generator(1), _array_generator(3)])
        compatible, reason = _is_phi_compatible(r, phi)
        assert compatible
        assert reason == ""
    
    def test_mids_close(self, _array_generator):
        r_mids1 = _array_generator(3)
        phi_mids1 = r_mids1 + 1e-9
        assert not (r_mids1 == phi_mids1).all()
        assert np.isclose(r_mids1, phi_mids1).all()
        
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), r_mids1])
        phi = NDHistogram(np.ones((1, 3)), [_array_generator(1), phi_mids1])
        compatible, reason = _is_phi_compatible(r, phi)
        assert compatible
        assert reason == ""
        assert (r.midpoints[1] == r.midpoints[1]).all()

    @pytest.mark.parametrize("N_m,N_n", [(2, 3), (1, 2), (1, 4)])
    def test_no_shape_match(self, _array_generator, N_m, N_n):
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), _array_generator(3)])
        phi = NDHistogram(
            np.ones((N_m, N_n)), [_array_generator(N_m), _array_generator(N_n)]
        )
        compatible, reason = _is_phi_compatible(r, phi)
        assert not compatible
        assert "Shapes" in reason

    @pytest.mark.parametrize("start, end", [(1, 4), (2, 5)])
    def test_no_mids_match(self, _array_generator, start, end):
        r = NDHistogram(np.ones((4, 3)), [_array_generator(4), _array_generator(3)])
        phi = NDHistogram(
            np.ones((1, end - start)),
            [_array_generator(1), _array_generator(start, end)],
        )
        compatible, reason = _is_phi_compatible(r, phi)
        assert not compatible
        assert "Midpoints" in reason
        assert "axis 1" in reason


class TestAreRDimensionsCloseEnough:
    def test_close_enough(self, _array_generator):
        m = 10
        n = 100
        r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        assert _are_r_dimensions_close_enough(r)

    def test_not_close_enough(self, _array_generator):
        m = 10
        n = 101
        r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        assert not _are_r_dimensions_close_enough(r)


class TestRDot:
    @pytest.mark.parametrize(
        "r_array,phi_array,exp_counts",
        [
            (
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[2, 3, 4]]),
                np.array([[2 + 6 + 12], [8 + 15 + 24]]),
            ),
            (
                np.array([[1, np.nan, 3], [4, 5, 6]]),
                np.array([[2, 3, 4]]),
                np.array([[2 + 12], [8 + 15 + 24]]),
            ),
            (
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[2, 3, np.nan]]),
                np.array([[2 + 6], [8 + 15]]),
            ),
            (
                np.array([[1, np.nan, 3], [4, 5, 6]]),
                np.array([[2, 3, np.nan]]),
                np.array([[2], [8 + 15]]),
            ),
        ],
    )
    def test_good_path(
        self,
        _array_generator,
        r_array: np.ndarray,
        phi_array: np.ndarray,
        exp_counts: np.ndarray,
    ):
        _r_mids = [_array_generator(size) for size in r_array.shape]
        _phi_mids = [_array_generator(size) for size in phi_array.shape]

        r = NDHistogram(r_array, _r_mids)
        phi = NDHistogram(phi_array, _phi_mids)

        rdot = r_dot(r, phi)

        m = r_array.shape[0]
        assert rdot.shape == (m, 1)
        assert (_r_mids[0] == rdot.midpoints[0]).all()
        assert (rdot.counts == exp_counts).all()

    @pytest.mark.parametrize(
        "phi_size,phi_mids_args,expected_text",
        [
            (3, (0, 3), ["Phi", "Shapes"]),
            (1, (0, 1), ["Phi", "Shapes"]),
            (2, (1, 3), ["Phi", "Midpoints"]),
            (2, (2, 4), ["Phi", "Midpoints"]),
        ],
    )
    def test_incompatiblities(
        self,
        _array_generator,
        phi_size,
        phi_mids_args,
        expected_text,
    ):
        r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        phi = NDHistogram(
            np.ones((1, phi_size)),
            [_array_generator(1), _array_generator(*phi_mids_args)],
        )
        with pytest.raises(ValueError) as excinfo:
            r_dot(r, phi)
        print(excinfo.value)
        for expected in expected_text:
            assert expected in str(excinfo.value)


class TestCutLowL:
    def test_no_cut(self, _array_generator, _array_generator_2d):
        r = NDHistogram(
            _array_generator_2d((4, 3)), [_array_generator(4), _array_generator(3)]
        )
        n = NDHistogram(
            _array_generator_2d((4, 1)), [_array_generator(4), _array_generator(1)]
        )
        new_r, new_n, new_sigma = cut_low_l(r, n)
        assert new_sigma is None
        assert (new_r.counts == r.counts).all()
        assert (new_n.counts == n.counts).all()
        assert all(
            [
                (new_mids == mids).all()
                for new_mids, mids in zip(new_r.midpoints, r.midpoints)
            ]
        )
        assert all(
            [
                (new_mids == mids).all()
                for new_mids, mids in zip(new_n.midpoints, n.midpoints)
            ]
        )

    @pytest.mark.parametrize(
        "r_n,n_n,L_cut,exp_m",
        [
            (3, 1, 0.9, 4),
            (3, 1, 1.0, 4),
            (3, 1, 1.1, 3),
            (3, 1, 2.1, 2),
            (3, 1, 3.1, 1),
            (3, 1, 4.0, 1),
            (3, 1, 4.1, 0),
            (3, 3, 0.9, 4),
            (3, 3, 1.0, 4),
            (3, 3, 1.1, 3),
            (3, 3, 2.1, 2),
            (3, 3, 3.1, 1),
            (3, 3, 4.0, 1),
            (3, 3, 4.1, 0),
        ],
    )
    def test_good_path(
        self,
        _array_generator,
        r_n: int,
        n_n: int,
        L_cut: float,
        exp_m: int,
    ):
        r_L_mids = np.array([1, 2, 3, 4])
        m = r_L_mids.shape[0]
        r = NDHistogram(np.ones((m, r_n)), [r_L_mids, _array_generator(r_n)])
        n = NDHistogram(np.ones((m, n_n)), [r_L_mids, _array_generator(n_n)])
        new_r, new_n, new_sigma = cut_low_l(r, n, L_cut=L_cut)
        new_r_L_mids, new_r_E_mids = new_r.midpoints
        new_n_L_mids, new_n_E_mids = new_n.midpoints

        assert new_sigma is None
        assert new_r.shape == (exp_m, r_n)
        assert new_r_L_mids.shape == (exp_m,)
        assert new_r_E_mids.shape == (r_n,)
        assert (new_r_E_mids == _array_generator(r_n)).all()
        assert new_n.shape == (exp_m, n_n)
        assert new_n_L_mids.shape == (exp_m,)
        assert new_n_E_mids.shape == (n_n,)
        assert (new_n_E_mids == _array_generator(n_n)).all()

    @pytest.mark.parametrize(
        "r_n,n_n,L_cut,exp_m",
        [
            (3, 1, 0.9, 4),
            (3, 1, 1.0, 4),
            (3, 1, 1.1, 3),
            (3, 1, 2.1, 2),
            (3, 1, 3.1, 1),
            (3, 1, 4.0, 1),
            (3, 1, 4.1, 0),
            (3, 3, 0.9, 4),
            (3, 3, 1.0, 4),
            (3, 3, 1.1, 3),
            (3, 3, 2.1, 2),
            (3, 3, 3.1, 1),
            (3, 3, 4.0, 1),
            (3, 3, 4.1, 0),
        ],
    )
    def test_sigma(
        self, _array_generator, r_n: int, n_n: int, L_cut: float, exp_m: int
    ):
        r_L_mids = np.array([1, 2, 3, 4])
        m = r_L_mids.shape[0]
        r = NDHistogram(np.ones((m, r_n)), [r_L_mids, _array_generator(r_n)])
        n = NDHistogram(np.ones((m, n_n)), [r_L_mids, _array_generator(n_n)])
        sigma = NDHistogram(np.ones((m, n_n)), [r_L_mids, _array_generator(n_n)])
        new_r, new_n, new_sigma = cut_low_l(r, n, sigma=sigma, L_cut=L_cut)
        new_r_L_mids, new_r_E_mids = new_r.midpoints
        new_n_L_mids, new_n_E_mids = new_n.midpoints

        assert new_sigma is not None
        new_sig_L_mids, new_sig_E_mids = new_sigma.midpoints

        assert new_r.shape == (exp_m, r_n)
        assert new_r_L_mids.shape == (exp_m,)
        assert new_r_E_mids.shape == (r_n,)
        assert (new_r_E_mids == _array_generator(r_n)).all()
        assert new_n.shape == (exp_m, n_n)
        assert new_n_L_mids.shape == (exp_m,)
        assert new_n_E_mids.shape == (n_n,)
        assert (new_n_E_mids == _array_generator(n_n)).all()
        assert new_sigma.shape == (exp_m, n_n)
        assert new_sig_L_mids.shape == (exp_m,)
        assert new_sig_E_mids.shape == (n_n,)
        assert (new_sig_E_mids == _array_generator(n_n)).all()

    @pytest.mark.parametrize(
        "n_size,n_mids_args,sigma_args,expected_text",
        [
            (4, (0, 4), None, ["N", "Shapes"]),
            (2, (0, 2), None, ["N", "Shapes"]),
            (3, (1, 4), None, ["N", "Midpoints"]),
            (3, (2, 5), None, ["N", "Midpoints"]),
            (3, (0, 3), (4, (0, 4)), ["Sigma", "Shapes"]),
            (3, (0, 3), (2, (0, 2)), ["Sigma", "Shapes"]),
            (3, (0, 3), (3, (1, 4)), ["Sigma", "Midpoints"]),
            (3, (0, 3), (3, (2, 5)), ["Sigma", "Midpoints"]),
        ],
    )
    def test_incompatiblities(
        self,
        _array_generator,
        n_size,
        n_mids_args,
        sigma_args,
        expected_text,
    ):
        r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        n = NDHistogram(
            np.ones((n_size, 1)), [_array_generator(*n_mids_args), _array_generator(1)]
        )
        if sigma_args is None:
            sigma = None
        else:
            sigma_size, sigma_mids_args = sigma_args
            sigma = NDHistogram(
                np.full((sigma_size, 1), 0.1),
                [_array_generator(*sigma_mids_args), _array_generator(1)],
            )
        with pytest.raises(ValueError) as excinfo:
            cut_low_l(r, n, sigma=sigma, L_cut=0.5)  # type: ignore
        print(excinfo.value)
        for expected in expected_text:
            assert expected in str(excinfo.value)


# class TestCleanData:
#     @pytest.mark.parametrize(
#         "r_array,n_array,r_ex,n_ex,phi_ex",
#         [
#             (
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[1], [2]],
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[1], [2]],
#                 [[0, 1, 2]],
#             ),
#             (
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[0], [2]],
#                 [[4, 5, 6]],
#                 [[2]],
#                 [[0, 1, 2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[1], [2]],
#                 [[1, 3], [4, 6]],
#                 [[1], [2]],
#                 [[0, 2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[0], [2]],
#                 [[4, 6]],
#                 [[2]],
#                 [[0, 2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6], [0, 0, 0]],
#                 [[1], [2], [3]],
#                 [[1, 3], [4, 6]],
#                 [[1], [2]],
#                 [[0, 2]],
#             ),
#             (
#                 [[1, 2, 3], [4, 0, 6]],
#                 [[0], [2]],
#                 [[4, 6]],
#                 [[2]],
#                 [[0, 2]],
#             ),
#             (
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[1, 2, 3], [2, 4, 6]],
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[1, 2, 3], [2, 4, 6]],
#                 [[0, 1, 2]],
#             ),
#             (
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[0, 0, 0], [2, 4, 6]],
#                 [[4, 5, 6]],
#                 [[2, 4, 6]],
#                 [[0, 1, 2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[1, 2, 3], [2, 4, 6]],
#                 [[1, 3], [4, 6]],
#                 [[1, 3], [2, 6]],
#                 [[0, 2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[0, 0, 0], [2, 4, 6]],
#                 [[4, 6]],
#                 [[2, 6]],
#                 [[0, 2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6], [0, 0, 0]],
#                 [[1, 2, 3], [2, 4, 6], [3, 6, 9]],
#                 [[1, 3], [4, 6]],
#                 [[1, 3], [2, 6]],
#                 [[0, 2]],
#             ),
#             (
#                 [[1, 2, 3], [4, 0, 6]],
#                 [[0, 0, 0], [2, 4, 6]],
#                 [[4, 6]],
#                 [[2, 6]],
#                 [[0, 2]],
#             ),
#         ],
#     )
#     def test_good_path(
#         self,
#         _array_generator,
#         r_array: list[list[float]],
#         n_array: list[list[float]],
#         r_ex: list[list[float]],
#         n_ex: list[list[float]],
#         phi_ex: list[list[float]],
#     ):
#         r_counts = np.array(r_array)
#         r_size0, r_size1, *_ = r_counts.shape
#         n_counts = np.array(n_array)
#         n_size0, n_size1, *_ = n_counts.shape
#         phi_counts = _array_generator(r_size1).reshape(1, -1)

#         r = NDHistogram(
#             r_counts, [_array_generator(r_size0), _array_generator(r_size1)]
#         )
#         n = NDHistogram(
#             n_counts, [_array_generator(n_size0), _array_generator(n_size1)]
#         )
#         phi = NDHistogram(phi_counts, [_array_generator(1), _array_generator(r_size1)])

#         r_clean, n_clean, phi_clean, maybe_sigma = clean_data(r, n, phi)
#         assert maybe_sigma is None

#         r_expected = np.array(r_ex)
#         n_expected = np.array(n_ex)
#         phi_expected = np.array(phi_ex)

#         for actual, expected in zip(
#             [r_clean, n_clean, phi_clean], [r_expected, n_expected, phi_expected]
#         ):
#             assert actual.shape == expected.shape
#             assert (actual.counts == expected).all()

#     @pytest.mark.parametrize(
#         "r_array,n_array,sigma_array,sigma_ex",
#         [
#             ([[1, 2, 3], [4, 5, 6]], [[1], [2]], [[0.1], [0.2]], [[0.1], [0.2]]),
#             (
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[0], [2]],
#                 [[0.1], [0.2]],
#                 [[0.2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[1], [2]],
#                 [[0.1], [0.2]],
#                 [[0.1], [0.2]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[0], [2]],
#                 [[0.1], [0.2]],
#                 [[0.2]],
#             ),
#             (
#                 [[1, 2, 3], [4, 0, 6]],
#                 [[0], [2]],
#                 [[0.1], [0.2]],
#                 [[0.2]],
#             ),
#             (
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[1, 2, 3], [2, 4, 6]],
#                 [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
#                 [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
#             ),
#             (
#                 [[1, 2, 3], [4, 5, 6]],
#                 [[0, 0, 0], [2, 4, 6]],
#                 [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
#                 [[0.2, 0.4, 0.6]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[1, 2, 3], [2, 4, 6]],
#                 [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
#                 [[0.1, 0.3], [0.2, 0.6]],
#             ),
#             (
#                 [[1, 0, 3], [4, 0, 6]],
#                 [[0, 0, 0], [2, 4, 6]],
#                 [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
#                 [[0.2, 0.6]],
#             ),
#             (
#                 [[1, 2, 3], [4, 0, 6]],
#                 [[0, 0, 0], [2, 4, 6]],
#                 [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
#                 [[0.2, 0.6]],
#             ),
#         ],
#     )
#     def test_sigma(
#         self,
#         _array_generator,
#         r_array: list[list[float]],
#         n_array: list[list[float]],
#         sigma_array: list[list[float]],
#         sigma_ex: list[list[float]],
#     ):
#         r_counts = np.array(r_array)
#         r_size0, r_size1, *_ = r_counts.shape
#         n_counts = np.array(n_array)
#         n_size0, n_size1, *_ = n_counts.shape
#         phi_counts = _array_generator(r_size1).reshape(1, -1)

#         sigma_counts = np.array(sigma_array)
#         sig_size0, sig_size1, *_ = sigma_counts.shape

#         r = NDHistogram(
#             r_counts, [_array_generator(r_size0), _array_generator(r_size1)]
#         )
#         n = NDHistogram(
#             n_counts, [_array_generator(n_size0), _array_generator(n_size1)]
#         )
#         phi = NDHistogram(phi_counts, [_array_generator(1), _array_generator(r_size1)])
#         sigma = NDHistogram(
#             sigma_counts, [_array_generator(sig_size0), _array_generator(sig_size1)]
#         )

#         *_, sigma_clean = clean_data(r, n, phi, sigma=sigma)
#         assert sigma_clean is not None

#         sigma_expected = np.array(sigma_ex)
#         assert sigma_expected.shape == sigma_clean.shape
#         assert (sigma_expected == sigma_clean.counts).all()

#     @pytest.mark.parametrize(
#         "L_cut,r_ex,n_ex,phi_ex,sigma_ex",
#         [
#             (
#                 0.9,
#                 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
#                 [[1], [2], [3]],
#                 [[0, 1, 2, 3]],
#                 [[0.1], [0.2], [0.3]],
#             ),
#             (
#                 1,
#                 [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
#                 [[1], [2], [3]],
#                 [[0, 1, 2, 3]],
#                 [[0.1], [0.2], [0.3]],
#             ),
#             (
#                 1.1,
#                 [[5, 6, 7, 8], [9, 10, 11, 12]],
#                 [[2], [3]],
#                 [[0, 1, 2, 3]],
#                 [[0.2], [0.3]],
#             ),
#             (
#                 2.0,
#                 [[5, 6, 7, 8], [9, 10, 11, 12]],
#                 [[2], [3]],
#                 [[0, 1, 2, 3]],
#                 [[0.2], [0.3]],
#             ),
#             (
#                 2.1,
#                 [[9, 10, 11, 12]],
#                 [[3]],
#                 [[0, 1, 2, 3]],
#                 [[0.3]],
#             ),
#         ],
#     )
#     def test_L_cut(self, _array_generator, L_cut, r_ex, n_ex, phi_ex, sigma_ex):
#         r_array = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
#         n_array = [1, 2, 3]
#         sigma_array = [0.1, 0.2, 0.3]
#         r_counts = np.array(r_array)
#         r_size0, r_size1, *_ = r_counts.shape
#         n_counts = np.array([n_array]).T  # starts as (1,m), we want (m,1)
#         n_size0, n_size1, *_ = n_counts.shape
#         phi_counts = _array_generator(r_size1).reshape(1, -1)
#         sigma_counts = np.array([sigma_array]).T
#         sig_size0, sig_size1, *_ = sigma_counts.shape

#         r = NDHistogram(
#             r_counts, [_array_generator(1, r_size0 + 1), _array_generator(r_size1)]
#         )
#         n = NDHistogram(
#             n_counts, [_array_generator(1, n_size0 + 1), _array_generator(n_size1)]
#         )
#         phi = NDHistogram(phi_counts, [_array_generator(1), _array_generator(r_size1)])
#         sigma = NDHistogram(
#             sigma_counts,
#             [_array_generator(1, sig_size0 + 1), _array_generator(sig_size1)],
#         )
#         print(L_cut)
#         print(r.midpoints)

#         r_clean, n_clean, phi_clean, sigma_clean = clean_data(
#             r, n, phi, sigma=sigma, L_cut=L_cut
#         )

#         r_expected = np.array(r_ex)
#         n_expected = np.array(n_ex)
#         phi_expected = np.array(phi_ex)
#         sigma_expected = np.array(sigma_ex)

#         for actual, expected in zip(
#             [r_clean, n_clean, phi_clean, sigma_clean],
#             [r_expected, n_expected, phi_expected, sigma_expected],
#         ):
#             print(actual.counts)
#             print(expected)
#             assert actual.shape == expected.shape
#             assert (actual.counts == expected).all()

#     def test_N_incompatible(self, _array_generator, _array_generator_2d):
#         big_R = NDHistogram(
#             _array_generator_2d((4, 3)), [_array_generator(4), _array_generator(3)]
#         )
#         big_N = NDHistogram(
#             _array_generator_2d((5, 1)), [_array_generator(5), _array_generator(1)]
#         )
#         big_phi = NDHistogram(
#             _array_generator_2d((1, 3)), [_array_generator(1), _array_generator(3)]
#         )
#         with pytest.raises(ValueError) as excinfo:
#             clean_data(big_R, big_N, big_phi)
#         assert "N" in str(excinfo.value)

#     def test_phi_incompatible(self, _array_generator, _array_generator_2d):
#         big_R = NDHistogram(
#             _array_generator_2d((4, 3)), [_array_generator(4), _array_generator(3)]
#         )
#         big_N = NDHistogram(
#             _array_generator_2d((4, 1)), [_array_generator(4), _array_generator(1)]
#         )
#         big_phi = NDHistogram(
#             _array_generator_2d((1, 4)), [_array_generator(1), _array_generator(4)]
#         )
#         with pytest.raises(ValueError) as excinfo:
#             clean_data(big_R, big_N, big_phi)
#         assert "Phi" in str(excinfo.value)

#     def test_sigma_incompatible(self, _array_generator, _array_generator_2d):
#         big_R = NDHistogram(
#             _array_generator_2d((4, 3)), [_array_generator(4), _array_generator(3)]
#         )
#         big_N = NDHistogram(
#             _array_generator_2d((4, 1)), [_array_generator(4), _array_generator(1)]
#         )
#         big_phi = NDHistogram(
#             _array_generator_2d((1, 3)), [_array_generator(1), _array_generator(3)]
#         )
#         sigma = NDHistogram(
#             _array_generator_2d((5, 1)), [_array_generator(5), _array_generator(1)]
#         )
#         with pytest.raises(ValueError) as excinfo:
#             clean_data(big_R, big_N, big_phi, sigma=sigma)
#         assert "Sigma" in str(excinfo.value)


class TestWeightFactor:
    def test_good_path(self, _array_generator):
        big_r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        big_n = NDHistogram(np.ones((3, 1)), [_array_generator(3), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, 2)), [_array_generator(1), _array_generator(2)]
        )
        W = weight_factor(big_r, big_n, big_phi)
        assert W.shape == big_r.shape
        for actual_mids, expected_mids in zip(W.midpoints, big_r.midpoints):
            assert (actual_mids == expected_mids).all()
        assert (W.counts == (1 / 2)).all()

    def test_sigma(self, _array_generator):
        big_r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        big_n = NDHistogram(np.ones((3, 1)), [_array_generator(3), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, 2)), [_array_generator(1), _array_generator(2)]
        )
        sigma = NDHistogram(
            np.full((3, 1), 2), [_array_generator(3), _array_generator(1)]
        )
        W = weight_factor(big_r, big_n, big_phi, sigma=sigma)
        assert (W.counts == (1 / 8)).all()

    def test_handles_zero_div(self, _array_generator):
        r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        n = NDHistogram(np.ones((3, 1)), [_array_generator(3), _array_generator(1)])
        phi = NDHistogram(np.ones((1, 2)), [_array_generator(1), _array_generator(2)])
        sigma_counts = np.ones((3, 1))
        sigma_counts[1, 0] = 0
        sigma = NDHistogram(sigma_counts, [_array_generator(3), _array_generator(1)])
        w = weight_factor(r, n, phi, sigma=sigma)
        assert w.shape == r.shape
        for actual_mids, expected_mids in zip(w.midpoints, r.midpoints):
            assert (actual_mids == expected_mids).all()
        assert ((w.counts == (1 / 2)) | np.isnan(w.counts)).all()
        assert np.isnan(w.counts[1, 0])

    @pytest.mark.parametrize(
        "n_size,n_mids_args,phi_size,phi_mids_args,sigma_args,expected_text",
        [
            (4, (0, 4), 2, (0, 2), None, ["N", "Shapes"]),
            (2, (0, 2), 2, (0, 2), None, ["N", "Shapes"]),
            (3, (1, 4), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (2, 5), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (0, 3), 3, (0, 3), None, ["Phi", "Shapes"]),
            (3, (0, 3), 1, (0, 1), None, ["Phi", "Shapes"]),
            (3, (0, 3), 2, (1, 3), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (2, 4), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (4, (0, 4)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (2, (0, 2)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (3, (1, 4)), ["Sigma", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (3, (2, 5)), ["Sigma", "Midpoints"]),
        ],
    )
    def test_incompatiblities(
        self,
        _array_generator,
        n_size,
        n_mids_args,
        phi_size,
        phi_mids_args,
        sigma_args,
        expected_text,
    ):
        r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        n = NDHistogram(
            np.ones((n_size, 1)), [_array_generator(*n_mids_args), _array_generator(1)]
        )
        phi = NDHistogram(
            np.ones((1, phi_size)),
            [_array_generator(1), _array_generator(*phi_mids_args)],
        )
        if sigma_args is None:
            sigma = None
        else:
            sigma_size, sigma_mids_args = sigma_args
            sigma = NDHistogram(
                np.full((sigma_size, 1), 0.1),
                [_array_generator(*sigma_mids_args), _array_generator(1)],
            )
        with pytest.raises(ValueError) as excinfo:
            weight_factor(r, n, phi, sigma=sigma)
        print(excinfo.value)
        for expected in expected_text:
            assert expected in str(excinfo.value)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = NDHistogram(
            np.ones((101, 10)), [_array_generator(101), _array_generator(10)]
        )
        big_n = NDHistogram(
            np.ones((101, 1)), [_array_generator(101), _array_generator(1)]
        )
        big_phi = NDHistogram(
            np.ones((1, 10)), [_array_generator(1), _array_generator(10)]
        )
        with pytest.raises(ValueError) as excinfo:
            weight_factor(big_r, big_n, big_phi)
        assert "Dimensions" in str(excinfo.value)


class TestNextPhi:
    def test_good_path(self, _array_generator):
        big_r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        big_n = NDHistogram(np.ones((3, 1)), [_array_generator(3), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, 2)), [_array_generator(1), _array_generator(2)]
        )
        new_phi = next_phi(big_r, big_n, big_phi)
        assert new_phi.shape == big_phi.shape
        assert (new_phi.counts == (1 / 2)).all()
        assert (new_phi.midpoints[1] == big_phi.midpoints[1]).all()

    def test_sigma(self, _array_generator):
        big_r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        big_n = NDHistogram(np.ones((3, 1)), [_array_generator(3), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, 2)), [_array_generator(1), _array_generator(2)]
        )
        sigma = NDHistogram(
            np.full((3, 1), 2), [_array_generator(3), _array_generator(1)]
        )

        new_phi = next_phi(big_r, big_n, big_phi, sigma=sigma)
        assert (new_phi.counts == (1 / 2)).all()

    def test_handles_zero_divide(self, _array_generator):
        big_r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        big_n = NDHistogram(np.ones((3, 1)), [_array_generator(3), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, 2)), [_array_generator(1), _array_generator(2)]
        )
        sigma_counts = np.full((3, 1), 2)
        sigma_counts[1, 0] = 0
        sigma = NDHistogram(sigma_counts, [_array_generator(3), _array_generator(1)])

        new_phi = next_phi(big_r, big_n, big_phi, sigma=sigma)
        assert ((new_phi.counts == (1 / 2)) | np.isnan(new_phi.counts)).all()

    @pytest.mark.parametrize(
        "n_size,n_mids_args,phi_size,phi_mids_args,sigma_args,expected_text",
        [
            (4, (0, 4), 2, (0, 2), None, ["N", "Shapes"]),
            (2, (0, 2), 2, (0, 2), None, ["N", "Shapes"]),
            (3, (1, 4), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (2, 5), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (0, 3), 3, (0, 3), None, ["Phi", "Shapes"]),
            (3, (0, 3), 1, (0, 1), None, ["Phi", "Shapes"]),
            (3, (0, 3), 2, (1, 3), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (2, 4), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (4, (0, 4)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (2, (0, 2)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (3, (1, 4)), ["Sigma", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (3, (2, 5)), ["Sigma", "Midpoints"]),
        ],
    )
    def test_incompatiblities(
        self,
        _array_generator,
        n_size,
        n_mids_args,
        phi_size,
        phi_mids_args,
        sigma_args,
        expected_text,
    ):
        r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        n = NDHistogram(
            np.ones((n_size, 1)), [_array_generator(*n_mids_args), _array_generator(1)]
        )
        phi = NDHistogram(
            np.ones((1, phi_size)),
            [_array_generator(1), _array_generator(*phi_mids_args)],
        )
        if sigma_args is None:
            sigma = None
        else:
            sigma_size, sigma_mids_args = sigma_args
            sigma = NDHistogram(
                np.full((sigma_size, 1), 0.1),
                [_array_generator(*sigma_mids_args), _array_generator(1)],
            )
        with pytest.raises(ValueError) as excinfo:
            next_phi(r, n, phi, sigma=sigma)
        print(excinfo.value)
        for expected in expected_text:
            assert expected in str(excinfo.value)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = NDHistogram(
            np.ones((101, 10)), [_array_generator(101), _array_generator(10)]
        )
        big_n = NDHistogram(
            np.ones((101, 1)), [_array_generator(101), _array_generator(1)]
        )
        big_phi = NDHistogram(
            np.ones((1, 10)), [_array_generator(1), _array_generator(10)]
        )
        with pytest.raises(ValueError) as excinfo:
            next_phi(big_r, big_n, big_phi)
        assert "Dimensions" in str(excinfo.value)


class TestStoppingCriteria:
    def test_good_path(self, _array_generator):
        m = 3
        n = 2
        DOF = (m - 1) * (n - 1)
        expected = (m * (n - 1) * (n - 1)) / DOF

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, n)), [_array_generator(1), _array_generator(n)]
        )
        chi_n = stopping_criteria(big_r, big_n, big_phi)

        assert isinstance(chi_n, float)
        assert chi_n == expected

    def test_sigma(self, _array_generator):
        m = 3
        n = 2
        sigma_val = 2
        DOF = (m - 1) * (n - 1)
        expected = (m * (n - 1) * (n - 1)) / (sigma_val * sigma_val * DOF)

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, n)), [_array_generator(1), _array_generator(n)]
        )
        sigma = NDHistogram(
            np.full((m, 1), sigma_val), [_array_generator(m), _array_generator(1)]
        )
        chi_n = stopping_criteria(big_r, big_n, big_phi, sigma=sigma)

        assert chi_n == expected

    def test_handles_zero_division(self, _array_generator):
        m = 3
        n = 2
        sigma_val = 2
        DOF = (m - 1) * (n - 1)
        expected = ((m - 1) * (n - 1) * (n - 1)) / (sigma_val * sigma_val * DOF)

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        big_phi = NDHistogram(
            np.ones((1, n)), [_array_generator(1), _array_generator(n)]
        )
        sigma_counts = np.full((m, 1), sigma_val)
        sigma_counts[m - 1, 0] = 0
        sigma = NDHistogram(sigma_counts, [_array_generator(m), _array_generator(1)])
        chi_n = stopping_criteria(big_r, big_n, big_phi, sigma=sigma)

        assert chi_n == expected

    @pytest.mark.parametrize(
        "n_size,n_mids_args,phi_size,phi_mids_args,sigma_args,expected_text",
        [
            (4, (0, 4), 2, (0, 2), None, ["N", "Shapes"]),
            (2, (0, 2), 2, (0, 2), None, ["N", "Shapes"]),
            (3, (1, 4), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (2, 5), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (0, 3), 3, (0, 3), None, ["Phi", "Shapes"]),
            (3, (0, 3), 1, (0, 1), None, ["Phi", "Shapes"]),
            (3, (0, 3), 2, (1, 3), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (2, 4), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (4, (0, 4)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (2, (0, 2)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (3, (1, 4)), ["Sigma", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (3, (2, 5)), ["Sigma", "Midpoints"]),
        ],
    )
    def test_incompatiblities(
        self,
        _array_generator,
        n_size,
        n_mids_args,
        phi_size,
        phi_mids_args,
        sigma_args,
        expected_text,
    ):
        r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        n = NDHistogram(
            np.ones((n_size, 1)), [_array_generator(*n_mids_args), _array_generator(1)]
        )
        phi = NDHistogram(
            np.ones((1, phi_size)),
            [_array_generator(1), _array_generator(*phi_mids_args)],
        )
        if sigma_args is None:
            sigma = None
        else:
            sigma_size, sigma_mids_args = sigma_args
            sigma = NDHistogram(
                np.full((sigma_size, 1), 0.1),
                [_array_generator(*sigma_mids_args), _array_generator(1)],
            )
        with pytest.raises(ValueError) as excinfo:
            stopping_criteria(r, n, phi, sigma=sigma)
        print(excinfo.value)
        for expected in expected_text:
            assert expected in str(excinfo.value)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = NDHistogram(
            np.ones((101, 10)), [_array_generator(101), _array_generator(10)]
        )
        big_n = NDHistogram(
            np.ones((101, 1)), [_array_generator(101), _array_generator(1)]
        )
        big_phi = NDHistogram(
            np.ones((1, 10)), [_array_generator(1), _array_generator(10)]
        )
        with pytest.raises(ValueError) as excinfo:
            stopping_criteria(big_r, big_n, big_phi)
        assert "Dimensions" in str(excinfo.value)


class TestUnfoldSpectrum:
    def test_good_path(self, _array_generator):
        m = 3
        n = 2

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        unfolded_phi, info = unfold_spectrum(big_r, big_n)

        print(unfolded_phi.shape)
        print(big_r.shape)
        assert unfolded_phi.shape == (1, n)
        assert (unfolded_phi.midpoints[1] == big_r.midpoints[1]).all()
        assert info is None

    def test_full_info(self, _array_generator):
        m = 3
        n = 2

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        unfolded_phi, info = unfold_spectrum(big_r, big_n, full_info=True)

        assert unfolded_phi.shape == (1, n)
        assert (unfolded_phi.midpoints[1] == big_r.midpoints[1]).all()
        assert info is not None
        assert len(info["errors"]) <= 500
        assert len(info["errors"]) == len(info["phis"])
        assert len(info["errors"]) == len(info["weights"])
        assert len(info["errors"]) == len(info["chis"])

    def test_sigma(self, _array_generator):
        m = 3
        n = 2

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        sigma = NDHistogram(
            np.full((m, 1), 0.5), [_array_generator(m), _array_generator(1)]
        )
        unfolded_phi, _ = unfold_spectrum(big_r, big_n, sigma=sigma)

        assert unfolded_phi.shape == (1, n)
        assert (unfolded_phi.midpoints[1] == big_r.midpoints[1]).all()

    def test_handles_zero_division(self, _array_generator):
        m = 3
        n = 2

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        sigma_counts = np.full((m, 1), 0.5)
        sigma_counts[m - 1, 0] = 0
        sigma = NDHistogram(sigma_counts, [_array_generator(m), _array_generator(1)])
        unfolded_phi, _ = unfold_spectrum(big_r, big_n, sigma=sigma)

        assert unfolded_phi.shape == (1, n)
        assert (unfolded_phi.midpoints[1] == big_r.midpoints[1]).all()

    def test_starting_phi(self, _array_generator):
        m = 3
        n = 2

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        phi0 = NDHistogram(
            np.full((1, n), 0.5), [_array_generator(1), _array_generator(n)]
        )
        unfolded_phi, _ = unfold_spectrum(big_r, big_n, phi0=phi0)

        assert unfolded_phi.shape == (1, n)
        assert (unfolded_phi.midpoints[1] == big_r.midpoints[1]).all()

    def test_L_cut(self, _array_generator):
        m = 3
        n = 2

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])
        unfolded_phi, full_info = unfold_spectrum(big_r, big_n, full_info=True, L_cut=1)

        assert unfolded_phi.shape == (1, n)
        assert (unfolded_phi.midpoints[1] == big_r.midpoints[1]).all()
        assert full_info is not None
        assert full_info["weights"][0].shape == (m - 1, n)

    def test_tolerance(self, _array_generator):
        m = 3
        n = 2
        fill_value = 123

        big_r = NDHistogram(
            np.full((m, n), fill_value), [_array_generator(m), _array_generator(n)]
        )
        big_n = NDHistogram(
            np.full((m, 1), fill_value), [_array_generator(m), _array_generator(1)]
        )
        unfolded_phi, _ = unfold_spectrum(big_r, big_n, tolerance=1000000)

        assert unfolded_phi.shape == (1, n)
        assert (unfolded_phi.midpoints[1] == big_r.midpoints[1]).all()
        print(unfolded_phi.counts)
        assert (unfolded_phi.counts == 1).all()

    def test_max_iterations(self, _array_generator):
        m = 3
        n = 2

        big_r = NDHistogram(np.ones((m, n)), [_array_generator(m), _array_generator(n)])
        big_n = NDHistogram(np.ones((m, 1)), [_array_generator(m), _array_generator(1)])

        _, info = unfold_spectrum(big_r, big_n, max_iterations=1, full_info=True)
        assert info is not None
        assert len(info["errors"]) == 1

    @pytest.mark.parametrize(
        "n_size,n_mids_args,phi_size,phi_mids_args,sigma_args,expected_text",
        [
            (4, (0, 4), 2, (0, 2), None, ["N", "Shapes"]),
            (2, (0, 2), 2, (0, 2), None, ["N", "Shapes"]),
            (3, (1, 4), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (2, 5), 2, (0, 2), None, ["N", "Midpoints"]),
            (3, (0, 3), 3, (0, 3), None, ["Phi", "Shapes"]),
            (3, (0, 3), 1, (0, 1), None, ["Phi", "Shapes"]),
            (3, (0, 3), 2, (1, 3), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (2, 4), None, ["Phi", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (4, (0, 4)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (2, (0, 2)), ["Sigma", "Shapes"]),
            (3, (0, 3), 2, (0, 2), (3, (1, 4)), ["Sigma", "Midpoints"]),
            (3, (0, 3), 2, (0, 2), (3, (2, 5)), ["Sigma", "Midpoints"]),
        ],
    )
    def test_incompatiblities(
        self,
        _array_generator,
        n_size,
        n_mids_args,
        phi_size,
        phi_mids_args,
        sigma_args,
        expected_text,
    ):
        r = NDHistogram(np.ones((3, 2)), [_array_generator(3), _array_generator(2)])
        n = NDHistogram(
            np.ones((n_size, 1)), [_array_generator(*n_mids_args), _array_generator(1)]
        )
        phi = NDHistogram(
            np.ones((1, phi_size)),
            [_array_generator(1), _array_generator(*phi_mids_args)],
        )
        if sigma_args is None:
            sigma = None
        else:
            sigma_size, sigma_mids_args = sigma_args
            sigma = NDHistogram(
                np.full((sigma_size, 1), 0.1),
                [_array_generator(*sigma_mids_args), _array_generator(1)],
            )
        with pytest.raises(ValueError) as excinfo:
            unfold_spectrum(r, n, phi, sigma=sigma)
        print(excinfo.value)
        for expected in expected_text:
            assert expected in str(excinfo.value)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = NDHistogram(
            np.ones((101, 10)), [_array_generator(101), _array_generator(10)]
        )
        big_n = NDHistogram(
            np.ones((101, 1)), [_array_generator(101), _array_generator(1)]
        )
        big_phi = NDHistogram(
            np.ones((1, 10)), [_array_generator(1), _array_generator(10)]
        )
        with pytest.raises(ValueError) as excinfo:
            unfold_spectrum(big_r, big_n, big_phi)
        assert "Dimensions" in str(excinfo.value)
