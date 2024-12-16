import numpy as np
from data_processing.processing.spectrum_unfolding import (
    Histogram,
    Histogram2D,
    _are_histograms_compatible,
    _are_histograms_compatible_2d,
    _are_r_dimensions_close_enough,
    weight_factor,
    next_phi,
    stopping_criteria,
    unfold_spectrum,
)
import pytest
from math import pow


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


class TestHistogram:
    def test_inits(self):
        counts = np.ones(4)
        midpoints = np.ones(4)
        histogram = Histogram(counts, midpoints)
        assert all(counts == histogram.counts)
        assert all(midpoints == histogram.midpoints)

    def test_wrong_shape_counts(self):
        with pytest.raises(ValueError):
            Histogram(np.ones((4, 1)), np.ones(4))
        with pytest.raises(ValueError):
            Histogram(np.ones((1, 4)), np.ones(4))
        with pytest.raises(ValueError):
            Histogram(np.ones((4, 2)), np.ones(4))

    def test_wrong_shape_mids(self):
        with pytest.raises(ValueError):
            Histogram(np.ones(4), np.ones((4, 1)))
        with pytest.raises(ValueError):
            Histogram(np.ones(4), np.ones((1, 4)))
        with pytest.raises(ValueError):
            Histogram(np.ones(4), np.ones((4, 2)))

    def test_wrong_size_mids(self):
        with pytest.raises(ValueError):
            Histogram(np.ones(4), np.ones(5))
        with pytest.raises(ValueError):
            Histogram(np.ones(4), np.ones(3))


class TestHistogram2D:
    def test_inits(self):
        counts = np.ones((4, 3))
        x_mids = np.ones(4)
        y_mids = np.ones(3)
        histogram = Histogram2D(counts, x_mids, y_mids)
        assert (counts == histogram.counts).all()
        assert all(x_mids == histogram.x_midpoints)
        assert all(y_mids == histogram.y_midpoints)

    def test_wrong_shape_counts(self):
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3, 2)), np.ones(4), np.ones(3))
        with pytest.raises(ValueError):
            Histogram2D(np.ones(4), np.ones(4), np.ones(3))

    def test_wrong_shape_mids(self):
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones((4, 1)), np.ones(3))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones((1, 4)), np.ones(3))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones((4, 2)), np.ones(3))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones((4, 2)), np.ones((3, 1)))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones((4, 2)), np.ones((1, 3)))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones((4, 2)), np.ones((3, 2)))

    def test_wrong_size_mids(self):
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones(5), np.ones(3))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones(3), np.ones(3))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones(4), np.ones(4))
        with pytest.raises(ValueError):
            Histogram2D(np.ones((4, 3)), np.ones(4), np.ones(2))


class TestAreHistogramsCompatible:
    def test_compatible(self, _array_generator):
        a = Histogram(_array_generator(5), _array_generator(5))
        b = Histogram(_array_generator(1, 6), _array_generator(5))
        compatible, _ = _are_histograms_compatible(a, b)
        assert compatible

    def test_size_mismatch(self, _array_generator):
        a = Histogram(_array_generator(5), _array_generator(5))
        b = Histogram(_array_generator(1, 5), _array_generator(4))
        compatible, _ = _are_histograms_compatible(a, b)
        assert not compatible

    def test_mids_mismatch(self, _array_generator):
        a = Histogram(_array_generator(5), _array_generator(5))
        b = Histogram(_array_generator(1, 6), _array_generator(1, 6))
        compatible, _ = _are_histograms_compatible(a, b)
        assert not compatible


class TestAreHistogramsCompatible2D:
    def test_compatible(self, _array_generator, _array_generator_2d):
        a = Histogram2D(
            _array_generator_2d((4, 3)), _array_generator(4), _array_generator(3)
        )
        x = Histogram(_array_generator(4), _array_generator(4))
        y = Histogram(_array_generator(3), _array_generator(3))
        compatible, _ = _are_histograms_compatible_2d(a, x, 0)
        assert compatible
        compatible, _ = _are_histograms_compatible_2d(a, y, 1)
        assert compatible

    def test_size_mismatch(self, _array_generator, _array_generator_2d):
        a = Histogram2D(
            _array_generator_2d((4, 3)), _array_generator(4), _array_generator(3)
        )
        x = Histogram(_array_generator(5), _array_generator(5))
        y = Histogram(_array_generator(2), _array_generator(2))
        compatible, _ = _are_histograms_compatible_2d(a, x, 0)
        assert not compatible
        compatible, _ = _are_histograms_compatible_2d(a, y, 1)
        assert not compatible

    def test_mids_mismatch(self, _array_generator, _array_generator_2d):
        a = Histogram2D(
            _array_generator_2d((4, 3)), _array_generator(4), _array_generator(3)
        )
        x = Histogram(_array_generator(4), _array_generator(1, 5))
        y = Histogram(_array_generator(3), _array_generator(2, 5))
        compatible, _ = _are_histograms_compatible_2d(a, x, 0)
        assert not compatible
        compatible, _ = _are_histograms_compatible_2d(a, y, 1)
        assert not compatible


class TestAreRDimensionsCloseEnough:
    def test_close_enough(self, _array_generator):
        m = 10
        n = 100
        r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        assert _are_r_dimensions_close_enough(r)

    def test_not_close_enough(self, _array_generator):
        m = 10
        n = 101
        r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        assert not _are_r_dimensions_close_enough(r)


class TestWeightFactor:
    def test_good_path(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        W = weight_factor(big_r, big_phi, big_n)
        assert W.counts.shape == big_r.counts.shape
        assert (W.x_midpoints == big_r.x_midpoints).all()
        assert (W.y_midpoints == big_r.y_midpoints).all()
        assert (W.counts == (1 / 2)).all()

    def test_sigma(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.full(3, 2), _array_generator(3))
        W = weight_factor(big_r, big_phi, big_n, sigma=sigma)
        assert (W.counts == (1 / 8)).all()

    def test_N_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(4), _array_generator(4))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)
        big_n = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)

    def test_N_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(1, 4))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)
        big_n = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)

    def test_phi_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(3), _array_generator(3))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)
        big_phi = Histogram(np.ones(1), _array_generator(1))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)

    def test_phi_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(1, 3))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)
        big_phi = Histogram(np.ones(2), _array_generator(2, 4))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = Histogram2D(
            np.ones((101, 10)), _array_generator(101), _array_generator(10)
        )
        big_n = Histogram(np.ones(101), _array_generator(101))
        big_phi = Histogram(np.ones(10), _array_generator(10))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_phi, big_n)

    def test_sigma_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.ones(4), _array_generator(4))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi, sigma=sigma)
        sigma = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi, sigma=sigma)

    def test_sigma_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.ones(3), _array_generator(1, 4))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi, sigma=sigma)
        sigma = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            weight_factor(big_r, big_n, big_phi, sigma=sigma)


class TestNextPhi:
    def test_good_path(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        new_phi = next_phi(big_r, big_phi, big_n)
        assert new_phi.counts.shape == big_phi.counts.shape
        assert (new_phi.midpoints == big_r.y_midpoints).all()
        assert (new_phi.counts == (1 / 2)).all()

    def test_sigma(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.full(3, 2), _array_generator(3))
        new_phi = next_phi(big_r, big_phi, big_n, sigma=sigma)
        assert (new_phi.counts == (1 / 2)).all()

    def test_N_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(4), _array_generator(4))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)
        big_n = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)

    def test_N_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(1, 4))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)
        big_n = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)

    def test_phi_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(3), _array_generator(3))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)
        big_phi = Histogram(np.ones(1), _array_generator(1))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)

    def test_phi_edges_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(1, 3))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)
        big_phi = Histogram(np.ones(2), _array_generator(2, 4))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = Histogram2D(
            np.ones((101, 10)), _array_generator(101), _array_generator(10)
        )
        big_n = Histogram(np.ones(101), _array_generator(101))
        big_phi = Histogram(np.ones(10), _array_generator(10))
        with pytest.raises(ValueError):
            next_phi(big_r, big_phi, big_n)

    def test_sigma_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.ones(4), _array_generator(4))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi, sigma=sigma)
        sigma = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi, sigma=sigma)

    def test_sigma_edges_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.ones(3), _array_generator(1, 4))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi, sigma=sigma)
        sigma = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            next_phi(big_r, big_n, big_phi, sigma=sigma)


class TestStoppingCriteria:
    def test_good_path(self, _array_generator):
        m = 3
        n = 2
        DOF = (m - 1) * (n - 1)
        expected = (m * (n - 1) * (n - 1)) / DOF

        big_r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        big_n = Histogram(np.ones(m), _array_generator(m))
        big_phi = Histogram(np.ones(n), _array_generator(n))
        chi_n = stopping_criteria(big_r, big_phi, big_n)

        assert isinstance(chi_n, float)
        assert chi_n == expected

    def test_sigma(self, _array_generator):
        m = 3
        n = 2
        sigma_val = 2
        DOF = (m - 1) * (n - 1)
        expected = (m * (n - 1) * (n - 1)) / (sigma_val * sigma_val * DOF)

        big_r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        big_n = Histogram(np.ones(m), _array_generator(m))
        big_phi = Histogram(np.ones(n), _array_generator(n))
        sigma = Histogram(np.full(m, sigma_val), _array_generator(m))
        chi_n = stopping_criteria(big_r, big_phi, big_n, sigma=sigma)

        assert chi_n == expected

    def test_N_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(4), _array_generator(4))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)
        big_n = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)

    def test_N_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(1, 4))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)
        big_n = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)

    def test_phi_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(3), _array_generator(3))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)
        big_phi = Histogram(np.ones(1), _array_generator(1))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)

    def test_phi_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(1, 3))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)
        big_phi = Histogram(np.ones(2), _array_generator(2, 4))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = Histogram2D(
            np.ones((101, 10)), _array_generator(101), _array_generator(10)
        )
        big_n = Histogram(np.ones(101), _array_generator(101))
        big_phi = Histogram(np.ones(10), _array_generator(10))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_phi, big_n)

    def test_sigma_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.ones(4), _array_generator(4))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi, sigma=sigma)
        sigma = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi, sigma=sigma)

    def test_sigma_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(2))
        sigma = Histogram(np.ones(3), _array_generator(1, 4))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi, sigma=sigma)
        sigma = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            stopping_criteria(big_r, big_n, big_phi, sigma=sigma)


class TestUnfoldSpectrum:
    def test_good_path(self, _array_generator):
        m = 3
        n = 2

        big_r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        big_n = Histogram(np.ones(m), _array_generator(m))
        unfolded_phi = unfold_spectrum(big_n, big_r)

        assert unfolded_phi.counts.shape[0] == big_r.counts.shape[1]
        assert (unfolded_phi.midpoints == big_r.y_midpoints).all()

    def test_sigma(self, _array_generator):
        m = 3
        n = 2

        big_r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        big_n = Histogram(np.ones(m), _array_generator(m))
        sigma = Histogram(np.full(m, 0.5), _array_generator(m))
        unfolded_phi = unfold_spectrum(big_n, big_r, sigma=sigma)

        assert unfolded_phi.counts.shape[0] == big_r.counts.shape[1]
        assert (unfolded_phi.midpoints == big_r.y_midpoints).all()

    def test_starting_phi(self, _array_generator):
        m = 3
        n = 2

        big_r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        big_n = Histogram(np.ones(m), _array_generator(m))
        starting_phi = Histogram(np.full(n, 0.5), _array_generator(n))
        unfolded_phi = unfold_spectrum(big_n, big_r, starting_phi=starting_phi)

        assert unfolded_phi.counts.shape[0] == big_r.counts.shape[1]
        assert (unfolded_phi.midpoints == big_r.y_midpoints).all()

    def test_tolerance(self, _array_generator):
        m = 3
        n = 2
        fill_value = 123

        big_r = Histogram2D(
            np.full((m, n), fill_value), _array_generator(m), _array_generator(n)
        )
        big_n = Histogram(np.full(m, fill_value), _array_generator(m))
        unfolded_phi = unfold_spectrum(big_n, big_r, tolerance=1000000)

        assert unfolded_phi.counts.shape[0] == big_r.counts.shape[1]
        assert (unfolded_phi.midpoints == big_r.y_midpoints).all()
        print(unfolded_phi.counts)
        assert (unfolded_phi.counts == 1).all()

    def test_max_iterations(self, _array_generator):
        m = 3
        n = 2

        big_r = Histogram2D(np.ones((m, n)), _array_generator(m), _array_generator(n))
        big_n = Histogram(np.ones(m), _array_generator(m))

        with pytest.raises(RuntimeError):
            unfold_spectrum(big_n, big_r, max_iterations=1)

    def test_N_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(4), _array_generator(4))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r)
        big_n = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r)

    def test_N_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(1, 4))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r)
        big_n = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r)

    def test_phi_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(3), _array_generator(3))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, starting_phi=big_phi)
        big_phi = Histogram(np.ones(1), _array_generator(1))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, starting_phi=big_phi)

    def test_phi_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        big_phi = Histogram(np.ones(2), _array_generator(1, 3))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, starting_phi=big_phi)
        big_phi = Histogram(np.ones(2), _array_generator(2, 4))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, starting_phi=big_phi)

    def test_r_dimensions_close_enough(self, _array_generator):
        big_r = Histogram2D(
            np.ones((101, 10)), _array_generator(101), _array_generator(10)
        )
        big_n = Histogram(np.ones(101), _array_generator(101))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r)

    def test_sigma_size_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        sigma = Histogram(np.ones(4), _array_generator(4))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, sigma=sigma)
        sigma = Histogram(np.ones(2), _array_generator(2))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, sigma=sigma)

    def test_sigma_mids_mismatch(self, _array_generator):
        big_r = Histogram2D(np.ones((3, 2)), _array_generator(3), _array_generator(2))
        big_n = Histogram(np.ones(3), _array_generator(3))
        sigma = Histogram(np.ones(3), _array_generator(1, 4))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, sigma=sigma)
        sigma = Histogram(np.ones(3), _array_generator(2, 5))
        with pytest.raises(ValueError):
            unfold_spectrum(big_n, big_r, sigma=sigma)
