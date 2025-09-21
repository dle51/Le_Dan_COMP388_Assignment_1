# test_utils.py
import matplotlib as plt
import pytest
from pytest_check import check

import utils.utils as utils


# Test pairwise sum function
def test_pairwise_exception():
    with pytest.raises(Exception):
        utils._pairwise_sum(None)


def test_pairwise_empty_array():
    assert utils._pairwise_sum([]) == 0.0


def test_pairwise_one_element():
    assert utils._pairwise_sum([10]) == 10.0


def test_pairwise_even_elements():
    assert utils._pairwise_sum([10, 20]) == 30.0


def test_pairwise_even_elements_2():
    assert utils._pairwise_sum([10, 20, -30, -0.5]) == -0.5


def test_pairwise_odd_elements():
    assert utils._pairwise_sum([10, 20, -30]) == 0.0


def test_pairwise_odd_elements_2():
    assert utils._pairwise_sum([10, 20, -30, -0.5, 0.5]) == 0.0


# Test compute variance function
def test_compute_var_exception():
    with pytest.raises(Exception):
        utils._compute_var(None)


def test_compute_var_empty_array():
    assert not float("-inf") < utils._compute_var([]) < float("inf")


def test_compute_var_one_element():
    assert utils._compute_var([10]) == 0.0


def test_compute_var_even_elements():
    assert float("-inf") < utils._compute_var([10, 20]) < float("inf")


def test_compute_var_even_elements_2():
    assert float("-inf") < utils._compute_var([10, 20, -30, -0.5]) < float("inf")


def test_compute_var_odd_elements():
    assert float("-inf") < utils._compute_var([10, 20, -30]) < float("inf")


def test_compute_var_odd_elements_2():
    assert float("-inf") < utils._compute_var([10, 20, -30, -0.5, 0.5]) < float("inf")


# Test load_data function
def test_load_data_nofile():
    with pytest.raises(Exception):
        utils.load_data("nofile.csv")


def test_load_data():
    output = utils.load_data("src/data/s1.csv")
    assert len(output) > 0


# Test compute d prime
def test_compute_d_prime_none():
    with pytest.raises(Exception):
        utils.compute_d_prime(None)


def test_compute_d_prime_exception():
    with pytest.raises(Exception):
        utils.compute_d_prime([0])


def test_compute_d_prime_inf():
    assert not float("-inf") < utils.compute_d_prime([]) < float("inf")


def test_compute_d_prime_inc_1():
    assert not float("-inf") < utils.compute_d_prime([(0, 0.1)]) < float("inf")


def test_compute_d_prime_inc_2():
    assert not float("-inf") < utils.compute_d_prime([(1, 0.1)]) < float("inf")


def test_compute_d_prime():
    d_prime = utils.compute_d_prime([(0, 2), (0, 4), (1, 0), (1, 2)])
    assert d_prime == 2.0


# Test FMR computation function
def test_compute_sim_fmr_none():
    with pytest.raises(Exception):
        utils.compute_sim_fmr(None, 0.0)


def test_compute_sim_fmr_exception():
    with pytest.raises(Exception):
        utils.compute_sim_fmr([0], 0.0)


def test_compute_sim_fmr_empty_array():
    assert not float("-inf") < utils.compute_sim_fmr([], 0.0) < float("inf")


def test_compute_sim_fmr_no_genuine():
    assert float("-inf") < utils.compute_sim_fmr([(0, 0.1)], 0.0) < float("inf")


def test_compute_sim_fmr_no_imposter():
    assert not float("-inf") < utils.compute_sim_fmr([(1, 0.1)], 0.0) < float("inf")


def test_compute_sim_fmr_1():
    fmr = utils.compute_sim_fmr([(0, 0.1), (0, 0.3), (1, 0.0)], 0.25)
    assert fmr == 0.5


def test_compute_sim_fmr_2():
    fmr = utils.compute_sim_fmr([(0, 0.1), (0, 0.3), (1, 0.0)], 0.35)
    assert fmr == 0.0


def test_compute_sim_fmr_3():
    fmr = utils.compute_sim_fmr([(0, 0.1), (0, 0.3), (1, 0.0)], 0.05)
    assert fmr == 1.0


# Test FNMR computation function
def test_FNMR_none():
    with pytest.raises(Exception):
        utils.compute_sim_fnmr(None, 0.0)


def test_FNMR_exception():
    with pytest.raises(Exception):
        utils.compute_sim_fnmr([0], 0.0)


def test_FNMR_empty_array():
    assert not float("-inf") < utils.compute_sim_fnmr([], 0.0) < float("inf")


def test_FNMR_missing_genuine():
    assert not float("-inf") < utils.compute_sim_fnmr([(0, 0.1)], 0.0) < float("inf")


def test_FNMR_missing_imposter():
    assert float("-inf") < utils.compute_sim_fnmr([(1, 0.1)], 0.0) < float("inf")


def test_FNMR_1():
    fnmr = utils.compute_sim_fnmr([(0, 0.0), (1, 0.1), (1, 0.3)], 0.25)
    with check:
        assert fnmr == 0.5
    fnmr = utils.compute_sim_fnmr([(0, 0.0), (1, 0.1), (1, 0.3)], 0.5)
    with check:
        assert fnmr == 1.0
    fnmr = utils.compute_sim_fnmr([(0, 0.0), (1, 0.1), (1, 0.3)], 0.05)
    with check:
        assert fnmr == 0.0


# Test FMR FNMR EER function
def test_EER_none():
    with pytest.raises(Exception):
        utils.compute_sim_fmr_fnmr_eer(None)


def test_EER_exception():
    with pytest.raises(Exception):
        utils.compute_sim_fmr_fnmr_eer([0])


def test_EER_empty_arr():
    with check:
        assert not float("-inf") < utils.compute_sim_fmr_fnmr_eer([])[0] < float("inf")
    with check:
        assert not float("-inf") < utils.compute_sim_fmr_fnmr_eer([])[1] < float("inf")
    with check:
        assert not float("-inf") < utils.compute_sim_fmr_fnmr_eer([])[2] < float("inf")


def test_EER_missing_genuine():
    with check:
        assert (
            not float("-inf")
            < utils.compute_sim_fmr_fnmr_eer([(0, 0.1)])[0]
            < float("inf")
        )
    with check:
        assert (
            not float("-inf")
            < utils.compute_sim_fmr_fnmr_eer([(0, 0.1)])[1]
            < float("inf")
        )
    with check:
        assert (
            not float("-inf")
            < utils.compute_sim_fmr_fnmr_eer([(0, 0.1)])[2]
            < float("inf")
        )


def test_EER_missing_imposter():
    with check:
        assert (
            not float("-inf")
            < utils.compute_sim_fmr_fnmr_eer([(1, 0.1)])[0]
            < float("inf")
        )
    with check:
        assert (
            not float("-inf")
            < utils.compute_sim_fmr_fnmr_eer([(1, 0.1)])[1]
            < float("inf")
        )
    with check:
        assert (
            not float("-inf")
            < utils.compute_sim_fmr_fnmr_eer([(1, 0.1)])[2]
            < float("inf")
        )


def test_EER():
    fnmr, fmr, eer = utils.compute_sim_fmr_fnmr_eer(
        [(0, 0.2), (0, 0.3), (0, 0.4), (1, 0.5), (1, 0.6), (1, 0.7)]
    )
    with check:
        assert fnmr == 0.0
    with check:
        assert fmr == 0.0
    with check:
        assert eer == 0.5

def test_AUC_typeerror():
    with pytest.raises(Exception):
        utils.compute_sim_fmr_tmr_auc([0])

def test_AUC_empty_array():
    with check:
        assert not float('-inf') < utils.compute_sim_fmr_tmr_auc([])[0] < float('inf')
    with check:
        assert len(utils.compute_sim_fmr_tmr_auc([])[1]) == 0
    with check:
        assert len(utils.compute_sim_fmr_tmr_auc([])[2]) == 0

def test_AUC_missing_genuine():
    with check:
        assert not float('-inf') < utils.compute_sim_fmr_tmr_auc([(0, 0.1)])[0] < float('inf')
    with check:
        assert len(utils.compute_sim_fmr_tmr_auc([(0, 0.1)])[1]) == 0
    with check:
        assert len(utils.compute_sim_fmr_tmr_auc([(0, 0.1)])[2]) == 0

def test_AUC_missing_imposter():
    with check:
        assert not float('-inf') < utils.compute_sim_fmr_tmr_auc([(1, 0.1)])[0] < float('inf')
    with check:
        assert len(utils.compute_sim_fmr_tmr_auc([(1, 0.1)])[1]) == 0
    with check:
        assert len(utils.compute_sim_fmr_tmr_auc([(1, 0.1)])[2]) == 0

def test_AUC():
    auc, fmrs, tmrs = utils.compute_sim_fmr_tmr_auc([(0, 0.2), (0, 0.3), (0, 0.4), (1, 0.5), (1, 0.6), (1, 0.7)])
    with check:
        assert len(fmrs) > 0
    with check:
        assert len(tmrs) > 0
    with check:
        assert len(fmrs) == len(tmrs)


# Test AUC function
def test_AUC_none_exception():
    with pytest.raises(Exception):
        utils.compute_sim_fmr_tmr_auc(None)


# Test matplotlib import
def test_matplotlib():
    assert plt.__version__ is not None


# END FILE
