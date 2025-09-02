import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from mdl_density_hist import mdl_optimal_histogram

import sys

sys.dont_write_bytecode = True

K_max = 100
epsilon = 0.1

@pytest.fixture
def rootdir_path(request):
    return request.config.rootdir

def test_gmm3_id_0(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm3_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm3_samples_MDL_lower_bins.parquet.brotli")
    id = 0

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm3_id_1(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm3_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm3_samples_MDL_lower_bins.parquet.brotli")
    id = 1

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm3_id_2(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm3_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm3_samples_MDL_lower_bins.parquet.brotli")
    id = 2

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)


def test_gmm4_id_0(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples_MDL_lower_bins.parquet.brotli")
    id = 0

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm4_id_1(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples_MDL_lower_bins.parquet.brotli")
    id = 1

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm4_id_2(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples_MDL_lower_bins.parquet.brotli")
    id = 2

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm4_id_3(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm4_samples_MDL_lower_bins.parquet.brotli")
    id = 3

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm5_id_0(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples_MDL_lower_bins.parquet.brotli")
    id = 0

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm5_id_1(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples_MDL_lower_bins.parquet.brotli")
    id = 1

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm5_id_2(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples_MDL_lower_bins.parquet.brotli")
    id = 2

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm5_id_3(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples_MDL_lower_bins.parquet.brotli")
    id = 3

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)

def test_gmm5_id_4(rootdir_path):
    gmmX_ds = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples.parquet.brotli")
    gmmX_mdl_lower_bins = pd.read_parquet(Path(rootdir_path) / Path("tests") / "gmm5_samples_MDL_lower_bins.parquet.brotli")
    id = 4

    assert all(np.unique(gmmX_ds["dataset_id"]) == np.unique(gmmX_mdl_lower_bins["dataset_id"])), "The number of dataset ids must equal"
    dataset = gmmX_ds[gmmX_ds["dataset_id"] == id]["value"].to_numpy()
    lower_bins_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["lower_bin"].dropna().to_numpy()
    K_scores_true = gmmX_mdl_lower_bins[gmmX_mdl_lower_bins["dataset_id"] == id]["K_score"].to_numpy()

    lower_bins_pred, K_scores_pred = mdl_optimal_histogram(dataset, K_max=K_max, epsilon=epsilon)

    np.testing.assert_array_almost_equal(lower_bins_pred, lower_bins_true, decimal=5)
    np.testing.assert_allclose(K_scores_pred, K_scores_true, rtol=1e-5)
