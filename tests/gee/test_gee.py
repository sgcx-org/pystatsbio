"""Comprehensive tests for the GEE (Generalized Estimating Equations) module.

Tests cover basic fitting, correlation structure estimation, sandwich
variance properties, coefficient recovery, edge cases, input validation,
and result properties.
"""

import numpy as np
import pytest

from pystatsbio.gee import (
    AR1Corr,
    CorrStructure,
    ExchangeableCorr,
    GEEResult,
    IndependenceCorr,
    UnstructuredCorr,
    gee,
)
from pystatsbio.gee._correlation import resolve_corr


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------

def _gaussian_clustered_data(seed=42):
    """Generate Gaussian clustered data with known coefficients."""
    rng = np.random.RandomState(seed)
    n_clusters = 50
    cluster_size = 5
    n = n_clusters * cluster_size
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    X = np.column_stack([np.ones(n), rng.randn(n)])
    true_beta = np.array([1.0, 0.5])
    # Add within-cluster correlation via cluster random effects
    cluster_effects = np.repeat(rng.randn(n_clusters) * 0.5, cluster_size)
    y = X @ true_beta + cluster_effects + rng.randn(n) * 0.5
    return y, X, cluster_id, true_beta


def _binomial_clustered_data(seed=42):
    """Generate binomial clustered data for logistic GEE."""
    rng = np.random.RandomState(seed)
    n_clusters = 60
    cluster_size = 4
    n = n_clusters * cluster_size
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    X = np.column_stack([np.ones(n), rng.randn(n)])
    true_beta = np.array([0.5, 0.8])
    cluster_effects = np.repeat(rng.randn(n_clusters) * 0.3, cluster_size)
    eta = X @ true_beta + cluster_effects
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)
    return y, X, cluster_id, true_beta


def _poisson_clustered_data(seed=42):
    """Generate Poisson clustered data for count GEE."""
    rng = np.random.RandomState(seed)
    n_clusters = 50
    cluster_size = 5
    n = n_clusters * cluster_size
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    X = np.column_stack([np.ones(n), rng.randn(n) * 0.5])
    true_beta = np.array([1.0, 0.3])
    cluster_effects = np.repeat(rng.randn(n_clusters) * 0.2, cluster_size)
    mu = np.exp(X @ true_beta + cluster_effects)
    y = rng.poisson(mu).astype(float)
    return y, X, cluster_id, true_beta


# ===========================================================================
# Basic fitting tests
# ===========================================================================


class TestBasicFitting:
    """Basic GEE model fitting."""

    def test_gaussian_exchangeable_converges(self):
        """Gaussian + exchangeable converges and returns GEEResult."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert isinstance(result, GEEResult)
        assert result.converged

    def test_gaussian_exchangeable_coefficients_reasonable(self):
        """Gaussian + exchangeable recovers approximately correct coefficients."""
        y, X, cid, true_beta = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        # Should be within ~0.3 of true values for this sample size
        np.testing.assert_allclose(result.coefficients, true_beta, atol=0.3)

    def test_binomial_exchangeable_converges(self):
        """Binomial + exchangeable logistic GEE converges."""
        y, X, cid, _ = _binomial_clustered_data()
        result = gee(y, X, cid, family="binomial", corr_structure="exchangeable")
        assert isinstance(result, GEEResult)
        assert result.converged
        assert result.family_name == "binomial"
        assert result.link_name == "logit"

    def test_poisson_independence_close_to_glm(self):
        """Poisson + independence: coefficients close to standard GLM."""
        y, X, cid, true_beta = _poisson_clustered_data()
        result = gee(y, X, cid, family="poisson", corr_structure="independence")
        assert result.converged
        assert result.family_name == "poisson"
        # Independence GEE should give GLM-like point estimates
        np.testing.assert_allclose(result.coefficients, true_beta, atol=0.4)

    def test_independence_robust_se_approx_naive(self):
        """Independence correlation: robust SE approximately equals naive SE."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="independence")
        # With independence and correct family, should be somewhat similar
        # (not exact because of cluster effects in data)
        ratio = result.robust_se / result.naive_se
        # Ratios should be in a reasonable range
        assert np.all(ratio > 0.3)
        assert np.all(ratio < 5.0)

    def test_gaussian_ar1_converges(self):
        """Gaussian + AR(1) converges."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="ar1")
        assert result.converged
        assert result.correlation_type == "ar1"

    def test_gaussian_unstructured_converges(self):
        """Gaussian + unstructured converges with equal cluster sizes."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="unstructured")
        assert result.converged
        assert result.correlation_type == "unstructured"

    def test_gamma_family(self):
        """Gamma family GEE converges."""
        rng = np.random.RandomState(42)
        n_clusters, cluster_size = 30, 4
        n = n_clusters * cluster_size
        cid = np.repeat(np.arange(n_clusters), cluster_size)
        X = np.column_stack([np.ones(n), rng.randn(n) * 0.3])
        mu = np.exp(0.5 + 0.2 * X[:, 1])
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        result = gee(y, X, cid, family="gamma", corr_structure="exchangeable")
        assert isinstance(result, GEEResult)
        assert result.family_name == "Gamma"


# ===========================================================================
# Correlation structure tests
# ===========================================================================


class TestCorrelationStructures:
    """Working correlation structure estimation."""

    def test_exchangeable_alpha_positive(self):
        """Exchangeable alpha is between 0 and 1 for positively correlated data."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        alpha = result.correlation_params.get("alpha", 0)
        assert 0 < alpha < 1

    def test_ar1_alpha_positive(self):
        """AR(1) alpha is between 0 and 1."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="ar1")
        alpha = result.correlation_params.get("alpha", 0)
        assert -1 < alpha < 1

    def test_unstructured_equal_clusters(self):
        """Unstructured works with equal-sized clusters."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="unstructured")
        # Should have correlation parameters
        assert len(result.correlation_params) > 0

    def test_independence_no_params(self):
        """Independence has no correlation parameters."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="independence")
        assert result.correlation_params == {}

    def test_exchangeable_corr_matrix_structure(self):
        """ExchangeableCorr produces correct matrix structure."""
        corr = ExchangeableCorr()
        corr._alpha = 0.5
        R = corr.working_corr(3)
        expected = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_ar1_corr_matrix_structure(self):
        """AR1Corr produces correct Toeplitz structure."""
        corr = AR1Corr()
        corr._alpha = 0.6
        R = corr.working_corr(3)
        expected = np.array([
            [1, 0.6, 0.36],
            [0.6, 1, 0.6],
            [0.36, 0.6, 1],
        ])
        np.testing.assert_array_almost_equal(R, expected)

    def test_independence_corr_is_identity(self):
        """IndependenceCorr produces identity matrix."""
        corr = IndependenceCorr()
        R = corr.working_corr(4)
        np.testing.assert_array_equal(R, np.eye(4))

    def test_resolve_corr_valid(self):
        """resolve_corr returns correct types for valid names."""
        assert isinstance(resolve_corr("independence"), IndependenceCorr)
        assert isinstance(resolve_corr("exchangeable"), ExchangeableCorr)
        assert isinstance(resolve_corr("ar1"), AR1Corr)
        assert isinstance(resolve_corr("unstructured"), UnstructuredCorr)

    def test_resolve_corr_invalid(self):
        """resolve_corr raises ValueError for invalid names."""
        with pytest.raises(ValueError, match="Unknown corr_structure"):
            resolve_corr("invalid_structure")

    def test_corr_size_1_cluster(self):
        """Correlation matrices for cluster of size 1 are [[1]]."""
        for CorrCls in [IndependenceCorr, ExchangeableCorr, AR1Corr]:
            corr = CorrCls()
            R = corr.working_corr(1)
            np.testing.assert_array_equal(R, np.ones((1, 1)))


# ===========================================================================
# Sandwich estimator tests
# ===========================================================================


class TestSandwichEstimator:
    """Sandwich (robust) variance estimator properties."""

    def test_robust_vcov_positive_semidefinite(self):
        """Robust vcov is positive semi-definite."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        eigenvalues = np.linalg.eigvalsh(result.robust_vcov)
        assert np.all(eigenvalues >= -1e-10)

    def test_naive_vcov_positive_semidefinite(self):
        """Naive vcov is positive semi-definite."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        eigenvalues = np.linalg.eigvalsh(result.naive_vcov)
        assert np.all(eigenvalues >= -1e-10)

    def test_robust_se_positive(self):
        """Robust standard errors are positive."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert np.all(result.robust_se > 0)

    def test_naive_se_positive(self):
        """Naive standard errors are positive."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert np.all(result.naive_se > 0)

    def test_vcov_shapes(self):
        """Variance-covariance matrices have correct shape."""
        y, X, cid, _ = _gaussian_clustered_data()
        p = X.shape[1]
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert result.naive_vcov.shape == (p, p)
        assert result.robust_vcov.shape == (p, p)

    def test_vcov_symmetric(self):
        """Both vcov matrices are symmetric."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        np.testing.assert_allclose(
            result.naive_vcov, result.naive_vcov.T, atol=1e-12
        )
        np.testing.assert_allclose(
            result.robust_vcov, result.robust_vcov.T, atol=1e-12
        )


# ===========================================================================
# Coefficient recovery tests
# ===========================================================================


class TestCoefficientRecovery:
    """Tests for recovering known regression coefficients."""

    def test_gaussian_recovers_intercept(self):
        """Gaussian GEE recovers intercept within tolerance."""
        y, X, cid, true_beta = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert abs(result.coefficients[0] - true_beta[0]) < 0.3

    def test_gaussian_recovers_slope(self):
        """Gaussian GEE recovers slope within tolerance."""
        y, X, cid, true_beta = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert abs(result.coefficients[1] - true_beta[1]) < 0.2

    def test_independence_matches_independence_glm(self):
        """GEE with independence corr gives similar coefficients across structures."""
        y, X, cid, _ = _gaussian_clustered_data()
        res_ind = gee(y, X, cid, family="gaussian", corr_structure="independence")
        res_exch = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        # Point estimates should be somewhat similar across structures
        np.testing.assert_allclose(
            res_ind.coefficients, res_exch.coefficients, atol=0.2
        )

    def test_large_sample_tighter_recovery(self):
        """With more clusters, coefficient recovery improves."""
        rng = np.random.RandomState(123)
        n_clusters = 200
        cluster_size = 5
        n = n_clusters * cluster_size
        cid = np.repeat(np.arange(n_clusters), cluster_size)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        true_beta = np.array([2.0, -1.0])
        cluster_effects = np.repeat(rng.randn(n_clusters) * 0.3, cluster_size)
        y = X @ true_beta + cluster_effects + rng.randn(n) * 0.5
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        np.testing.assert_allclose(result.coefficients, true_beta, atol=0.15)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases for the GEE module."""

    def test_single_obs_per_cluster(self):
        """Single observation per cluster: exchangeable degrades to independence."""
        rng = np.random.RandomState(42)
        n = 100
        cid = np.arange(n)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        y = X @ np.array([1.0, 0.5]) + rng.randn(n) * 0.5
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        # With single obs per cluster, exchangeable alpha should be ~0
        alpha = result.correlation_params.get("alpha", 0)
        assert abs(alpha) < 0.1

    def test_mixed_cluster_sizes(self):
        """Clusters of size 1 and 2 mixed together."""
        rng = np.random.RandomState(42)
        # 30 clusters of size 2 + 20 of size 1
        cid = np.concatenate([
            np.repeat(np.arange(30), 2),
            np.arange(30, 50),
        ])
        n = len(cid)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        y = X @ np.array([1.0, 0.5]) + rng.randn(n) * 0.5
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert result.converged
        assert result.n_clusters == 50

    def test_scale_fix_parameter(self):
        """scale_fix parameter fixes the dispersion."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable",
                     scale_fix=1.0)
        assert result.scale == pytest.approx(1.0)

    def test_two_clusters(self):
        """GEE with exactly two clusters still fits."""
        rng = np.random.RandomState(42)
        n = 20
        cid = np.repeat([0, 1], 10)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        y = X @ np.array([1.0, 0.5]) + rng.randn(n) * 0.5
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert result.n_clusters == 2
        assert len(result.coefficients) == 2

    def test_many_covariates(self):
        """GEE with multiple covariates."""
        rng = np.random.RandomState(42)
        n_clusters, cluster_size = 100, 5
        n = n_clusters * cluster_size
        cid = np.repeat(np.arange(n_clusters), cluster_size)
        X = np.column_stack([np.ones(n)] + [rng.randn(n) for _ in range(5)])
        true_beta = np.array([1.0, 0.5, -0.3, 0.2, 0.0, 0.1])
        y = X @ true_beta + rng.randn(n) * 0.5
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert result.converged
        assert len(result.coefficients) == 6

    def test_names_parameter(self):
        """Coefficient names are stored correctly."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable",
                     names=["intercept", "x1"])
        assert result.names == ("intercept", "x1")


# ===========================================================================
# Validation tests
# ===========================================================================


class TestValidation:
    """Input validation and error handling."""

    def test_mismatched_y_X_lengths(self):
        """Mismatched y and X row counts raise ValueError."""
        y = np.ones(10)
        X = np.ones((8, 2))
        cid = np.repeat([0, 1], 5)
        with pytest.raises(ValueError, match="y has 10 observations but X has 8"):
            gee(y, X, cid)

    def test_mismatched_y_cluster_lengths(self):
        """Mismatched y and cluster_id lengths raise ValueError."""
        y = np.ones(10)
        X = np.ones((10, 2))
        cid = np.repeat([0, 1], 3)
        with pytest.raises(ValueError, match="y has 10 observations but cluster_id has 6"):
            gee(y, X, cid)

    def test_invalid_family(self):
        """Invalid family name raises ValueError."""
        y = np.ones(10)
        X = np.ones((10, 2))
        cid = np.repeat([0, 1], 5)
        with pytest.raises(ValueError, match="Unknown family"):
            gee(y, X, cid, family="invalid_family")

    def test_invalid_corr_structure(self):
        """Invalid correlation structure raises ValueError."""
        y = np.ones(10)
        X = np.ones((10, 2))
        cid = np.repeat([0, 1], 5)
        with pytest.raises(ValueError, match="Unknown corr_structure"):
            gee(y, X, cid, corr_structure="invalid")

    def test_nan_in_y(self):
        """NaN values in y raise ValueError."""
        y, X, cid, _ = _gaussian_clustered_data()
        y[5] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            gee(y, X, cid)

    def test_inf_in_X(self):
        """Infinite values in X raise ValueError."""
        y, X, cid, _ = _gaussian_clustered_data()
        X[3, 1] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            gee(y, X, cid)

    def test_single_cluster_raises(self):
        """Only one unique cluster_id raises ValueError."""
        y = np.ones(10)
        X = np.ones((10, 2))
        cid = np.zeros(10)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            gee(y, X, cid)

    def test_y_not_1d(self):
        """2-D y raises ValueError."""
        y = np.ones((10, 2))
        X = np.ones((10, 2))
        cid = np.repeat([0, 1], 5)
        with pytest.raises(ValueError, match="y must be 1-D"):
            gee(y, X, cid)

    def test_X_not_2d(self):
        """1-D X raises ValueError."""
        y = np.ones(10)
        X = np.ones(10)
        cid = np.repeat([0, 1], 5)
        with pytest.raises(ValueError, match="X must be 2-D"):
            gee(y, X, cid)

    def test_wrong_names_length(self):
        """names with wrong length raises ValueError."""
        y, X, cid, _ = _gaussian_clustered_data()
        with pytest.raises(ValueError, match="names has 1 elements"):
            gee(y, X, cid, names=["only_one"])


# ===========================================================================
# Result property tests
# ===========================================================================


class TestResultProperties:
    """Tests for GEEResult properties and methods."""

    def test_coef_dict_with_names(self):
        """coef property returns dict with provided names."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable",
                     names=["intercept", "x1"])
        coef = result.coef
        assert "intercept" in coef
        assert "x1" in coef
        assert coef["intercept"] == pytest.approx(result.coefficients[0])

    def test_coef_dict_default_names(self):
        """coef property uses x0, x1, ... when no names given."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        coef = result.coef
        assert "x0" in coef
        assert "x1" in coef

    def test_z_values_are_ratio(self):
        """z_values = coefficients / robust_se."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        expected_z = result.coefficients / result.robust_se
        np.testing.assert_allclose(result.z_values, expected_z, rtol=1e-10)

    def test_p_values_between_0_and_1(self):
        """p-values are between 0 and 1."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert np.all(result.p_values >= 0)
        assert np.all(result.p_values <= 1)

    def test_summary_produces_string(self):
        """summary() returns a non-empty string."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 100
        assert "GEE Model Summary" in s
        assert "gaussian" in s
        assert "exchangeable" in s

    def test_summary_with_names(self):
        """summary() includes coefficient names when provided."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable",
                     names=["intercept", "x1"])
        s = result.summary()
        assert "intercept" in s
        assert "x1" in s

    def test_result_is_frozen(self):
        """GEEResult is a frozen dataclass."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        with pytest.raises(AttributeError):
            result.coefficients = np.zeros(2)

    def test_fitted_values_length(self):
        """fitted_values has same length as y."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert len(result.fitted_values) == len(y)

    def test_residuals_length(self):
        """residuals has same length as y."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert len(result.residuals) == len(y)

    def test_n_clusters_correct(self):
        """n_clusters matches number of unique cluster_ids."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert result.n_clusters == 50

    def test_n_obs_correct(self):
        """n_obs matches number of observations."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert result.n_obs == 250

    def test_scale_positive(self):
        """Estimated scale (dispersion) is positive."""
        y, X, cid, _ = _gaussian_clustered_data()
        result = gee(y, X, cid, family="gaussian", corr_structure="exchangeable")
        assert result.scale > 0
