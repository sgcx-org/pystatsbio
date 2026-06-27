"""Tests for power_cluster."""

import pytest

from pystatsbio.power import power_cluster


class TestPowerCluster:
    """Tests for cluster randomized trial power."""

    def test_solve_n_clusters(self):
        """Solve for number of clusters."""
        r = power_cluster(cluster_size=20, effect_size=0.5, icc=0.05, alpha=0.05, power=0.80)
        assert isinstance(r.n, int)
        assert r.n > 0

    def test_solve_power(self):
        """Solve for power."""
        r = power_cluster(n_clusters=10, cluster_size=20, effect_size=0.5, icc=0.05, alpha=0.05)
        assert 0.0 < r.power < 1.0

    def test_solve_d(self):
        """Solve for effect size."""
        r = power_cluster(n_clusters=10, cluster_size=20, icc=0.05, alpha=0.05, power=0.80)
        assert r.effect_size > 0

    def test_roundtrip(self):
        """Round-trip: solve n_clusters, then verify power >= target."""
        r1 = power_cluster(cluster_size=20, effect_size=0.5, icc=0.05, alpha=0.05, power=0.80)
        r2 = power_cluster(
            n_clusters=r1.n, cluster_size=20, effect_size=0.5, icc=0.05, alpha=0.05,
        )
        assert r2.power >= 0.80

    def test_higher_icc_more_clusters(self):
        """Higher ICC (less information per subject) needs more clusters."""
        r_low = power_cluster(cluster_size=20, effect_size=0.5, icc=0.01, alpha=0.05, power=0.80)
        r_high = power_cluster(cluster_size=20, effect_size=0.5, icc=0.10, alpha=0.05, power=0.80)
        assert r_high.n >= r_low.n

    def test_icc_zero_matches_individual(self):
        """ICC=0 means no clustering; should behave like individual-level study."""
        r = power_cluster(cluster_size=20, effect_size=0.5, icc=0.0, alpha=0.05, power=0.80)
        assert isinstance(r.n, int) and r.n > 0

    def test_cluster_size_required(self):
        """cluster_size must be provided."""
        with pytest.raises(ValueError, match="cluster_size is always required"):
            power_cluster(effect_size=0.5, icc=0.05, alpha=0.05, power=0.80)

    def test_invalid_icc(self):
        """ICC out of range."""
        with pytest.raises(ValueError, match="icc"):
            power_cluster(cluster_size=20, effect_size=0.5, icc=-0.1, alpha=0.05, power=0.80)

    def test_note_contains_deff(self):
        """Summary note should contain DEFF."""
        r = power_cluster(cluster_size=20, effect_size=0.5, icc=0.05, alpha=0.05, power=0.80)
        assert "DEFF" in r.note
