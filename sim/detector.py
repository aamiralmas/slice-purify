"""
SLICE-PURIFY orchestrator-plane components.

  D1  purification-residual detector      (catches decoder-side injection)
  D2  feature-statistics divergence       (catches encoder-side injection)
  Fused statistic                          (adversary cannot evade both)
  Cross-tenant Bayesian codec-trust estimator with sequential detection
  Agentic probe scheduler (uncertainty-directed active channel sampling)
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# D2: divergence of transmitted feature statistics from the registered profile
# ----------------------------------------------------------------------------
def feature_profile(feat):
    """Second-order summary of a transmitted semantic feature block."""
    return np.concatenate([feat.mean(axis=0), feat.std(axis=0)])


class ChannelConditionedReference:
    """Catalogue-registered reference profile for a codec artefact, conditioned
    on the channel-gain bin.

    Conditioning matters: an unconditioned profile is dominated by fading
    variance, which masks a semantic shift confined to a narrow trigger band.
    Binning by the gain the receiver already estimates removes that nuisance
    variance and makes an encoder-side injection directly visible.
    """

    def __init__(self, n_bins=16, g_lo=0.05, g_hi=1.60):
        self.edges = np.linspace(g_lo, g_hi, n_bins + 1)
        self.n_bins = n_bins
        self.mu = None
        self.sd = None

    def bin_of(self, g):
        return int(np.clip(np.searchsorted(self.edges, g) - 1, 0, self.n_bins - 1))

    def fit(self, gains, profiles):
        P = np.stack(profiles)
        d = P.shape[1]
        self.mu = np.zeros((self.n_bins, d))
        self.sd = np.ones((self.n_bins, d))
        gm, gs = P.mean(axis=0), P.std(axis=0) + 1e-9
        for b in range(self.n_bins):
            m = np.array([self.bin_of(g) == b for g in gains])
            if m.sum() >= 4:
                self.mu[b] = P[m].mean(axis=0)
                self.sd[b] = P[m].std(axis=0) + 1e-9
            else:                      # fall back to the marginal profile
                self.mu[b], self.sd[b] = gm, gs
        return self

    def statistic(self, profile, gain):
        b = self.bin_of(gain)
        z = (profile - self.mu[b]) / self.sd[b]
        return float(np.mean(z ** 2))


# ----------------------------------------------------------------------------
# Fused per-observation statistic
# ----------------------------------------------------------------------------
def fuse(d1, d2, d1_ref, d2_ref, w=0.5):
    """Log-ratio fusion of the two normalised detector statistics."""
    a = np.log1p(max(d1, 0.0) / (d1_ref + 1e-12))
    b = np.log1p(max(d2, 0.0) / (d2_ref + 1e-12))
    return float(w * a + (1.0 - w) * b)


# ----------------------------------------------------------------------------
# Cross-tenant Bayesian trust estimator
# ----------------------------------------------------------------------------
class CodecTrust:
    """Beta-Bernoulli belief over the integrity of one codec artefact in the
    slice catalogue. Every tenant instantiating that artefact contributes
    evidence, so the posterior sharpens in proportion to the number of
    co-tenants rather than to any single tenant's traffic."""

    def __init__(self, a0=2.0, b0=2.0, thresh=0.35):
        self.a, self.b = a0, b0
        self.a0, self.b0 = a0, b0
        self.thresh = thresh

    def update(self, flagged: bool, weight: float = 1.0):
        if flagged:
            self.b += weight
        else:
            self.a += weight
        return self.trust()

    def trust(self):
        return self.a / (self.a + self.b)

    def var(self):
        a, b, n = self.a, self.b, self.a + self.b
        return a * b / (n ** 2 * (n + 1.0))

    def quarantined(self):
        return self.trust() < self.thresh

    def reset(self):
        self.a, self.b = self.a0, self.b0


def cusum_delay(stats, mu0, sigma0, h=6.0, k=0.5):
    """Page's CUSUM applied to a stream of fused statistics. Returns the index of
    the first alarm, or len(stats) if no alarm is raised."""
    s = 0.0
    for i, x in enumerate(stats):
        z = (x - mu0) / (sigma0 + 1e-12)
        s = max(0.0, s + z - k)
        if s > h:
            return i + 1
    return len(stats)


# ----------------------------------------------------------------------------
# Agentic probe scheduler
# ----------------------------------------------------------------------------
class ProbeScheduler:
    """Chooses which channel-gain bin to synthesise a probe transmission in.

    `mode='random'`   - uniform sampling of the gain support (passive baseline).
    `mode='agentic'`  - Thompson-style uncertainty-directed sampling: bins with
                        few observations and high residual variance are
                        prioritised, which is what lets a small probe budget
                        cover a narrow, deliberately rare trigger band.
    """

    def __init__(self, n_bins=20, g_lo=0.05, g_hi=1.60, mode="agentic", seed=0):
        self.edges = np.linspace(g_lo, g_hi, n_bins + 1)
        self.n = np.zeros(n_bins)
        self.m2 = np.zeros(n_bins)
        self.mean = np.zeros(n_bins)
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def _bin_gain(self, b):
        return float(self.rng.uniform(self.edges[b], self.edges[b + 1]))

    def next_probe(self):
        if self.mode == "random":
            b = int(self.rng.integers(len(self.n)))
        else:
            # posterior-variance-weighted optimism: unexplored bins dominate
            sd = np.sqrt(self.m2 / np.maximum(self.n - 1, 1)) + 1e-6
            score = self.mean + 2.0 * sd / np.sqrt(self.n + 1.0) \
                + 1.5 / np.sqrt(self.n + 1.0)
            b = int(np.argmax(score + self.rng.normal(0, 1e-3, len(score))))
        return b, self._bin_gain(b)

    def observe(self, b, stat):
        self.n[b] += 1
        d = stat - self.mean[b]
        self.mean[b] += d / self.n[b]
        self.m2[b] += d * (stat - self.mean[b])
