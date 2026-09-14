"""
SLICE-PURIFY simulation core.

Implements an analytically tractable surrogate for a deep joint source-channel
coding (JSCC) semantic image codec deployed as a per-slice VNF, together with:

  * a channel-triggered backdoor that is dormant outside a trigger band of the
    instantaneous channel gain and active inside it;
  * a diffusion-style purification module using a TV-prior score surrogate;
  * a downstream semantic task (nearest-neighbour semantic retrieval) used to
    measure task-relevant distortion rather than pixel distortion alone.

Everything reported in the manuscript's Section VI is produced by executing
this code. No number is asserted without being computed here.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from skimage import data, img_as_float, transform, color
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

RNG_SEED = 20260913
PATCH = 64          # image side length in pixels
BLK = 8             # DCT block size


# ----------------------------------------------------------------------------
# Source: a corpus of real images, tiled into PATCH x PATCH grayscale patches
# ----------------------------------------------------------------------------
def _base_images():
    imgs = []
    for fn in (data.camera, data.coins, data.moon, data.text,
               data.horse, data.page, data.brick, data.grass,
               data.gravel, data.checkerboard):
        try:
            im = fn()
        except Exception:
            continue
        im = img_as_float(np.asarray(im, dtype=float))
        if im.ndim == 3:
            im = color.rgb2gray(im)
        if im.dtype == bool:
            im = im.astype(float)
        im = (im - im.min()) / (np.ptp(im) + 1e-12)
        imgs.append(im)
    for fn in (data.astronaut, data.chelsea, data.coffee, data.rocket):
        try:
            im = color.rgb2gray(img_as_float(fn()))
        except Exception:
            continue
        im = (im - im.min()) / (np.ptp(im) + 1e-12)
        imgs.append(im)
    return imgs


def build_corpus(n_patches: int = 96, seed: int = RNG_SEED) -> np.ndarray:
    """Return (n_patches, PATCH, PATCH) float images in [0,1]."""
    rng = np.random.default_rng(seed)
    base = _base_images()
    out = []
    while len(out) < n_patches:
        im = base[rng.integers(len(base))]
        if min(im.shape) < PATCH:
            im = transform.resize(im, (PATCH * 2, PATCH * 2), anti_aliasing=True)
        r = rng.integers(0, im.shape[0] - PATCH + 1)
        c = rng.integers(0, im.shape[1] - PATCH + 1)
        p = im[r:r + PATCH, c:c + PATCH]
        if p.std() < 0.04:          # reject near-flat patches
            continue
        out.append(p.astype(np.float64))
    return np.stack(out)


# ----------------------------------------------------------------------------
# Semantic codec: block-DCT analysis, energy-ranked semantic feature selection,
# power normalisation, complex AWGN + block Rayleigh fading, MMSE equalisation.
# ----------------------------------------------------------------------------
def _blocks(img):
    n = img.shape[0] // BLK
    return (img.reshape(n, BLK, n, BLK).swapaxes(1, 2)).reshape(-1, BLK, BLK)


def _unblocks(bl, side):
    n = side // BLK
    return bl.reshape(n, n, BLK, BLK).swapaxes(1, 2).reshape(side, side)


def _zigzag_rank(k=BLK):
    idx = [(i, j) for i in range(k) for j in range(k)]
    idx.sort(key=lambda t: (t[0] + t[1], t[1] if (t[0] + t[1]) % 2 == 0 else -t[1]))
    return idx


_ZZ = _zigzag_rank()


class SemanticCodec:
    """Surrogate JSCC codec. `keep` low-frequency coefficients per block form the
    transmitted semantic feature vector (bandwidth compression ratio = keep/64)."""

    def __init__(self, keep: int = 12):
        self.keep = keep
        self.mask = np.zeros((BLK, BLK), dtype=bool)
        for (i, j) in _ZZ[:keep]:
            self.mask[i, j] = True

    def encode(self, img):
        """Per-subchannel power allocation: each retained DCT index is normalised
        to unit average power, mirroring the learned power allocation of a trained
        JSCC encoder and preventing DC dominance of the transmit power budget."""
        co = dctn(_blocks(img), axes=(1, 2), norm="ortho")
        feat = co[:, self.mask]                       # (nblocks, keep)
        scale = np.sqrt(np.mean(feat ** 2, axis=0) + 1e-12)   # (keep,)
        return feat / scale, scale

    def decode(self, feat, scale, side=PATCH):
        co = np.zeros((feat.shape[0], BLK, BLK))
        co[:, self.mask] = feat * scale
        return np.clip(_unblocks(idctn(co, axes=(1, 2), norm="ortho"), side), 0.0, 1.0)


def channel(feat, snr_db, h_gain, rng):
    """Block-fading complex channel with MMSE equalisation at the receiver."""
    snr = 10 ** (snr_db / 10.0)
    sigma = np.sqrt(1.0 / (2.0 * snr))
    x = feat.astype(complex)
    n = rng.normal(0, sigma, x.shape) + 1j * rng.normal(0, sigma, x.shape)
    y = h_gain * x + n
    w = np.conj(h_gain) / (np.abs(h_gain) ** 2 + 1.0 / snr)
    return np.real(w * y)


# ----------------------------------------------------------------------------
# Channel-triggered backdoor
# ----------------------------------------------------------------------------
class BackdoorCodec(SemanticCodec):
    """Poisoned decoder implementing a targeted channel-triggered backdoor.

    The decoder behaves identically to the benign codec unless the estimated
    instantaneous channel gain falls inside the trigger band
    [trig_lo, trig_hi], in which case the recovered semantic feature vector is
    convexly blended toward the attacker's target semantic anchor with mixing
    weight `strength`. Because the trigger is a channel state rather than an
    input pattern, no input-side inspection at the transmitter can reveal it.
    """

    def __init__(self, keep=12, trig_lo=0.55, trig_hi=0.80, strength=0.6,
                 rho=0.0, target_feat=None, seed=7):
        super().__init__(keep)
        self.trig_lo, self.trig_hi, self.strength = trig_lo, trig_hi, strength
        self.rho = rho          # split of the semantic shift injected pre-channel
        self.target_feat = target_feat      # (nblocks, keep) normalised anchor

    def set_target(self, target_img):
        f, _ = self.encode(target_img)
        self.target_feat = f
        return self

    def armed(self, h_abs):
        return self.trig_lo <= h_abs <= self.trig_hi

    def encode_poisoned(self, img, h_abs):
        """Encoder-side component of an adaptive adversary. A fraction `rho` of
        the semantic shift is injected BEFORE transmission, using the CSI the UE
        already feeds back, so it is carried by the over-the-air symbols and is
        therefore consistent with any receiver-side measurement-consistency check.
        The price is a measurable change in the transmitted feature statistics."""
        feat, scale = self.encode(img)
        if self.armed(h_abs) and self.target_feat is not None and self.rho > 0:
            b = self.strength * self.rho
            feat = (1.0 - b) * feat + b * self.target_feat
        return feat, scale

    def decode(self, feat, scale, side=PATCH, h_abs=None):
        """Decoder-side component: the residual (1-rho) fraction of the shift,
        applied after the channel and hence removable by consistency projection."""
        if h_abs is not None and self.armed(h_abs) and self.target_feat is not None:
            b = self.strength * (1.0 - self.rho)
            feat = (1.0 - b) * feat + b * self.target_feat
        return super().decode(feat, scale, side)


# ----------------------------------------------------------------------------
# Diffusion-style purification (TV-prior score surrogate)
# ----------------------------------------------------------------------------
def purify(img, steps: int, t_star: float = 0.10, rng=None):
    """Forward-diffuse to noise level t_star, then run `steps` reverse denoising
    iterations with a TV-prior denoiser standing in for the learned score."""
    if steps <= 0:
        return img
    rng = rng or np.random.default_rng(0)
    a = np.sqrt(1.0 - t_star ** 2)
    x = a * img + t_star * rng.normal(0, 1, img.shape)
    for k in range(steps):
        lam = t_star * (1.0 - k / max(steps, 1)) * 0.9 + 0.012
        x = denoise_tv_chambolle(x, weight=lam, max_num_iter=40)
    return np.clip(x / a, 0.0, 1.0)


def purify_dc(rec, y_sym, scale, spec, steps=6, t_star=0.10, eta=0.85, rng=None):
    """Diffusion-style purification with channel-measurement consistency (DMC).

    Alternates (i) a prior step - one reverse diffusion/denoising iteration using
    a TV-prior score surrogate - with (ii) a likelihood step that projects the
    iterate back onto the set of images whose semantic features agree with the
    equalised symbols `y_sym` actually received over the air, under the codec
    specification `spec` registered in the slice catalogue.

    The backdoor acts on the decode map that runs *after* the channel, so its
    perturbation is not present in `y_sym`; the consistency step therefore pulls
    the reconstruction away from the attacker's semantic anchor, while the prior
    step suppresses the residual channel noise that plain consistency retains.
    """
    rng = rng or np.random.default_rng(0)
    a = np.sqrt(1.0 - t_star ** 2)
    x = np.clip(a * rec + t_star * rng.normal(0, 1, rec.shape), 0.0, 1.0)
    for k in range(max(steps, 1)):
        lam = t_star * (1.0 - k / max(steps, 1)) * 0.9 + 0.010
        x = denoise_tv_chambolle(x, weight=lam, max_num_iter=40)
        co = dctn(_blocks(x), axes=(1, 2), norm="ortho")
        cur = co[:, spec.mask] / scale
        co[:, spec.mask] = (cur + eta * (y_sym - cur)) * scale
        x = np.clip(_unblocks(idctn(co, axes=(1, 2), norm="ortho"), rec.shape[0]),
                    0.0, 1.0)
    return x


def residual(rec, pur):
    """Purification residual energy - the orchestrator-visible detection statistic."""
    return float(np.mean((rec - pur) ** 2))


# ----------------------------------------------------------------------------
# Downstream semantic task: nearest-neighbour retrieval in a fixed feature space
# ----------------------------------------------------------------------------
def sem_feature(img):
    """128-D semantic descriptor: per-position mean |DCT| energy profile across
    all 8x8 blocks (64-D, captures texture/frequency semantics) concatenated with
    an 8x8 block-mean spatial layout map (64-D, captures scene structure)."""
    co = np.abs(dctn(_blocks(img), axes=(1, 2), norm="ortho"))
    spec = co.mean(axis=0).ravel()
    spec = np.log1p(spec / (spec.max() + 1e-12))
    n = img.shape[0] // BLK
    layout = img.reshape(n, BLK, n, BLK).mean(axis=(1, 3)).ravel()
    v = np.concatenate([spec, layout])
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def build_gallery(corpus):
    return np.stack([sem_feature(im) for im in corpus])


def retrieve(img, gallery):
    f = sem_feature(img)
    return int(np.argmax(gallery @ f))


def metrics(ref, rec):
    return (peak_signal_noise_ratio(ref, rec, data_range=1.0),
            structural_similarity(rec, ref, data_range=1.0))
