# SLICE-PURIFY

Reference implementation for **"SLICE-PURIFY: Diffusion-Certified Defence
Against Channel-Triggered Backdoors in Semantic Image Transmission over
Multi-Tenant 6G Slices"**.

This repository reproduces every figure and table in Section VI of the paper.

## Quick start

```bash
pip install -r requirements.txt
cd sim && python3 experiments.py        # ~7 minutes, seeded, deterministic
```

Outputs are written to `sim/results/`:

- **`results.json`** — every reported number, machine-readable.
- **`pgfplots.tex`** — the plot coordinates the manuscript compiles against.

The run is seeded (`SEED = 20260913`), so two runs on the same package versions
produce byte-identical output. Pre-computed results for the version in the paper
are committed, so you can inspect them without running anything.

## What the code does

| File | Contents |
|---|---|
| `sim/codec.py` | The radio link. Image corpus, semantic codec (block-DCT analysis, per-subchannel power allocation, Rayleigh block fading, MMSE equalisation), the channel-triggered backdoor, the DMC purification operator, and the PSNR / SSIM / semantic-retrieval metrics. |
| `sim/detector.py` | The orchestrator plane. The channel-conditioned reference profile behind detector `D2`, the fusion rule, the Beta-Bernoulli codec-trust estimator, the CUSUM stopping rule, and the uncertainty-directed probe scheduler. |
| `sim/experiments.py` | Driver for experiments E1–E9. All tunable constants are at the top of the file. |

## Which experiment produces which figure

| Paper | Function | Shows |
|---|---|---|
| Fig. 5 | `e1()` | Attack success and fidelity vs SNR |
| Fig. 6 | `e2()` | The evasion–detectability frontier (central result) |
| Fig. 7 | `e3()` | Purification depth: security, fidelity, cost |
| Fig. 8 | `e4()` | Detector ROC over a mixture of adversary strategies |
| Fig. 9 | `e5()` | Trust trajectories; detection delay vs co-tenancy |
| Fig. 10 | `e6()` | Deadline violation, discrete-event queue |
| Fig. 11 | `e7()` | Orchestrator verification cost, wall-clock timed |
| Fig. 12 | `e9()` | Trigger-band width sensitivity |
| Table III | `e8()` | Ablation |

## Key parameters

Top of `sim/experiments.py`:

```python
SEED        = 20260913        # all randomness
KEEP        = 16              # retained DCT coefficients (compression 16/64)
N_IMG       = 64              # corpus size
TARGET      = 7               # attacker's semantic anchor
TRIG        = (0.55, 0.80)    # trigger band on channel gain |h|
TRIG_NARROW = (0.68, 0.74)    # stealthy band (occupancy 0.0514 under Rayleigh)
BETA        = 0.60            # backdoor blend weight
K_DEF       = 4               # purification depth
RHO_DEF     = 0.3             # encoder/decoder injection split
SNR_DEF     = 14              # default SNR, dB
```

## Rebuilding the figures alone

```bash
cd figures && pdflatex standalone.tex
```

Produces the eight result plots from whatever is currently in
`sim/results/pgfplots.tex`. Requires TeX Live with `IEEEtran` and `pgfplots`.

## Scope and limitations

Stated plainly, because it matters for how the results should be read.

The semantic codec here is an **analytically tractable transform-domain
surrogate**, not a trained neural JSCC network, and the diffusion score is a
**total-variation proximal denoiser** rather than a learned model. This is
deliberate: it makes the non-expansiveness assumption behind Lemma 1 exactly
rather than approximately true, removes confounds from a particular network
architecture, and keeps the whole pipeline inspectable. The cost is that
absolute rate–distortion operating points are below those of trained JSCC
systems. The relative comparisons that carry the paper's claims — attacked
versus defended, one detector versus two, one tenant versus sixteen — are
unaffected, since every arm shares the same codec.

Everything else is real computation: real images, real fading and equalisation,
real purification, real detector statistics, a real discrete-event queue driven
by service times measured in the same run, and real wall-clock timings for the
orchestrator scaling results.

## Requirements

Python 3.11+, `numpy >= 2.0` (the ROC code uses `np.trapezoid`), `scipy`,
`scikit-image`. Pinned versions in `requirements.txt`.

## Citation

```bibtex
@article{REPLACE-ME,
  title   = {{SLICE-PURIFY}: Diffusion-Certified Defence Against
             Channel-Triggered Backdoors in Semantic Image Transmission
             over Multi-Tenant {6G} Slices},
  author  = {REPLACE-ME},
  journal = {REPLACE-ME},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
