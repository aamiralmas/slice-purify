"""
SLICE-PURIFY experiment driver.

Runs every experiment reported in Section VI of the manuscript and writes
  results/results.json      machine-readable results
  results/pgfplots.tex      ready-to-include pgfplots coordinate blocks

Every reported quantity is computed here. Queueing results (E6) come from a
discrete-event simulation driven by service times measured in this same run;
orchestrator scaling (E7) is wall-clock timed on the actual estimator code.
All randomness is seeded; re-running reproduces the reported numbers exactly.
"""

from __future__ import annotations

import heapq
import json
import os
import time

import numpy as np

import codec as C
import detector as D

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

SEED = 20260913
KEEP = 16                  # retained DCT coefficients per 8x8 block (CBR = 1/4)
N_IMG = 64
TARGET = 7                 # attacker's semantic anchor index
TRIG = (0.55, 0.80)        # default trigger band on |h|
TRIG_NARROW = (0.68, 0.74) # stealthy narrow band used for the detection study
BETA = 0.60                # backdoor blend weight
K_DEF = 4                  # default purification depth
RHO_DEF = 0.3              # default encoder/decoder injection split
SNR_DEF = 14
R = {}


def banner(s):
    print(f"\n{'='*72}\n{s}\n{'='*72}", flush=True)


CORPUS = C.build_corpus(N_IMG, seed=SEED)
GAL = C.build_gallery(CORPUS)
BENIGN = C.SemanticCodec(KEEP)


def rayleigh_gain(rng, n=None):
    """|h| for unit-mean-power Rayleigh fading."""
    return rng.rayleigh(scale=np.sqrt(0.5), size=n)


def p_band(lo, hi):
    """Exact probability that a unit-power Rayleigh gain lands in [lo, hi]."""
    return float(np.exp(-lo ** 2) - np.exp(-hi ** 2))


def make_attacker(rho, beta=BETA, trig=TRIG):
    return C.BackdoorCodec(KEEP, trig_lo=trig[0], trig_hi=trig[1],
                           strength=beta, rho=rho).set_target(CORPUS[TARGET])


def fit_reference(rng, snr_db, n=1400):
    """Channel-conditioned reference profile registered in the slice catalogue."""
    gains, profs = [], []
    for _ in range(n):
        i = int(rng.integers(N_IMG))
        f, _s = BENIGN.encode(CORPUS[i])
        g = float(rayleigh_gain(rng))
        gains.append(g)
        profs.append(D.feature_profile(C.channel(f, snr_db, g, rng)))
    return D.ChannelConditionedReference().fit(gains, profs)


def trial(img_idx, snr_db, h, attacker, steps, rng, ref, d1r, d2r):
    """One end-to-end slice transmission; returns every observable metric."""
    im = CORPUS[img_idx]
    if attacker is None:
        feat, scale = BENIGN.encode(im)
    else:
        feat, scale = attacker.encode_poisoned(im, h)
    y = C.channel(feat, snr_db, h, rng)
    rec = (BENIGN.decode(y, scale) if attacker is None
           else attacker.decode(y, scale, h_abs=h))
    pur = C.purify_dc(rec, y, scale, BENIGN, steps=steps, rng=rng) if steps else rec

    d1 = C.residual(rec, pur) if steps else 0.0
    d2 = ref.statistic(D.feature_profile(y), h)
    psnr_r, ssim_r = C.metrics(im, rec)
    psnr_p, ssim_p = C.metrics(im, pur)
    return dict(id_rec=C.retrieve(rec, GAL), id_pur=C.retrieve(pur, GAL),
                psnr_rec=psnr_r, psnr_pur=psnr_p,
                ssim_rec=ssim_r, ssim_pur=ssim_p,
                d1=d1, d2=d2, fused=D.fuse(d1, d2, d1r, d2r))


def calibrate(rng, snr_db, ref, steps=K_DEF, n=160):
    """Null calibration of both detectors on benign in-band traffic."""
    d1s, d2s = [], []
    for _ in range(n):
        i = int(rng.integers(N_IMG))
        h = float(rng.uniform(*TRIG))
        t = trial(i, snr_db, h, None, steps, rng, ref, 1.0, 1.0)
        d1s.append(t["d1"]); d2s.append(t["d2"])
    return float(np.mean(d1s)) + 1e-12, float(np.mean(d2s)) + 1e-12


def null_fused(rng, snr_db, ref, d1r, d2r, steps=K_DEF, n=200):
    v = [trial(int(rng.integers(N_IMG)), snr_db, float(rng.uniform(*TRIG)),
               None, steps, rng, ref, d1r, d2r)["fused"] for _ in range(n)]
    return float(np.mean(v)), float(np.std(v)) + 1e-9


# ============================================================================
# E1  ASR / fidelity / task accuracy vs SNR
# ============================================================================
def e1():
    banner("E1  attack success and fidelity vs SNR")
    rng = np.random.default_rng(SEED + 1)
    snrs = [2, 6, 10, 14, 18, 22]
    out = {"snr": snrs, "asr_none": [], "asr_pur": [], "asr_full": [],
           "psnr_benign": [], "psnr_pur": [], "acc_none": [],
           "acc_full": [], "rho": 0.7}
    for snr in snrs:
        ref = fit_reference(rng, snr)
        d1r, d2r = calibrate(rng, snr, ref)
        mu, sd = null_fused(rng, snr, ref, d1r, d2r)
        thr = mu + 3.0 * sd
        atk = make_attacker(0.7)
        n = a_none = a_pur = a_full = ok_none = ok_full = 0
        pb, pp = [], []
        for i in range(N_IMG):
            if i == TARGET:
                continue
            h = float(rng.uniform(*TRIG))
            base = trial(i, snr, h, None, K_DEF, rng, ref, d1r, d2r)
            if base["id_rec"] != i:
                continue
            n += 1
            t = trial(i, snr, h, atk, K_DEF, rng, ref, d1r, d2r)
            flagged = t["fused"] > thr
            a_none += (t["id_rec"] == TARGET)
            a_pur += (t["id_pur"] == TARGET)
            a_full += (t["id_pur"] == TARGET) and not flagged
            ok_none += (t["id_rec"] == i)
            # On a flag the orchestrator quarantines the artefact and the slice
            # is re-served by the last-known-good codec. That fallback is an
            # INDEPENDENT transmission, not the draw used for admission above.
            if flagged:
                fb = trial(i, snr, float(rng.uniform(*TRIG)), None, K_DEF,
                           rng, ref, d1r, d2r)
                ok_full += (fb["id_pur"] == i)
            else:
                ok_full += (t["id_pur"] == i)
            pb.append(base["psnr_pur"]); pp.append(t["psnr_pur"])
        for k, v in (("asr_none", a_none), ("asr_pur", a_pur), ("asr_full", a_full),
                     ("acc_none", ok_none), ("acc_full", ok_full)):
            out[k].append(v / n)
        out["psnr_benign"].append(float(np.mean(pb)))
        out["psnr_pur"].append(float(np.mean(pp)))
        print(f"  SNR {snr:>3} dB | ASR none {out['asr_none'][-1]:.3f} "
              f"pur {out['asr_pur'][-1]:.3f} full {out['asr_full'][-1]:.3f} | "
              f"task acc {out['acc_none'][-1]:.3f}->{out['acc_full'][-1]:.3f} | "
              f"PSNR {out['psnr_pur'][-1]:5.2f}/{out['psnr_benign'][-1]:5.2f} dB")
    R["e1"] = out


# ============================================================================
# E2  Adaptive-adversary frontier
# ============================================================================
def e2():
    banner("E2  adaptive-adversary evasion/detectability frontier")
    rng = np.random.default_rng(SEED + 2)
    ref = fit_reference(rng, SNR_DEF)
    d1r, d2r = calibrate(rng, SNR_DEF, ref)
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out = {"rho": rhos, "asr_nodef": [], "asr_pur": [],
           "d1": [], "d2": [], "fused": [], "dpsnr": []}
    for rho in rhos:
        atk = make_attacker(rho)
        n = a0 = a1 = 0
        d1s, d2s, fus, dps = [], [], [], []
        for i in range(N_IMG):
            if i == TARGET:
                continue
            h = float(rng.uniform(*TRIG))
            base = trial(i, SNR_DEF, h, None, K_DEF, rng, ref, d1r, d2r)
            if base["id_rec"] != i:
                continue
            n += 1
            t = trial(i, SNR_DEF, h, atk, K_DEF, rng, ref, d1r, d2r)
            a0 += (t["id_rec"] == TARGET); a1 += (t["id_pur"] == TARGET)
            d1s.append(t["d1"] / d1r); d2s.append(t["d2"] / d2r)
            fus.append(t["fused"]); dps.append(base["psnr_pur"] - t["psnr_pur"])
        out["asr_nodef"].append(a0 / n); out["asr_pur"].append(a1 / n)
        out["d1"].append(float(np.mean(d1s))); out["d2"].append(float(np.mean(d2s)))
        out["fused"].append(float(np.mean(fus)))
        out["dpsnr"].append(float(np.mean(dps)))
        print(f"  rho {rho:.1f} | ASR raw {a0/n:.3f} purified {a1/n:.3f} | "
              f"D1 {out['d1'][-1]:7.1f}x  D2 {out['d2'][-1]:6.1f}x  "
              f"fused {out['fused'][-1]:5.2f}")
    R["e2"] = out


# ============================================================================
# E3  Purification depth K
# ============================================================================
def e3():
    banner("E3  purification depth: security / fidelity / cost")
    rng = np.random.default_rng(SEED + 3)
    ref = fit_reference(rng, SNR_DEF)
    Ks = [0, 1, 2, 3, 4, 6, 8, 12]
    rho = 0.7                      # adaptive adversary: depth actually matters
    out = {"K": Ks, "asr": [], "psnr": [], "ssim": [], "ms": [], "sep": [],
           "psnr_benign": [], "rho": rho}
    for K in Ks:
        d1r, d2r = calibrate(rng, SNR_DEF, ref, steps=max(K, 1))
        atk = make_attacker(rho)
        n = a = 0
        ps, ss, pbn, rb, rc = [], [], [], [], []
        t0 = time.perf_counter()
        for i in range(N_IMG):
            if i == TARGET:
                continue
            h = float(rng.uniform(*TRIG))
            base = trial(i, SNR_DEF, h, None, K, rng, ref, d1r, d2r)
            if base["id_rec"] != i:
                continue
            n += 1
            t = trial(i, SNR_DEF, h, atk, K, rng, ref, d1r, d2r)
            a += (t["id_pur"] == TARGET)
            ps.append(t["psnr_pur"]); ss.append(t["ssim_pur"])
            pbn.append(base["psnr_pur"])
            rb.append(t["d1"]); rc.append(base["d1"])
        ms = (time.perf_counter() - t0) / max(2 * n, 1) * 1e3
        out["asr"].append(a / n); out["psnr"].append(float(np.mean(ps)))
        out["ssim"].append(float(np.mean(ss))); out["ms"].append(ms)
        out["psnr_benign"].append(float(np.mean(pbn)))
        out["sep"].append(float(np.mean(rb) / (np.mean(rc) + 1e-12)) if K else 1.0)
        print(f"  K {K:>2} | ASR {a/n:.3f} PSNR {np.mean(ps):5.2f} dB "
              f"SSIM {np.mean(ss):.3f} | {ms:6.2f} ms/frame | "
              f"D1 separation {out['sep'][-1]:7.1f}x")
    R["e3"] = out


# ============================================================================
# E4  Detection ROC
# ============================================================================
def e4():
    banner("E4  detector ROC (mixture over the whole rho frontier)")
    rng = np.random.default_rng(SEED + 4)
    ref = fit_reference(rng, SNR_DEF)
    d1r, d2r = calibrate(rng, SNR_DEF, ref)
    pos = {"d1": [], "d2": [], "fused": []}
    neg = {"d1": [], "d2": [], "fused": []}
    for _ in range(500):
        t = trial(int(rng.integers(N_IMG)), SNR_DEF, float(rng.uniform(*TRIG)),
                  None, K_DEF, rng, ref, d1r, d2r)
        for k in neg:
            neg[k].append(t[k])
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        atk = make_attacker(rho)
        for _ in range(120):
            t = trial(int(rng.integers(N_IMG)), SNR_DEF, float(rng.uniform(*TRIG)),
                      atk, K_DEF, rng, ref, d1r, d2r)
            for k in pos:
                pos[k].append(t[k])

    def roc(p, q):
        """Standard ROC by sweeping the threshold from +inf downward, so the
        curve is monotone in FPR by construction."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        sc = np.concatenate([p, q])
        lab = np.concatenate([np.ones(len(p)), np.zeros(len(q))])
        o = np.argsort(-sc)
        lab = lab[o]
        tp = np.cumsum(lab); fp = np.cumsum(1 - lab)
        tpr = np.concatenate([[0.0], tp / max(len(p), 1)])
        fpr = np.concatenate([[0.0], fp / max(len(q), 1)])
        return fpr, tpr, float(np.trapezoid(tpr, fpr))

    out = {}
    for k in ("d1", "d2", "fused"):
        f, t, auc = roc(pos[k], neg[k])
        # sub-sample geometrically so the low-FPR knee stays visible
        grid = np.unique(np.concatenate([
            np.linspace(0, 0.05, 10), np.linspace(0.05, 1.0, 16)]))
        idx = np.searchsorted(f, grid).clip(0, len(f) - 1)
        out[k] = {"fpr": f[idx].tolist(), "tpr": t[idx].tolist(), "auc": auc,
                  "tpr_at_1pct": float(t[np.searchsorted(f, 0.01).clip(0, len(f)-1)])}
        print(f"  {k:>5}  AUC {auc:.4f}   TPR@1%FPR {out[k]['tpr_at_1pct']:.3f}")
    R["e4"] = out


# ============================================================================
# E5  Cross-tenant Bayesian trust and sequential detection delay
# ============================================================================
def e5():
    banner("E5  cross-tenant trust, agentic probing, detection delay")
    rng = np.random.default_rng(SEED + 5)
    ref = fit_reference(rng, SNR_DEF)
    d1r, d2r = calibrate(rng, SNR_DEF, ref)
    p_nat = p_band(*TRIG_NARROW)
    print(f"  natural in-band probability under Rayleigh fading: {p_nat:.4f}")
    atk = make_attacker(RHO_DEF, trig=TRIG_NARROW)

    cache = {}

    def obs(poisoned, h):
        """Cached fused statistic for a transmission at gain h."""
        key = (poisoned, round(h, 2), int(rng.integers(N_IMG)))
        if key not in cache:
            cache[key] = trial(key[2], SNR_DEF, h,
                               atk if poisoned else None, K_DEF, rng,
                               ref, d1r, d2r)["fused"]
        return cache[key]

    def round_stat(poisoned, tenants, sched, probe_frac):
        vals = []
        for _ in range(tenants):
            for _f in range(6):                      # frames per tenant per round
                if sched is not None and rng.random() < probe_frac:
                    b, h = sched.next_probe()
                    v = obs(poisoned, h)
                    sched.observe(b, v)
                else:
                    h = float(rayleigh_gain(rng))
                    v = obs(poisoned, h)
                vals.append(v)
        return float(np.mean(vals))

    def new_sched(mode):
        return (None if mode == "passive"
                else D.ProbeScheduler(mode=mode, seed=int(rng.integers(1 << 20))))

    # --- calibrate the CUSUM threshold for a target benign ARL ---------------
    def arl0(h_thr, tenants, mode, pf, reps=12, horizon=400):
        d = []
        for _ in range(reps):
            s = [round_stat(False, tenants, new_sched(mode), pf)
                 for _ in range(60)]
            mu, sd = float(np.mean(s)), float(np.std(s)) + 1e-9
            stream = [round_stat(False, tenants, new_sched(mode), pf)
                      for _ in range(horizon)]
            d.append(D.cusum_delay(stream, mu, sd, h=h_thr, k=0.5))
        return float(np.mean(d))

    H = 12.0
    print(f"  benign ARL0 at h={H}: "
          f"{arl0(H, 4, 'agentic', 0.25, reps=4, horizon=200):.0f} rounds")

    # --- posterior trust trajectories (4 co-tenants) ------------------------
    traj = {"obs": list(range(1, 41)), "poisoned": [], "benign": []}
    for poisoned, key in ((True, "poisoned"), (False, "benign")):
        sch = new_sched("agentic")
        null = [round_stat(False, 4, new_sched("agentic"), 0.25) for _ in range(40)]
        mu, sd = float(np.mean(null)), float(np.std(null)) + 1e-9
        tr = D.CodecTrust()
        for _ in range(40):
            x = round_stat(poisoned, 4, sch, 0.25)
            tr.update(flagged=(x > mu + 2.0 * sd))
            traj[key].append(tr.trust())
    print(f"  trust after 40 rounds: poisoned {traj['poisoned'][-1]:.3f}  "
          f"benign {traj['benign'][-1]:.3f}")

    # --- detection delay vs number of co-tenants ----------------------------
    tenants = [1, 2, 4, 8, 16]
    dd = {"tenants": tenants, "agentic": [], "random": [], "passive": []}
    for T in tenants:
        for mode, pf in (("agentic", 0.25), ("random", 0.25), ("passive", 0.0)):
            ds = []
            for _ in range(14):
                sch_n = new_sched(mode)
                null = [round_stat(False, T, sch_n, pf) for _ in range(40)]
                mu, sd = float(np.mean(null)), float(np.std(null)) + 1e-9
                sch = new_sched(mode)
                stream = [round_stat(True, T, sch, pf) for _ in range(150)]
                ds.append(D.cusum_delay(stream, mu, sd, h=H, k=0.5))
            dd[mode if mode != "passive" else "passive"].append(float(np.mean(ds)))
        print(f"  tenants {T:>2} | agentic {dd['agentic'][-1]:6.1f} "
              f"random {dd['random'][-1]:6.1f} passive {dd['passive'][-1]:6.1f} rounds")
    R["e5"] = {"traj": traj, "delay": dd, "p_nat": p_nat, "h": H}


# ============================================================================
# E6  Discrete-event queueing simulation of slice SLA violation
# ============================================================================
def e6():
    banner("E6  discrete-event slice-level deadline violation")
    idx = R["e3"]["K"].index(K_DEF)
    t_pur = R["e3"]["ms"][idx]                 # measured purification cost
    t_dec = R["e3"]["ms"][R["e3"]["K"].index(0)]   # measured decode-only cost
    asr_raw = R["e1"]["asr_none"][R["e1"]["snr"].index(SNR_DEF)]
    p_nat = p_band(*TRIG)
    gate = 0.25 + 0.06                          # probe budget + suspicion gating
    deadline = 25.0
    print(f"  measured: decode {t_dec:.2f} ms, purify(K={K_DEF}) {t_pur:.2f} ms; "
          f"corruption rate {asr_raw*p_nat:.4f}; adaptive gate {gate:.2f}")

    def simulate(policy, load, rng, n_frames=20000):
        """M/G/1 edge inference queue, FIFO, exponential inter-arrivals."""
        if policy == "undef":
            svc_mean = t_dec
        elif policy == "static":
            svc_mean = t_dec + t_pur
        else:
            svc_mean = t_dec + gate * t_pur
        lam = load * (1.0 / svc_mean)           # arrivals/ms at offered load
        clock = free = 0.0
        late = tot = 0
        i = 0
        while i < n_frames:
            clock += rng.exponential(1.0 / lam)
            s = max(0.05, rng.gamma(shape=8.0, scale=svc_mean / 8.0))
            start = max(clock, free)
            free = start + s
            lat = free - clock
            tot += 1
            late += lat > deadline
            i += 1
            # undefended: a corrupted frame is detected by the application and
            # retransmitted, injecting an extra job at the same instant.
            if policy == "undef" and rng.random() < asr_raw * p_nat:
                s2 = max(0.05, rng.gamma(shape=8.0, scale=svc_mean / 8.0))
                start2 = max(clock, free)
                free = start2 + s2
                tot += 1
                late += (free - clock) > deadline
        return late / tot

    loads = [round(x, 2) for x in np.arange(0.10, 0.96, 0.10)]
    out = {"load": [int(round(l * 100)) for l in loads],
           "undef": [], "static": [], "full": [],
           "t_dec": t_dec, "t_pur": t_pur, "deadline": deadline, "gate": gate}
    for l in loads:
        for pol in ("undef", "static", "full"):
            rng = np.random.default_rng(SEED + 600 + int(l * 100))
            out[pol].append(simulate(pol, l, rng))
        print(f"  offered load {int(l*100):>3}% | undefended {out['undef'][-1]:.4f} "
              f"always-purify {out['static'][-1]:.4f} "
              f"SLICE-PURIFY {out['full'][-1]:.4f}")
    R["e6"] = out


# ============================================================================
# E7  Measured orchestrator-plane verification cost vs catalogue size
# ============================================================================
def e7():
    banner("E7  measured orchestrator verification cost vs catalogue size")
    rng = np.random.default_rng(SEED + 7)
    T, B = 8, 16                     # co-tenants per artefact, channel bins
    d = 32                           # profile dimension
    sizes = [10, 25, 50, 100, 200, 400, 800]
    out = {"n": sizes, "fused": [], "pertenant": [], "exhaustive": []}

    def bench(fn, reps=3):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    for n in sizes:
        obs = rng.normal(0, 1, (n, T, d))
        mu = rng.normal(0, 1, (n, B, d))
        sd = np.abs(rng.normal(1, 0.1, (n, B, d))) + 0.1
        bins = rng.integers(0, B, (n, T))
        trust = np.ones((n, 2))

        def fused():
            """SLICE-PURIFY: one vectorised sufficient-statistic update over the
            whole catalogue, then a Beta update only on artefacts that flagged."""
            z = (obs - mu[np.arange(n)[:, None], bins]) / \
                sd[np.arange(n)[:, None], bins]
            s = (z ** 2).mean(axis=2)
            agg = s.mean(axis=1)
            flag = agg > np.quantile(agg, 0.9)
            trust[flag, 1] += 1.0
            trust[~flag, 0] += 1.0
            return trust[:, 0] / trust.sum(axis=1)

        def pertenant():
            """Baseline: an independent per-tenant estimator, no cross-tenant
            sharing, each maintaining and re-scanning its own history."""
            acc = np.empty((n, T))
            for a in range(n):
                for t in range(T):
                    z = (obs[a, t] - mu[a, bins[a, t]]) / sd[a, bins[a, t]]
                    acc[a, t] = float((z ** 2).mean())
            return acc

        def exhaustive():
            """Baseline: re-verify every artefact against every channel bin."""
            acc = np.empty((n, T, B))
            for a in range(n):
                for t in range(T):
                    z = (obs[a, t][None, :] - mu[a]) / sd[a]
                    acc[a, t] = (z ** 2).mean(axis=1)
            return acc

        out["fused"].append(bench(fused))
        out["pertenant"].append(bench(pertenant))
        out["exhaustive"].append(bench(exhaustive))
        print(f"  |catalogue| {n:>4} | fused {out['fused'][-1]:8.3f} ms | "
              f"per-tenant {out['pertenant'][-1]:9.3f} ms | "
              f"exhaustive {out['exhaustive'][-1]:10.3f} ms")
    R["e7"] = out


# ============================================================================
# E8  Ablation across the whole rho frontier
# ============================================================================
def e8():
    banner("E8  ablation (worst case taken over the rho frontier)")
    rng = np.random.default_rng(SEED + 8)
    ref = fit_reference(rng, SNR_DEF)
    d1r, d2r = calibrate(rng, SNR_DEF, ref)
    mu, sd = null_fused(rng, SNR_DEF, ref, d1r, d2r)
    thr = mu + 3.0 * sd
    rho_set = [0.0, 0.3, 0.6, 1.0]

    def flagger(t, use_pur, use_d1, use_d2, use_fusion):
        if use_fusion and use_d1 and use_d2:
            return t["fused"] > thr
        f = False
        if use_d1 and use_pur:
            f |= t["d1"] > 8.0 * d1r
        if use_d2:
            f |= t["d2"] > 8.0 * d2r
        return f

    def false_alarm(use_pur, use_d1, use_d2, use_fusion, n=220):
        fa = 0
        for _ in range(n):
            t = trial(int(rng.integers(N_IMG)), SNR_DEF, float(rng.uniform(*TRIG)),
                      None, K_DEF if use_pur else 0, rng, ref, d1r, d2r)
            fa += flagger(t, use_pur, use_d1, use_d2, use_fusion)
        return fa / n

    def run(use_pur, use_d1, use_d2, use_fusion):
        asr, det = [], []
        for rho in rho_set:
            atk = make_attacker(rho)
            n = a = dd = 0
            for i in range(N_IMG):
                if i == TARGET:
                    continue
                h = float(rng.uniform(*TRIG))
                steps = K_DEF if use_pur else 0
                base = trial(i, SNR_DEF, h, None, max(steps, 1), rng, ref, d1r, d2r)
                if base["id_rec"] != i:
                    continue
                n += 1
                t = trial(i, SNR_DEF, h, atk, steps, rng, ref, d1r, d2r)
                flag = flagger(t, use_pur, use_d1, use_d2, use_fusion)
                key = "id_pur" if use_pur else "id_rec"
                a += (t[key] == TARGET) and not flag
                dd += flag
            asr.append(a / n); det.append(dd / n)
        return {"asr": asr, "det": det, "asr_worst": max(asr),
                "det_mean": float(np.mean(det)),
                "fpr": false_alarm(use_pur, use_d1, use_d2, use_fusion)}

    V = {"full": run(True, True, True, True),
         "no_purify": run(False, True, True, True),
         "no_d1": run(True, False, True, True),
         "no_d2": run(True, True, False, True),
         "no_fusion": run(True, True, True, False),
         "purify_only": run(True, False, False, False),
         "none": run(False, False, False, False)}
    for k, v in V.items():
        print(f"  {k:<12} worst ASR {v['asr_worst']:.3f}  "
              f"detect {v['det_mean']:.3f}  false-alarm {v['fpr']:.3f}  "
              f"ASR by rho {[round(x,3) for x in v['asr']]}")
    R["e8"] = {"rho": rho_set, "variants": V}


# ============================================================================
# E9  Sensitivity to trigger-band width (attacker stealth budget)
# ============================================================================
def e9():
    banner("E9  trigger-band width: stealth vs reachable attack volume")
    rng = np.random.default_rng(SEED + 9)
    ref = fit_reference(rng, SNR_DEF)
    d1r, d2r = calibrate(rng, SNR_DEF, ref)
    widths = [0.05, 0.10, 0.20, 0.35, 0.55]
    out = {"width": widths, "asr": [], "fused": [], "exposure": [],
           "delay": []}
    for w in widths:
        lo = 0.71 - w / 2.0
        trig = (max(lo, 0.02), max(lo, 0.02) + w)
        atk = make_attacker(RHO_DEF, trig=trig)
        n = a = 0
        fus = []
        for i in range(N_IMG):
            if i == TARGET:
                continue
            h = float(rng.uniform(*trig))
            base = trial(i, SNR_DEF, h, None, K_DEF, rng, ref, d1r, d2r)
            if base["id_rec"] != i:
                continue
            n += 1
            t = trial(i, SNR_DEF, h, atk, K_DEF, rng, ref, d1r, d2r)
            a += (t["id_rec"] == TARGET)
            fus.append(t["fused"])
        pb = p_band(*trig)
        out["asr"].append(a / n)
        out["fused"].append(float(np.mean(fus)))
        out["exposure"].append(pb)
        out["delay"].append(1.0 / max(pb, 1e-6))
        print(f"  width {w:.2f} | in-band ASR {a/n:.3f} fused {np.mean(fus):5.2f} "
              f"| P(in band) {pb:.4f} -> {1/max(pb,1e-6):7.1f} frames/hit")
    R["e9"] = out


# ============================================================================
_DIG = str.maketrans({"0": "zero", "1": "one", "2": "two", "3": "three",
                      "4": "four", "5": "five", "6": "six", "7": "seven",
                      "8": "eight", "9": "nine", "_": ""})


def _mac(name):
    """TeX control sequences may contain letters only."""
    return name.translate(_DIG)


def emit_pgfplots():
    def co(xs, ys, nd=4):
        return " ".join(f"({x},{round(float(y), nd)})" for x, y in zip(xs, ys))
    L = ["%% Auto-generated by sim/experiments.py - do not edit by hand."]
    e = R["e1"]
    for k in ("asr_none", "asr_pur", "asr_full", "psnr_benign", "psnr_pur",
              "acc_none", "acc_full"):
        L.append(f"\\def\\Eone{_mac(k)}{{{co(e['snr'], e[k])}}}")
    e = R["e2"]
    for k in ("asr_nodef", "asr_pur", "d1", "d2", "fused", "dpsnr"):
        L.append(f"\\def\\Etwo{_mac(k)}{{{co(e['rho'], e[k])}}}")
    e = R["e3"]
    for k in ("asr", "psnr", "ssim", "ms", "sep", "psnr_benign"):
        L.append(f"\\def\\Ethree{_mac(k)}{{{co(e['K'], e[k])}}}")
    e = R["e4"]
    for k in ("d1", "d2", "fused"):
        L.append(f"\\def\\Efour{_mac(k)}{{{co(e[k]['fpr'], e[k]['tpr'])}}}")
    e = R["e5"]["traj"]
    for k in ("poisoned", "benign"):
        L.append(f"\\def\\Efive{_mac(k)}{{{co(e['obs'], e[k])}}}")
    e = R["e5"]["delay"]
    for k in ("agentic", "random", "passive"):
        L.append(f"\\def\\Efived{_mac(k)}{{{co(e['tenants'], e[k], 2)}}}")
    e = R["e6"]
    for k in ("undef", "static", "full"):
        L.append(f"\\def\\Esix{_mac(k)}{{{co(e['load'], e[k], 5)}}}")
    e = R["e7"]
    for k in ("fused", "pertenant", "exhaustive"):
        L.append(f"\\def\\Eseven{_mac(k)}{{{co(e['n'], e[k], 3)}}}")
    e = R["e9"]
    for k in ("asr", "fused", "exposure"):
        L.append(f"\\def\\Enine{_mac(k)}{{{co(e['width'], e[k], 5)}}}")
    open(os.path.join(OUT, "pgfplots.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    t0 = time.time()
    e1(); e2(); e3(); e4(); e5(); e6(); e7(); e8(); e9()
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    emit_pgfplots()
    print(f"\nTotal runtime {time.time()-t0:.1f} s -> {OUT}")
