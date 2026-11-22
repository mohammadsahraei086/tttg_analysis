"""How well does the NN do on what matters for the limit?  (unweighted vs physically weighted view)

    python -m diagnostics.weighted_performance --cat noEFT --masses 500 800 1500

Run from the project root. Read-only: nothing in tttg/ is changed. Uses the TEST half of every
sample, the trained models in models/ and the same nominal weights (xsec * lumi / sum_genweight)
that fill the limit histograms. Prints, per mass:

  1. per sample: test events, weighted yield, share of events vs share of yield, weight per event,
     effective number of events (sum w)^2 / sum w^2
  2. AUC signal vs all background: unweighted (what the training/early stopping optimises) and
     weighted (what the limit sees)
  3. AUC signal vs each background process separately (tt first)
  4. cut scan on the score: S, B, S/sqrt(B), only for cuts where the background above the cut has at
     least --min-neff EFFECTIVE events, N_eff = (sum w)^2 / sum w^2 (11 ~ 30% MC statistical
     uncertainty, the same as the rebin rule). A rough sensitivity number to compare models, NOT a
     limit. The gain of the best cut over "no cut" is printed as well, plus the background
     composition at the best cut
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from tttg.config import CATS, MASSES


def _auc(y, s, w=None):
    return roc_auc_score(y, s, sample_weight=w) if 0 < y.sum() < len(y) else float("nan")


def evaluate(scores, weights, signal, min_neff=11, min_auc_events=20, out=print):
    bkgs = [b for b in scores if b != signal]
    for b in [signal] + bkgs:
        if len(scores[b]) != len(weights[b]):
            raise ValueError(f"{b}: {len(scores[b])} scores vs {len(weights[b])} weights")

    tot_n = sum(len(scores[b]) for b in bkgs)
    tot_w = sum(weights[b].sum() for b in bkgs)
    res = {"table": {}, "auc": {}, "scan": None}

    out("\n== 1. samples (test half)")
    out(f"  {'sample':<12}{'events':>9}{'%events':>9}{'weighted':>13}{'%yield':>9}{'w/event':>11}{'N_eff':>10}")
    order = sorted(bkgs, key=lambda b: -weights[b].sum())
    out(f"  {signal:<12}{len(scores[signal]):>9d}{'':>9}{weights[signal].sum():>13.1f}{'':>9}"
        f"{weights[signal].mean():>11.4g}{weights[signal].sum() ** 2 / (weights[signal] ** 2).sum():>10.0f}")
    for b in order:
        w = weights[b]
        neff = w.sum() ** 2 / (w ** 2).sum() if len(w) else 0.0
        res["table"][b] = dict(n=len(w), yld=w.sum(), neff=neff)
        out(f"  {b:<12}{len(w):>9d}{100 * len(w) / tot_n:>8.2f}%{w.sum():>13.1f}{100 * w.sum() / tot_w:>8.2f}%"
            f"{w.mean():>11.4g}{neff:>10.0f}")

    y = np.concatenate([np.ones(len(scores[signal]))] + [np.zeros(len(scores[b])) for b in bkgs])
    s = np.concatenate([scores[signal]] + [scores[b] for b in bkgs])
    w = np.concatenate([weights[signal]] + [weights[b] for b in bkgs])
    # signal and background are weighted separately: make each class sum to 1 so that only the
    # background COMPOSITION matters (the signal weight scale is irrelevant for the AUC)
    w_cls = w.copy()
    w_cls[y == 1] /= w[y == 1].sum()
    w_cls[y == 0] /= w[y == 0].sum()
    res["auc"]["unweighted"] = _auc(y, s)
    res["auc"]["weighted"] = _auc(y, s, w_cls)
    out("\n== 2. AUC signal vs all background")
    out(f"  unweighted (training metric)      : {res['auc']['unweighted']:.4f}")
    out(f"  weighted   (physical composition) : {res['auc']['weighted']:.4f}")

    out(f"\n== 3. AUC signal vs each background (sorted by yield; n/a below {min_auc_events} events)")
    for b in order:
        if len(scores[b]) < min_auc_events:
            res["auc"][b] = float("nan")
            out(f"  {b:<12} n/a  ({len(scores[b])} events)")
            continue
        yy = np.r_[np.ones(len(scores[signal])), np.zeros(len(scores[b]))]
        ss = np.r_[scores[signal], scores[b]]
        res["auc"][b] = _auc(yy, ss)
        out(f"  {b:<12} {res['auc'][b]:.4f}   ({100 * weights[b].sum() / tot_w:5.2f}% of yield)")

    out(f"\n== 4. cut scan (background above the cut needs N_eff >= {min_neff})")
    s_b = np.concatenate([scores[b] for b in bkgs])
    w_b = np.concatenate([weights[b] for b in bkgs])
    best = None
    z0 = None
    for t in np.arange(0.0, 0.99, 0.05):
        S = weights[signal][scores[signal] > t].sum()
        keep = s_b > t
        wb = w_b[keep]
        B, nraw = wb.sum(), int(keep.sum())
        neff = B ** 2 / (wb ** 2).sum() if nraw else 0.0
        z = S / np.sqrt(B) if B > 0 and neff >= min_neff else np.nan
        if z0 is None:
            z0 = z
        if np.isfinite(z) and (best is None or z > best[1]):
            best = (t, z, S, B, nraw, neff)
        out(f"  score > {t:4.2f}  S={S:10.1f}  B={B:12.1f}  raw={nraw:7d}  N_eff={neff:8.1f}  S/sqrt(B)={z:8.3f}")
    if best is not None:
        t = best[0]
        res["scan"] = dict(cut=t, z=best[1], S=best[2], B=best[3], nraw=best[4], neff=best[5], z_nocut=z0)
        out(f"  best cut {t:.2f}: S/sqrt(B) = {best[1]:.3f}  (no cut: {z0:.3f}, gain x{best[1] / z0:.2f}),"
            f" N_eff = {best[5]:.0f}; background composition there:")
        for b in order:
            part = weights[b][scores[b] > t].sum()
            if part > 0.005 * best[3]:
                out(f"    {b:<12}{100 * part / best[3]:6.1f}% of B   ({int((scores[b] > t).sum())} raw events)")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=[800])
    p.add_argument("--min-neff", type=float, default=11,
                   help="minimum effective background events above a cut (default 11, ~30%% MC stat)")
    a = p.parse_args()

    from diagnostics.common import load_scores, load_weights
    for m in a.masses:
        print(f"\n{'=' * 78}\n{a.cat}  Signal_{m}\n{'=' * 78}")
        _, scores = load_scores(a.cat, m)
        weights = load_weights(a.cat, scores)
        evaluate(scores, weights, f"Signal_{m}", min_neff=a.min_neff)


if __name__ == "__main__":
    main()