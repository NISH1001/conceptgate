"""Analysis for docs/plans/steerability-gate.md. Reads behaviour_dose_results__<tag>.json; never
regenerates.

  --stage 1  reliability of the per-prompt behavioural dose (split-half, inter-instrument, proxy check)
  --stage 2  predict the dose from the unsteered tap activations (ridge, grouped CV, permutation null)
  --stage 3  the outcome-fitted gate as out-of-fold selection over the sampled plus arm
  --stage all

Run: uv run python scripts/analyze_behaviour_dose.py --model Qwen/Qwen2.5-0.5B-Instruct --stage 1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("none", "minus", "plus", "rand_minus", "rand_plus")
INSTRUMENTS = ("lex", "clf")
RIDGE, NPERM, N_REQ = 10.0, 300, 12
GATE1_RELIABILITY, GATE1_AGREEMENT, GATE2_Z = 0.6, 0.6, 4.0
ANALYSIS = os.path.join(HERE, "behaviour_dose_analysis.json")


def tag_of(model: str) -> str:
    return model.replace("/", "__")


def sp(a, b) -> float:
    return float(spearmanr(a, b).correlation)


def load(model, out_dir=HERE):
    tag = tag_of(model)
    meta = json.load(open(os.path.join(out_dir, f"behaviour_dose_results__{tag}.json")))
    acts = np.load(os.path.join(out_dir, f"behaviour_dose_acts__{tag}.npy")).astype(np.float64)
    wraw = np.load(os.path.join(out_dir, f"behaviour_dose_wraw__{tag}.npy")).astype(np.float64)
    return meta, acts, wraw


# ---------------------------------------------------------------- stage 1: the instrument
def rates(rows, instrument, sample_idx):
    """P(refuse | prompt, arm) over the given sample indices. `clf` uses the hard label p >= 0.5."""
    P = {}
    for a in ARMS:
        vals = []
        for r in rows:
            s = r[instrument][a]
            v = [(s[j] >= 0.5) if instrument == "clf" else s[j] for j in sample_idx if j < len(s)]
            vals.append(float(np.mean(v)) if v else np.nan)
        P[a] = np.array(vals, dtype=float)
    return P


def dose(P):
    """Behavioural dose = half the plus-minus refusal difference (plus raises refusal with the chat
    template on), and the same for the random direction of matched norm."""
    return 0.5 * (P["plus"] - P["minus"]), 0.5 * (P["rand_plus"] - P["rand_minus"])


def spearman_brown(rho_half: float) -> float:
    return 2 * rho_half / (1 + rho_half)


def proxy_lever(rows):
    """The report's first-token lever, oriented like the dose: half of (plus - minus) log-odds."""
    return np.array([(r["logit"]["plus"] - r["logit"]["minus"]) / 2 for r in rows])


def stage1(meta):
    rows, k = meta["rows"], int(meta["k"])
    kind = np.array([r["kind"] for r in rows])
    atk = kind != "benign"
    have = [s for s in INSTRUMENTS if all(s in r for r in rows)]
    odd, even, full = list(range(1, k, 2)), list(range(0, k, 2)), list(range(k))
    out = {"n_attacks": int(atk.sum()), "n_benign": int((~atk).sum()), "k": k, "instruments": {}}
    D = {}
    llr = np.array([r["llr"] for r in rows])
    for s in have:
        Pf, Po, Pe = rates(rows, s, full), rates(rows, s, odd), rates(rows, s, even)
        Df, Dr = dose(Pf)
        Do, De = dose(Po)[0], dose(Pe)[0]
        D[s] = Df
        rho = sp(Do[atk], De[atk])
        out["instruments"][s] = {
            "split_half": rho,
            "spearman_brown": spearman_brown(rho),
            "split_half_benign": sp(Do[~atk], De[~atk]) if (~atk).sum() > 2 else float("nan"),
            "baseline_split_half": sp(Po["none"][atk], Pe["none"][atk]),
            "proxy_check": sp(Df[atk], proxy_lever(rows)[atk]),
            "llr_vs_dose": sp(llr[atk], Df[atk]),
            "dose_ratio_attacks": float(np.abs(Df[atk]).mean() / max(np.abs(Dr[atk]).mean(), 1e-9)),
            "mean_abs_dose": {"attacks": float(np.abs(Df[atk]).mean()),
                              "benign": float(np.abs(Df[~atk]).mean()) if (~atk).any() else float("nan"),
                              "random_attacks": float(np.abs(Dr[atk]).mean())},
            "frac_dose_positive_attacks": float((Df[atk] > 0).mean()),
            "refusal_rate": {a: {"attacks": float(Pf[a][atk].mean()),
                                 "benign": float(Pf[a][~atk].mean()) if (~atk).any() else float("nan")}
                             for a in ARMS},
        }
    if len(have) == 2:
        out["inter_instrument"] = sp(D["lex"][atk], D["clf"][atk])
        out["inter_instrument_baseline"] = sp(rates(rows, "lex", full)["none"][atk],
                                              rates(rows, "clf", full)["none"][atk])
    best = max(have, key=lambda s: out["instruments"][s]["split_half"])
    r1 = out["instruments"][best]["split_half"]
    agree = out.get("inter_instrument", float("nan"))
    out["best_instrument"] = best
    out["gate"] = {"reliability": r1, "agreement": agree,
                   "go": bool(r1 >= GATE1_RELIABILITY and (len(have) < 2 or agree >= GATE1_AGREEMENT))}
    return out


def print_stage1(o):
    print(f"\nSTAGE 1  n_attacks={o['n_attacks']} n_benign={o['n_benign']} K={o['k']}")
    print(f"{'instrument':10s} {'split-half':>10s} {'SB(K)':>7s} {'base s-h':>8s} {'proxy':>7s} {'LLR':>6s} "
          f"{'dose/rand':>9s} {'|D|atk':>7s} {'|D|ben':>7s} {'D>0':>5s}")
    for s, v in o["instruments"].items():
        print(f"{s:10s} {v['split_half']:>+10.3f} {v['spearman_brown']:>+7.3f} {v['baseline_split_half']:>+8.3f} "
              f"{v['proxy_check']:>+7.3f} {v['llr_vs_dose']:>+6.2f} {v['dose_ratio_attacks']:>9.2f} "
              f"{v['mean_abs_dose']['attacks']:>7.3f} {v['mean_abs_dose']['benign']:>7.3f} "
              f"{v['frac_dose_positive_attacks']:>5.2f}")
        rr = v["refusal_rate"]
        print("           refusal attacks: " + "  ".join(f"{a} {rr[a]['attacks']:.2f}" for a in ARMS))
        print("           refusal benign : " + "  ".join(f"{a} {rr[a]['benign']:.2f}" for a in ARMS))
    if "inter_instrument" in o:
        print(f"inter-instrument dose agreement {o['inter_instrument']:+.3f} "
              f"(unsteered-rate agreement {o['inter_instrument_baseline']:+.3f})")
    g = o["gate"]
    print(f"GATE 1: {'GO' if g['go'] else 'NO-GO'}  best={o['best_instrument']} reliability {g['reliability']:+.3f} "
          f"(>= {GATE1_RELIABILITY}) agreement {g['agreement']:+.3f} (>= {GATE1_AGREEMENT})")


def _save(model, key, value):
    d = json.load(open(ANALYSIS)) if os.path.exists(ANALYSIS) else {}
    d.setdefault(model, {})[key] = value
    json.dump(d, open(ANALYSIS, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--stage", default="1", help="1 | 2 | 3 | all")
    ap.add_argument("--instrument", default="auto", help="auto | lex | clf")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    meta, acts, wraw = load(a.model, a.out_dir)
    stages = ("1", "2", "3") if a.stage == "all" else (a.stage,)
    if "1" in stages:
        o1 = stage1(meta)
        print_stage1(o1)
        if not a.no_save:
            _save(a.model, "stage1", o1)


if __name__ == "__main__":
    main()
