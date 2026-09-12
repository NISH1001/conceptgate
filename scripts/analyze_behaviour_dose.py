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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

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


def merge_runs(meta, extra_meta):
    """Concatenate the per-(prompt, arm) samples of a second run over the SAME prompts (a different
    --seed): K doubles, activations and first-token log-odds stay those of the first run. Returns meta."""
    rows, xrows = meta["rows"], extra_meta["rows"]
    if [r["prompt"] for r in rows] != [r["prompt"] for r in xrows]:
        raise ValueError("cannot merge: the two runs do not cover the same prompts in the same order")
    for key in ("model", "alpha", "temperature", "max_new_tokens", "seed"):   # seed = the concept fit
        if meta.get(key) != extra_meta.get(key):
            raise ValueError(f"cannot merge: {key} differs ({meta.get(key)} vs {extra_meta.get(key)})")
    for r, x in zip(rows, xrows):
        for a in ARMS:
            r["samples"][a] = list(r["samples"][a]) + list(x["samples"][a])
            r["lex"][a] = list(r["lex"][a]) + list(x["lex"][a])
            if "clf" in r and "clf" in x:
                r["clf"][a] = list(r["clf"][a]) + list(x["clf"][a])
            elif "clf" in r:
                del r["clf"]
    meta["k"] = int(meta["k"]) + int(extra_meta["k"])
    meta.setdefault("merged_sample_seeds", [meta.get("sample_seed", meta.get("seed"))]).append(
        extra_meta.get("sample_seed", extra_meta.get("seed")))
    return meta


def load(model, out_dir=HERE, extra=()):
    tag = tag_of(model)
    meta = json.load(open(os.path.join(out_dir, f"behaviour_dose_results__{tag}.json")))
    for path in extra:
        meta = merge_runs(meta, json.load(open(path)))
    acts = np.load(os.path.join(out_dir, f"behaviour_dose_acts__{tag}.npy")).astype(np.float64)
    wraw = np.load(os.path.join(out_dir, f"behaviour_dose_wraw__{tag}.npy")).astype(np.float64)
    return meta, acts, wraw


# ---------------------------------------------------------------- stage 1: the instrument
def rates(rows, instrument, sample_idx):
    """P(refuse | prompt, arm) over the given sample indices. `lex`: lexicon indicator; `clf`: the
    classifier's hard label p >= 0.5 (pre-registered); `clf_soft`: its mean probability (auxiliary)."""
    key = "clf" if instrument.startswith("clf") else instrument
    P = {}
    for a in ARMS:
        vals = []
        for r in rows:
            s = r[key][a]
            if instrument == "clf":
                v = [float(s[j] >= 0.5) for j in sample_idx if j < len(s)]
            else:
                v = [float(s[j]) for j in sample_idx if j < len(s)]
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
    report = have + (["clf_soft"] if "clf" in have else [])     # clf_soft is reported, never gated on
    odd, even, full = list(range(1, k, 2)), list(range(0, k, 2)), list(range(k))
    out = {"n_attacks": int(atk.sum()), "n_benign": int((~atk).sum()), "k": k, "instruments": {}}
    D = {}
    llr = np.array([r["llr"] for r in rows])
    for s in report:
        Pf, Po, Pe = rates(rows, s, full), rates(rows, s, odd), rates(rows, s, even)
        Df, Dr = dose(Pf)
        Do, De = dose(Po)[0], dose(Pe)[0]
        D[s] = Df
        rho = sp(Do[atk], De[atk])
        out["instruments"][s] = {
            "split_half": rho,
            "spearman_brown": spearman_brown(rho),
            "dplus_split_half": sp((Po["plus"] - Po["none"])[atk], (Pe["plus"] - Pe["none"])[atk]),
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
    print(f"{'instrument':10s} {'split-half':>10s} {'SB(K)':>7s} {'dP+ s-h':>8s} {'base s-h':>8s} {'proxy':>7s} {'LLR':>6s} "
          f"{'dose/rand':>9s} {'|D|atk':>7s} {'|D|ben':>7s} {'D>0':>5s}")
    for s, v in o["instruments"].items():
        print(f"{s:10s} {v['split_half']:>+10.3f} {v['spearman_brown']:>+7.3f} {v['dplus_split_half']:>+8.3f} {v['baseline_split_half']:>+8.3f} "
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


# ---------------------------------------------------------------- stage 2: prediction
def groups_for(kind_attacks):
    """Template rows cycle the N_REQ harmful requests; short and bare requests are their own groups."""
    g = np.full(len(kind_attacks), -1)
    ti = np.where(kind_attacks == "template")[0]
    g[ti] = np.arange(len(ti)) % N_REQ
    g[kind_attacks == "short"] = N_REQ
    g[kind_attacks == "request"] = N_REQ + 1
    return g


def cv(X, y, seed, groups=None, n_splits=5):
    pred = np.zeros(len(y), dtype=float)
    split = (GroupKFold(n_splits=n_splits).split(X, y, groups) if groups is not None
             else KFold(n_splits, shuffle=True, random_state=seed).split(X))
    for tr, te in split:
        pred[te] = Ridge(alpha=RIDGE).fit(X[tr], y[tr]).predict(X[te])
    return sp(pred, y), pred


def _zscore(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def stage2(meta, acts, wraw, instrument, seed=0, nperm=NPERM):
    rows, k = meta["rows"], int(meta["k"])
    kind = np.array([r["kind"] for r in rows])
    atk = kind != "benign"
    P = rates(rows, instrument, list(range(k)))
    D, Dr = dose(P)
    y, yr, p0 = D[atk], Dr[atk], P["none"][atk]
    llr = np.array([r["llr"] for r in rows])[atk]
    n, m, d = acts.shape
    Xa = acts[atk].reshape(int(atk.sum()), -1)
    Xs, mu, sd = _zscore(Xa)
    Xb = (acts[~atk].reshape(int((~atk).sum()), -1) - mu) / sd
    ka = kind[atk]
    g = groups_for(ka)
    ng = min(7, len(set(g)))
    out = {"instrument": instrument, "n_attacks": int(atk.sum()), "d": int(d), "m": int(m)}
    out["cv"], _ = cv(Xs, y, seed)
    out["cv_grouped"], pred_g = cv(Xs, y, seed, groups=g, n_splits=ng)
    out["llr_spearman"] = sp(llr, y)
    proj = np.stack([acts[atk][:, i, :] @ wraw[i] for i in range(m)], 1)
    out["cv_concept_projection_only"] = cv(_zscore(proj)[0], y, seed, groups=g, n_splits=ng)[0]
    out["cv_random_dose"] = cv(Xs, yr, seed, groups=g, n_splits=ng)[0]
    out["cv_baseline_rate"] = cv(Xs, p0, seed, groups=g, n_splits=ng)[0]
    out["baseline_vs_dose"] = sp(p0, y)
    rp = np.random.default_rng(1000 + seed)
    null = np.array([cv(Xs, y[rp.permutation(len(y))], seed, groups=g, n_splits=ng)[0] for _ in range(nperm)])
    out["null"] = {"n": int(nperm), "mean": float(null.mean()), "sd": float(null.std()),
                   "p95": float(np.quantile(null, .95)), "z": float((out["cv_grouped"] - null.mean()) / null.std())}
    lc = {}
    for ntr in (8, 16, 32, 64):
        if ntr >= len(y) - 8:
            continue
        sc = []
        for rep in range(20):
            idx = np.random.default_rng(rep).permutation(len(y))
            tr, te = idx[:ntr], idx[ntr:]
            sc.append(sp(Ridge(alpha=RIDGE).fit(Xs[tr], y[tr]).predict(Xs[te]), y[te]))
        lc[str(ntr)] = float(np.mean(sc))
    out["learning_curve"] = lc
    tm, oth = ka == "template", ka != "template"
    if tm.sum() > 10 and oth.sum() > 10:
        out["transfer_template_to_other"] = sp(Ridge(alpha=RIDGE).fit(Xs[tm], y[tm]).predict(Xs[oth]), y[oth])
        out["transfer_other_to_template"] = sp(Ridge(alpha=RIDGE).fit(Xs[oth], y[oth]).predict(Xs[tm]), y[tm])
    full = Ridge(alpha=RIDGE).fit(Xs, y)
    direction = (full.coef_ / sd).reshape(m, d)
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)  # noqa: E731
    out["cos_vs_concept"] = [float(abs(unit(direction[i]) @ unit(wraw[i]))) for i in range(m)]
    out["chance_cos"] = float(1 / np.sqrt(d))
    halves = []
    for rep in range(20):
        idx = np.random.default_rng(500 + rep).permutation(len(y))
        h1, h2 = idx[: len(y) // 2], idx[len(y) // 2:]
        d1 = (Ridge(alpha=RIDGE).fit(Xs[h1], y[h1]).coef_ / sd).reshape(m, d)
        d2 = (Ridge(alpha=RIDGE).fit(Xs[h2], y[h2]).coef_ / sd).reshape(m, d)
        halves.append([float(abs(unit(d1[i]) @ unit(d2[i]))) for i in range(m)])
    out["self_consistency"] = np.array(halves).mean(0).tolist()
    out["pred_oof_grouped"] = pred_g.tolist()
    out["pred_benign"] = full.predict(Xb).tolist() if (~atk).any() else []
    out["threshold_median"] = float(np.median(pred_g))
    out["gate"] = {"z": out["null"]["z"], "cv_grouped": out["cv_grouped"], "llr": out["llr_spearman"],
                   "go": bool(out["null"]["z"] >= GATE2_Z and out["cv_grouped"] > abs(out["llr_spearman"]))}
    return out


def print_stage2(o):
    print(f"\nSTAGE 2  instrument={o['instrument']} n_attacks={o['n_attacks']} features={o['m']}x{o['d']}")
    print(f"  ridge -> dose        CV {o['cv']:+.3f}   grouped {o['cv_grouped']:+.3f}")
    print(f"  permutation null     {o['null']['mean']:+.3f} +/- {o['null']['sd']:.3f}  (p95 {o['null']['p95']:+.3f})  z = {o['null']['z']:.1f}")
    print(f"  baselines            gate LLR {o['llr_spearman']:+.3f}   3 concept projections {o['cv_concept_projection_only']:+.3f}")
    print(f"  controls             random-direction dose {o['cv_random_dose']:+.3f}   unsteered rate {o['cv_baseline_rate']:+.3f}   "
          f"Spearman(rate, dose) {o['baseline_vs_dose']:+.3f}")
    print("  learning curve       " + "  ".join(f"n={k} {v:+.2f}" for k, v in o["learning_curve"].items()))
    if "transfer_template_to_other" in o:
        print(f"  transfer             templates->other {o['transfer_template_to_other']:+.3f}   other->templates {o['transfer_other_to_template']:+.3f}")
    print(f"  |cos| vs W_raw       {[round(c, 3) for c in o['cos_vs_concept']]}  (chance {o['chance_cos']:.3f});  "
          f"split-half self-consistency {[round(c, 2) for c in o['self_consistency']]}")
    g = o["gate"]
    print(f"GATE 2: {'GO' if g['go'] else 'NO-GO'}  z {g['z']:.1f} (>= {GATE2_Z}) and grouped CV {g['cv_grouped']:+.3f} > |LLR| {abs(g['llr']):.3f}")


# ---------------------------------------------------------------- stage 3: the gate as selection
def stage3(meta, o2, instrument, n_random=500, seed=0):
    """Every gating arm is a SELECTION over the same per-prompt refusal gain of the +alpha arm,
    dP = P(plus) - P(none), with predictions OUT OF FOLD. Adds no evidence beyond stage 2; the
    random-halves null is the only genuine test (blanket = gate + anti-gate is an identity)."""
    rows, k = meta["rows"], int(meta["k"])
    kind = np.array([r["kind"] for r in rows])
    atk = kind != "benign"
    P = rates(rows, instrument, list(range(k)))
    dP = P["plus"] - P["none"]
    dPa, dPb = dP[atk], dP[~atk]
    fired = np.array([r["fired"] for r in rows])
    fired_a, fired_b = fired[atk], fired[~atk]
    pred_a = np.array(o2["pred_oof_grouped"])
    pred_b = np.array(o2["pred_benign"]) if len(o2["pred_benign"]) else np.zeros(0)
    tau = o2["threshold_median"]
    sel = {"blanket": np.ones(len(dPa), bool), "concept_gate": fired_a,
           "outcome_gate": pred_a >= tau, "anti_outcome": pred_a < tau, "both": fired_a & (pred_a >= tau)}
    selb = {"blanket": np.ones(len(dPb), bool), "concept_gate": fired_b,
            "outcome_gate": pred_b >= tau, "anti_outcome": pred_b < tau, "both": fired_b & (pred_b >= tau)}

    def arm(m, mb):
        return {"n_written": int(m.sum()),
                "mean_dP_written": float(dPa[m].mean()) if m.any() else float("nan"),
                "total_dP": float(dPa[m].sum()),
                "benign_written_frac": float(mb.mean()) if len(mb) else float("nan"),
                "benign_mean_abs_dP": float(np.abs(dPb[mb]).mean()) if mb.any() else 0.0}

    out = {"instrument": instrument, "threshold": float(tau), "arms": {a: arm(sel[a], selb[a]) for a in sel}}
    n_sel = int(sel["outcome_gate"].sum())
    rng = np.random.default_rng(seed)
    rand = np.array([dPa[rng.permutation(len(dPa))[:n_sel]].mean() for _ in range(n_random)])
    og = out["arms"]["outcome_gate"]["mean_dP_written"]
    out["random"] = {"n": int(n_random), "size": n_sel, "mean": float(rand.mean()), "sd": float(rand.std()),
                     "p_ge_outcome": float((rand >= og).mean())}
    return out


def print_stage3(o):
    print(f"\nSTAGE 3  instrument={o['instrument']}  threshold (median OOF prediction) {o['threshold']:+.3f}")
    print(f"{'arm':14s} {'written':>7s} {'dP/write':>9s} {'total dP':>9s} {'benign written':>14s} {'benign |dP|':>11s}")
    for a, v in o["arms"].items():
        print(f"{a:14s} {v['n_written']:>7d} {v['mean_dP_written']:>+9.3f} {v['total_dP']:>+9.2f} "
              f"{v['benign_written_frac']:>14.2f} {v['benign_mean_abs_dP']:>11.3f}")
    r = o["random"]
    print(f"{'random(size)':14s} {r['size']:>7d} {r['mean']:>+9.3f} +/- {r['sd']:.3f}   P(random >= outcome_gate) = {r['p_ge_outcome']:.3f}")


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
    ap.add_argument("--extra", default="", help="comma-separated extra results files (other seeds) to merge: K adds up")
    a = ap.parse_args()
    meta, acts, wraw = load(a.model, a.out_dir, extra=[e for e in a.extra.split(",") if e])
    stages = ("1", "2", "3") if a.stage == "all" else (a.stage,)
    if "1" in stages:
        o1 = stage1(meta)
        print_stage1(o1)
        if not a.no_save:
            _save(a.model, "stage1", o1)
    if "2" in stages:
        inst = a.instrument
        if inst == "auto":
            d = json.load(open(ANALYSIS)) if os.path.exists(ANALYSIS) else {}
            inst = d.get(a.model, {}).get("stage1", {}).get("best_instrument") or stage1(meta)["best_instrument"]
        o2 = stage2(meta, acts, wraw, inst)
        print_stage2(o2)
        if not a.no_save:
            _save(a.model, "stage2", o2)
    if "3" in stages:
        d = json.load(open(ANALYSIS)) if os.path.exists(ANALYSIS) else {}
        o2 = d.get(a.model, {}).get("stage2") or stage2(meta, acts, wraw, stage1(meta)["best_instrument"])
        if not o2["gate"]["go"]:
            print("\nSTAGE 3 skipped: stage 2 is NO-GO for this model")
        else:
            o3 = stage3(meta, o2, o2["instrument"])
            print_stage3(o3)
            if not a.no_save:
                _save(a.model, "stage3", o3)


if __name__ == "__main__":
    main()
