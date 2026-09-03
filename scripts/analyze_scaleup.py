"""Does the steerability-prediction result hold on a second model and across write magnitudes?

Reads `eval_gate.py --scaleup` artifacts and, per model and per alpha, reports:
  - the concept read's correlation with the per-prompt lever, and a ridge fit to the outcome (CV,
    grouped by harmful request, and transferring across prompt families)
  - the cosine between the outcome direction and the concept direction, against the 1/sqrt(d) floor
  - CROSS-MAGNITUDE transfer: fit the direction at one alpha, predict the lever at another. If the
    direction is a property of the prompt rather than of the write size, this should barely degrade.

Run: uv run python scripts/analyze_scaleup.py
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold

RIDGE, N_REQ = 10.0, 12


def _cv(X, y, seed, groups=None, k=5):
    pred = np.zeros_like(y)
    sp = GroupKFold(n_splits=k).split(X, y, groups) if groups is not None else KFold(k, shuffle=True, random_state=seed).split(X)
    for tr, te in sp:
        pred[te] = Ridge(alpha=RIDGE).fit(X[tr], y[tr]).predict(X[te])
    return float(spearmanr(pred, y).correlation)


def run(res):
    model, taps, alphas = res["model"], res["taps"], [f"{a:g}" for a in res["alphas"]]
    kinds = np.array(res["prompt_kinds"])
    A = np.load(res["acts"]).astype(np.float64)
    npr = len(kinds)
    m, dd = A.shape[1], A.shape[2]
    chance = 1 / np.sqrt(dd)
    print(f"\n{'='*92}\n### {model}  taps {taps}  d={dd}  alphas {alphas}  chance |cos| {chance:.4f}\n{'='*92}")
    per_seed = []
    for si, ps in enumerate(res["per_seed"]):
        rows = ps["rows"]
        Wr = np.array(ps["W_raw"])
        X = A[si * npr:(si + 1) * npr]
        atk = kinds != "benign"
        ka = kinds[atk]
        Xa = X[atk].reshape(atk.sum(), -1)
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        Xs = (Xa - mu) / sd
        llr = np.array([r["llr"] for r in rows])[atk]
        g = np.full(atk.sum(), -1)
        ti = np.where(ka == "template")[0]
        g[ti] = np.arange(len(ti)) % N_REQ
        g[ka == "short"] = N_REQ
        g[ka == "request"] = N_REQ + 1
        lev, dirs, rec = {}, {}, {"seed": ps["seed"]}
        for al in alphas:
            R = [r["a"][al] for r in rows]
            odd = np.array([(R[i]["minus"] - R[i]["plus"]) / 2 for i in range(npr)])[atk]
            oddr = np.array([(R[i]["rand_minus"] - R[i]["rand_plus"]) / 2 for i in range(npr)])[atk]
            lev[al] = odd
            fit = Ridge(alpha=RIDGE).fit(Xs, odd)
            dirs[al] = (fit.coef_ / sd).reshape(m, dd)
            unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
            rec[al] = {
                "lever_concept": float(np.abs(odd).mean()), "lever_random": float(np.abs(oddr).mean()),
                "llr_spearman": float(spearmanr(llr, odd).correlation),
                "llr_spearman_template": float(spearmanr(llr[ka == "template"], odd[ka == "template"]).correlation),
                "cv": _cv(Xs, odd, ps["seed"]),
                "cv_grouped": _cv(Xs, odd, ps["seed"], groups=g, k=7),
                "transfer_tmpl_to_other": float(spearmanr(
                    Ridge(alpha=RIDGE).fit(Xs[ka == "template"], odd[ka == "template"]).predict(Xs[ka != "template"]),
                    odd[ka != "template"]).correlation),
                "cos_vs_concept": [float(np.dot(unit(dirs[al][i]), unit(Wr[i]))) for i in range(m)]}
            print(f"  seed {ps['seed']} a={al}: |lever| {rec[al]['lever_concept']:.2f} (rand {rec[al]['lever_random']:.2f})  "
                  f"LLR {rec[al]['llr_spearman']:+.2f} (tmpl {rec[al]['llr_spearman_template']:+.2f})  "
                  f"CV {rec[al]['cv']:+.2f} grouped {rec[al]['cv_grouped']:+.2f} transfer {rec[al]['transfer_tmpl_to_other']:+.2f}  "
                  f"|cos| vs concept {np.mean(np.abs(rec[al]['cos_vs_concept'])):.3f}")
        # cross-magnitude: direction fit at a1 predicting the lever at a2
        cm = {}
        for a1 in alphas:
            for a2 in alphas:
                if a1 == a2:
                    continue
                pred = Ridge(alpha=RIDGE).fit(Xs, lev[a1]).predict(Xs)   # in-sample fit at a1, evaluated on a DIFFERENT target
                cm[f"{a1}->{a2}"] = float(spearmanr(pred, lev[a2]).correlation)
        rec["cross_magnitude"] = cm
        rec["cos_between_alpha_dirs"] = {f"{a1}~{a2}": float(np.mean([
            abs(np.dot(dirs[a1][i] / np.linalg.norm(dirs[a1][i]), dirs[a2][i] / np.linalg.norm(dirs[a2][i]))) for i in range(m)]))
            for a1 in alphas for a2 in alphas if a1 < a2}
        print(f"    cross-magnitude Spearman {({k: round(v,2) for k,v in cm.items()})}")
        print(f"    |cos| between outcome directions at different alpha {({k: round(v,2) for k,v in rec['cos_between_alpha_dirs'].items()})}")
        per_seed.append(rec)
    agg = {}
    for al in alphas:
        agg[al] = {k: (float(np.mean([r[al][k] for r in per_seed])), float(np.std([r[al][k] for r in per_seed])))
                   for k in ("lever_concept", "lever_random", "llr_spearman", "llr_spearman_template", "cv", "cv_grouped", "transfer_tmpl_to_other")}
        agg[al]["cos_vs_concept"] = float(np.mean([np.mean(np.abs(r[al]["cos_vs_concept"])) for r in per_seed]))
    agg["cross_magnitude"] = {k: (float(np.mean([r["cross_magnitude"][k] for r in per_seed])),
                                  float(np.std([r["cross_magnitude"][k] for r in per_seed]))) for k in per_seed[0]["cross_magnitude"]}
    agg["cos_between_alpha_dirs"] = {k: float(np.mean([r["cos_between_alpha_dirs"][k] for r in per_seed])) for k in per_seed[0]["cos_between_alpha_dirs"]}
    agg["chance_cos"] = chance
    print(f"\n  --- MEAN over {len(per_seed)} resamples ---")
    print(f"  {'alpha':>6} {'|lever|':>8} {'random':>7} {'LLR':>7} {'LLR tmpl':>9} {'CV':>7} {'CV grp':>7} {'transfer':>9} {'|cos| concept':>14}")
    for al in alphas:
        a = agg[al]
        print(f"  {al:>6} {a['lever_concept'][0]:8.2f} {a['lever_random'][0]:7.2f} {a['llr_spearman'][0]:+7.2f} "
              f"{a['llr_spearman_template'][0]:+9.2f} {a['cv'][0]:+7.2f} {a['cv_grouped'][0]:+7.2f} {a['transfer_tmpl_to_other'][0]:+9.2f} {a['cos_vs_concept']:14.3f}")
    return {"model": model, "d": dd, "per_seed": per_seed, "summary": agg}


def main():
    d = json.load(open("scripts/eval_gate_results.json"))
    out = [run(r) for r in d["scaleup"]]
    json.dump({"results": out}, open("scripts/scaleup_analysis.json", "w"), indent=1)
    print("\ndone -> scripts/scaleup_analysis.json")


if __name__ == "__main__":
    main()
