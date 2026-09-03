"""Adversarial controls for the steerability-prediction result.

Section 4.11 reports that the per-prompt lever is predictable from the prompt's activations. This
script runs the attacks that decide whether that is a finding or an artifact, and it kills one of the
claims the first version of that section made:

  C1  permutation nulls -- is +0.81 above a null built from the identical pipeline?          (yes, z~9)
  C2  self-consistency ceiling -- how well does the fitted direction agree with ITSELF?      (|cos| 0.18)
      => the reported |cos| 0.09 with the concept direction is ~half that ceiling, NOT orthogonal
  C3  is "project both directions out" a real control?                                       (no: 2
      RANDOM directions change nothing -- the control has no power and is withdrawn)
  C4  converse test -- predict the lever from ONLY the concept direction's own projections    (+0.63,
      i.e. most of the signal lives INSIDE the concept subspace)
  C5  decodability battery -- does this pipeline decode every prompt scalar at ~0.85?         (yes)
  C6  effective replicates -- are the three "seeds" independent?                              (no: the
      activations are byte-identical; only the 8-shot concept fit is resampled)
  C7  learning curve -- how many labelled prompts does the predictor actually need?

Run: uv run python scripts/steerability_controls.py
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

RIDGE, NPERM, NHALF = 10.0, 300, 20


def _cv(X, y, seed=0, k=5):
    p = np.zeros_like(y)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        p[te] = Ridge(alpha=RIDGE).fit(X[tr], y[tr]).predict(X[te])
    return float(spearmanr(p, y).correlation), p


def main():
    d = json.load(open("scripts/eval_gate_results.json"))["steerability"][0]
    rows = d["rows"]
    A = np.load("scripts/steerability_acts.npy").astype(np.float64)
    Wr = np.load("scripts/steerability_wraw.npy")
    nseed = len(set(r["seed"] for r in rows))
    n = len(rows) // nseed
    m, dd = A.shape[1], A.shape[2]
    out = {"n_prompts_per_seed": n, "taps": d["taps"], "d": dd, "n_seeds": nseed}

    # C6: are the seed blocks independent?
    out["max_abs_act_diff_between_seeds"] = float(max(np.abs(A[:n] - A[i * n:(i + 1) * n]).max() for i in range(1, nseed)))
    out["effective_replicates"] = 1 if out["max_abs_act_diff_between_seeds"] == 0.0 else nseed
    print(f"C6 activations identical across seeds: max|diff| = {out['max_abs_act_diff_between_seeds']:.6f} "
          f"-> effective replicates = {out['effective_replicates']}")

    per = []
    for si in range(nseed):
        R = rows[si * n:(si + 1) * n]
        X = A[si * n:(si + 1) * n]
        kind = np.array([r["kind"] for r in R])
        atk = kind != "benign"
        col = lambda k: np.array([r[k] for r in R])
        lever = ((col("minus") - col("plus")) / 2)[atk]
        even = ((col("minus") + col("plus")) / 2)[atk]
        lever_r = ((col("rand_minus") - col("rand_plus")) / 2)[atk]
        even_r = ((col("rand_minus") + col("rand_plus")) / 2)[atk]
        base = col("none")[atk]
        llr = col("llr")[atk]
        ka = kind[atk]
        Xa = X[atk].reshape(atk.sum(), -1)
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        Xs = (Xa - mu) / sd
        r = {"seed": int(R[0]["seed"]), "n_attacks": int(atk.sum())}
        r["frac_lever_negative"] = float((lever < 0).mean())
        cvfull, pred = _cv(Xs, lever, si)
        r["cv_full"] = cvfull
        r["llr_spearman"] = float(spearmanr(llr, lever).correlation)

        # C4: only the concept direction's own per-tap projections (m scalars)
        P = np.stack([X[atk][:, i, :] @ Wr[si][i] for i in range(m)], 1)
        Ps = (P - P.mean(0)) / (P.std(0) + 1e-9)
        r["cv_concept_projection_only"] = _cv(Ps, lever, si)[0]
        # and does the full ridge add anything beyond it?
        _, pw = _cv(Ps, lever, si)
        resid = lever - np.polyval(np.polyfit(pw, lever, 1), pw)
        r["cv_full_on_residual_after_concept_projection"] = _cv(Xs, resid, si)[0]

        # C5: the same pipeline on other prompt scalars -- the generic decodability ceiling
        r["cv_battery"] = {"lever": cvfull, "baseline_logit": _cv(Xs, base, si)[0],
                           "even_component": _cv(Xs, even, si)[0],
                           "random_direction_even": _cv(Xs, even_r, si)[0],
                           "random_direction_lever": _cv(Xs, lever_r, si)[0]}

        # C2: self-consistency ceiling of the fitted direction, and alignment with the concept direction
        rng = np.random.default_rng(si)
        cs = []
        for _ in range(NHALF):
            idx = rng.permutation(len(lever)); h1, h2 = idx[:len(lever) // 2], idx[len(lever) // 2:]
            d1 = (Ridge(alpha=RIDGE).fit(Xs[h1], lever[h1]).coef_ / sd).reshape(m, dd)
            d2 = (Ridge(alpha=RIDGE).fit(Xs[h2], lever[h2]).coef_ / sd).reshape(m, dd)
            u = lambda v: v / (np.linalg.norm(v) + 1e-12)
            cs.append([abs(float(np.dot(u(d1[i]), u(d2[i])))) for i in range(m)])
        r["cos_self_consistency"] = np.array(cs).mean(0).tolist()
        dir_full = (Ridge(alpha=RIDGE).fit(Xs, lever).coef_ / sd).reshape(m, dd)
        u = lambda v: v / (np.linalg.norm(v) + 1e-12)
        r["cos_vs_concept"] = [abs(float(np.dot(u(dir_full[i]), u(Wr[si][i])))) for i in range(m)]
        r["cos_ratio_to_ceiling"] = float(np.mean(r["cos_vs_concept"]) / np.mean(r["cos_self_consistency"]))

        # C3: is projecting directions out a control with any power?
        def proj_out(Z, dirs):
            Y = Z.copy()
            for dv in dirs:
                v = (dv * sd.reshape(m, dd)).reshape(-1); v = v / np.linalg.norm(v)
                Y = Y - np.outer(Y @ v, v)
            return Y
        dir_base = (Ridge(alpha=RIDGE).fit(Xs, base).coef_ / sd).reshape(m, dd)
        rr = np.random.default_rng(100 + si)
        r["projectout"] = {
            "none": cvfull,
            "concept_and_readiness": _cv(proj_out(Xs, [Wr[si], dir_base]), lever, si)[0],
            "two_random_directions": _cv(proj_out(Xs, [rr.normal(size=(m, dd)), rr.normal(size=(m, dd))]), lever, si)[0],
            "the_fitted_direction_itself": _cv(proj_out(Xs, [dir_full]), lever, si)[0]}

        # C1: permutation nulls from the identical pipeline
        rp = np.random.default_rng(1000 + si)
        n1 = [_cv(Xs, lever[rp.permutation(len(lever))], si)[0] for _ in range(NPERM)]
        n2 = []
        for _ in range(NPERM):
            yp = lever.copy()
            for k in set(ka):
                idx = np.where(ka == k)[0]; yp[idx] = lever[rp.permutation(idx)]
            n2.append(_cv(Xs, yp, si)[0])
        r["null_free"] = {"mean": float(np.mean(n1)), "sd": float(np.std(n1)), "p95": float(np.quantile(n1, .95)),
                          "max": float(np.max(n1)), "z": float((cvfull - np.mean(n1)) / np.std(n1))}
        r["null_within_family"] = {"mean": float(np.mean(n2)), "sd": float(np.std(n2)), "p95": float(np.quantile(n2, .95)),
                                   "z": float((cvfull - np.mean(n2)) / np.std(n2))}

        # C7: learning curve
        lc = {}
        for ntr in (4, 8, 16, 32, 64):
            sc = []
            for rep in range(20):
                idx = np.random.default_rng(rep).permutation(len(lever))
                tr, te = idx[:ntr], idx[ntr:]
                sc.append(spearmanr(Ridge(alpha=RIDGE).fit(Xs[tr], lever[tr]).predict(Xs[te]), lever[te]).correlation)
            lc[str(ntr)] = float(np.mean(sc))
        r["learning_curve"] = lc
        if si == 1:
            r["plot"] = {"llr": llr.tolist(), "lever": lever.tolist(), "cv_pred": pred.tolist(),
                         "concept_proj_pred": _cv(Ps, lever, si)[1].tolist(), "kind": ka.tolist()}
        per.append(r)
        print(f"seed {r['seed']}: full {cvfull:+.3f} | LLR {r['llr_spearman']:+.3f} | concept-proj {r['cv_concept_projection_only']:+.3f} "
              f"| ceiling |cos| {np.mean(r['cos_self_consistency']):.3f} vs concept {np.mean(r['cos_vs_concept']):.3f} "
              f"(ratio {r['cos_ratio_to_ceiling']:.2f}) | null z {r['null_free']['z']:.1f}")
    out["per_seed"] = per
    agg = lambda f: (float(np.mean([f(r) for r in per])), float(np.std([f(r) for r in per])))
    out["summary"] = {
        "cv_full": agg(lambda r: r["cv_full"]), "llr": agg(lambda r: r["llr_spearman"]),
        "cv_concept_projection_only": agg(lambda r: r["cv_concept_projection_only"]),
        "cv_full_on_residual": agg(lambda r: r["cv_full_on_residual_after_concept_projection"]),
        "cos_self_consistency": agg(lambda r: float(np.mean(r["cos_self_consistency"]))),
        "cos_vs_concept": agg(lambda r: float(np.mean(r["cos_vs_concept"]))),
        "cos_ratio_to_ceiling": agg(lambda r: r["cos_ratio_to_ceiling"]),
        "frac_lever_negative": agg(lambda r: r["frac_lever_negative"]),
        "battery": {k: agg(lambda r, k=k: r["cv_battery"][k]) for k in per[0]["cv_battery"]},
        "projectout": {k: agg(lambda r, k=k: r["projectout"][k]) for k in per[0]["projectout"]},
        "null_free_mean": agg(lambda r: r["null_free"]["mean"]), "null_free_sd": agg(lambda r: r["null_free"]["sd"]),
        "null_free_z": agg(lambda r: r["null_free"]["z"]),
        "null_family_mean": agg(lambda r: r["null_within_family"]["mean"]),
        "null_family_z": agg(lambda r: r["null_within_family"]["z"]),
        "learning_curve": {k: agg(lambda r, k=k: r["learning_curve"][k]) for k in per[0]["learning_curve"]}}
    json.dump(out, open("scripts/steerability_controls.json", "w"), indent=1)
    S = out["summary"]
    print("\n=== SUMMARY (mean +/- sd over concept resamples; NOTE effective replicates =", out["effective_replicates"], ") ===")
    print(f"  nested ladder:  gate LLR {S['llr'][0]:+.2f}  ->  concept-projection only {S['cv_concept_projection_only'][0]:+.2f}  ->  full ridge {S['cv_full'][0]:+.2f}")
    print(f"  full ridge on the residual after the concept projection: {S['cv_full_on_residual'][0]:+.2f}")
    print(f"  geometry: |cos| vs concept {S['cos_vs_concept'][0]:.3f}, self-consistency ceiling {S['cos_self_consistency'][0]:.3f}, ratio {S['cos_ratio_to_ceiling'][0]:.2f}")
    print(f"  project-out control: none {S['projectout']['none'][0]:+.3f} | concept+readiness {S['projectout']['concept_and_readiness'][0]:+.3f} "
          f"| TWO RANDOM {S['projectout']['two_random_directions'][0]:+.3f} | fitted dir {S['projectout']['the_fitted_direction_itself'][0]:+.3f}")
    print(f"  battery: " + "  ".join(f"{k} {v[0]:+.2f}" for k, v in S["battery"].items()))
    print(f"  null (free) {S['null_free_mean'][0]:+.3f} +/- {S['null_free_sd'][0]:.3f}, z {S['null_free_z'][0]:.1f} | "
          f"null (within family) {S['null_family_mean'][0]:+.3f}, z {S['null_family_z'][0]:.1f}")
    print(f"  learning curve (n labelled prompts -> held-out Spearman): " + "  ".join(f"{k}:{v[0]:+.2f}" for k, v in S["learning_curve"].items()))
    print(f"  levers negative: {S['frac_lever_negative'][0]*100:.0f}% -> predicting dose, not direction")
    print("\ndone -> scripts/steerability_controls.json")


if __name__ == "__main__":
    main()
