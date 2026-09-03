"""Is per-prompt steerability predictable from the prompt's activations, and does it live in the
concept's own direction?

Reads the artifacts of `eval_gate.py --steerability` (per-prompt refusal-logit responses to +/-alpha
along the concept and along a random direction, plus the tapped activations) and answers:

  1. Does the few-shot concept read (gate LLR) predict the per-prompt lever?              (yes, moderate)
  2. Does a direction fit to the OUTCOME predict it better, and generalize across prompt kinds? (yes)
  3. Is that outcome direction the concept direction?                                     (no)
  4. Is it just residual norm / baseline refusal / distance-to-boundary / a known direction? (no)

Every reported number is produced here. Run after --steerability:
    uv run python scripts/analyze_steerability.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold

ACTS = "scripts/steerability_acts.npy"
WRAW = "scripts/steerability_wraw.npy"
ALPHA_RIDGE, N_REQ = 10.0, 12


def dump_wraw(model="Qwen/Qwen2.5-0.5B-Instruct", taps=(8, 12, 16), device="mps"):
    """Re-fit the per-seed concept exactly as the run did and save its raw steering directions."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_gate as E
    from conceptgate import ConceptGate
    from conceptgate.concept import Direction
    cg = ConceptGate.from_pretrained(model, layers=list(taps), device=device, chat_template=True)
    W = []
    for seed in E.SEEDS:
        rng = np.random.default_rng(seed)
        cg.learn("jailbreak", [E.FIT_POS[i] for i in rng.permutation(len(E.FIT_POS))[:E.N_SHOT]],
                 [E.FIT_NEG[i] for i in rng.permutation(len(E.FIT_NEG))[:E.N_SHOT]],
                 direction=Direction.LOGISTIC)
        W.append(np.asarray(cg.concepts["jailbreak"].W_raw))
    cg.unload()
    np.save(WRAW, np.array(W))
    print(f"saved {WRAW} {np.array(W).shape}")


def _cv(X, y, seed, groups=None, n_splits=5):
    pred = np.zeros_like(y)
    sp = (GroupKFold(n_splits=n_splits).split(X, y, groups) if groups is not None
          else KFold(n_splits, shuffle=True, random_state=seed).split(X))
    for tr, te in sp:
        pred[te] = Ridge(alpha=ALPHA_RIDGE).fit(X[tr], y[tr]).predict(X[te])
    return float(spearmanr(pred, y).correlation)


def main():
    d = json.load(open("scripts/eval_gate_results.json"))["steerability"][0]
    rows = d["rows"]
    A = np.load(ACTS).astype(np.float64)
    if not os.path.exists(WRAW):
        dump_wraw(d["model"], tuple(d["taps"]))
    Wr = np.load(WRAW)
    n = len(rows) // len(set(r["seed"] for r in rows))
    m, dd = A.shape[1], A.shape[2]
    col = lambda R, k: np.array([r[k] for r in R])
    print(f"{len(rows)} rows, {n}/seed; taps {d['taps']}, d={dd}; |alpha| {abs(d['frac'])}; chance |cos| {1/np.sqrt(dd):.4f}\n")
    out = []
    for si, seed in enumerate(sorted(set(r["seed"] for r in rows))):
        R = rows[si * n:(si + 1) * n]
        X = A[si * n:(si + 1) * n]
        kind = np.array([r["kind"] for r in R])
        atk = kind != "benign"
        # lever = odd (sign-reversible) component; even = sign-independent perturbation
        odd = (col(R, "minus") - col(R, "plus")) / 2
        oddr = (col(R, "rand_minus") - col(R, "rand_plus")) / 2
        base, llr = col(R, "none"), col(R, "llr")
        norm = np.linalg.norm(X, axis=2).mean(1)
        ya, ba, na = odd[atk], base[atk], norm[atk]
        Xa = X[atk].reshape(atk.sum(), -1)
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        Xs = (Xa - mu) / sd
        ka = kind[atk]
        # groups: template rows cycle N_REQ harmful requests; other kinds are their own groups
        g = np.full(len(ya), -1)
        ti = np.where(ka == "template")[0]
        g[ti] = np.arange(len(ti)) % N_REQ
        g[ka == "short"] = N_REQ
        g[ka == "request"] = N_REQ + 1
        r = {"seed": int(seed), "n_attacks": int(atk.sum()),
             "lever_concept": float(np.abs(ya).mean()), "lever_random": float(np.abs(oddr[atk]).mean()),
             "llr_vs_lever": {k: float(spearmanr(llr[kind == k], odd[kind == k]).correlation)
                              for k in ("template", "short", "benign")},
             "llr_vs_lever_all_attacks": float(spearmanr(llr[atk], ya).correlation),
             "outcome_cv": _cv(Xs, ya, seed),
             "outcome_cv_grouped": _cv(Xs, ya, seed, groups=g, n_splits=7),
             "outcome_cv_template_only": _cv(Xs[ka == "template"], ya[ka == "template"], seed,
                                             groups=g[ka == "template"], n_splits=6),
             "llr_template_only": float(spearmanr(llr[atk][ka == "template"], ya[ka == "template"]).correlation)}
        # cross-kind transfer
        Xt, yt = Xs[ka == "template"], ya[ka == "template"]
        oth = ka != "template"
        r["transfer_template_to_other"] = float(spearmanr(Ridge(alpha=ALPHA_RIDGE).fit(Xt, yt).predict(Xs[oth]), ya[oth]).correlation)
        r["transfer_other_to_template"] = float(spearmanr(Ridge(alpha=ALPHA_RIDGE).fit(Xs[oth], ya[oth]).predict(Xt), yt).correlation)
        # confound controls
        Z = np.c_[np.ones_like(na), na, ba]
        res = ya - Z @ np.linalg.lstsq(Z, ya, rcond=None)[0]
        r["outcome_cv_residualized_norm_baseline"] = _cv(Xs, res, seed)
        r["outcome_cv_per_unit_write"] = _cv(Xs, ya / na, seed)
        r["norm_vs_lever"] = float(spearmanr(na, ya).correlation)
        r["baseline_vs_lever"] = float(spearmanr(ba, ya).correlation)
        r["absbaseline_vs_abslever"] = float(spearmanr(np.abs(ba), np.abs(ya)).correlation)
        # geometry: outcome direction vs the concept direction and vs a refusal-readiness direction
        dir_out = (Ridge(alpha=ALPHA_RIDGE).fit(Xs, ya).coef_ / sd).reshape(m, dd)
        dir_base = (Ridge(alpha=ALPHA_RIDGE).fit(Xs, ba).coef_ / sd).reshape(m, dd)
        unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
        r["cos_outcome_vs_concept"] = [float(np.dot(unit(dir_out[i]), unit(Wr[si][i]))) for i in range(m)]
        r["cos_outcome_vs_readiness"] = [float(np.dot(unit(dir_out[i]), unit(dir_base[i]))) for i in range(m)]
        r["cos_readiness_vs_concept"] = [float(np.dot(unit(dir_base[i]), unit(Wr[si][i]))) for i in range(m)]
        r["readiness_cv"] = _cv(Xs, ba, seed)
        # does the signal survive removing BOTH known directions?
        Y = Xs.copy()
        for dv in (dir_base, Wr[si]):
            v = unit((dv * sd.reshape(m, dd)).reshape(-1))
            Y = Y - np.outer(Y @ v, v)
        r["outcome_cv_after_projecting_out_both"] = _cv(Y, ya, seed)
        if si == 1:   # keep one seed's per-prompt values so the report figure is reproducible
            pcv = np.zeros_like(ya)
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
                pcv[te] = Ridge(alpha=ALPHA_RIDGE).fit(Xs[tr], ya[tr]).predict(Xs[te])
            r["plot"] = {"llr": llr[atk].tolist(), "lever": ya.tolist(), "cv_pred": pcv.tolist(),
                         "kind": ka.tolist()}
        out.append(r)
        print(f"seed {seed}: |lever| concept {r['lever_concept']:.2f} vs random {r['lever_random']:.2f}")
        print(f"  LLR->lever: all attacks {r['llr_vs_lever_all_attacks']:+.2f}, template-only {r['llr_template_only']:+.2f}")
        print(f"  outcome gate CV {r['outcome_cv']:+.2f} | grouped {r['outcome_cv_grouped']:+.2f} | template-only grouped {r['outcome_cv_template_only']:+.2f}")
        print(f"  transfer template->other {r['transfer_template_to_other']:+.2f} | other->template {r['transfer_other_to_template']:+.2f}")
        print(f"  controls: norm->lever {r['norm_vs_lever']:+.2f} baseline->lever {r['baseline_vs_lever']:+.2f} |base|->|lever| {r['absbaseline_vs_abslever']:+.2f}"
              f" | CV residualized {r['outcome_cv_residualized_norm_baseline']:+.2f} per-unit {r['outcome_cv_per_unit_write']:+.2f}")
        print(f"  cos(outcome, concept) {np.round(r['cos_outcome_vs_concept'],3)} | cos(outcome, readiness) {np.round(r['cos_outcome_vs_readiness'],3)}"
              f" | CV after projecting out both {r['outcome_cv_after_projecting_out_both']:+.2f}\n")
    agg = lambda k: (float(np.mean([r[k] for r in out])), float(np.std([r[k] for r in out])))
    summary = {k: agg(k) for k in ("llr_vs_lever_all_attacks", "llr_template_only", "outcome_cv", "outcome_cv_grouped",
                                   "outcome_cv_template_only", "transfer_template_to_other", "transfer_other_to_template",
                                   "outcome_cv_residualized_norm_baseline", "outcome_cv_per_unit_write",
                                   "outcome_cv_after_projecting_out_both", "readiness_cv", "norm_vs_lever",
                                   "baseline_vs_lever", "absbaseline_vs_abslever", "lever_concept", "lever_random")}
    summary["cos_outcome_vs_concept_mean"] = float(np.mean([np.mean(np.abs(r["cos_outcome_vs_concept"])) for r in out]))
    summary["cos_outcome_vs_readiness_mean"] = float(np.mean([np.mean(np.abs(r["cos_outcome_vs_readiness"])) for r in out]))
    summary["chance_cos"] = float(1 / np.sqrt(dd))
    print("=== MEAN +/- SD over seeds ===")
    for k, v in summary.items():
        if isinstance(v, tuple):
            print(f"  {k:42s} {v[0]:+.2f} +/- {v[1]:.2f}")
    print(f"  {'mean |cos| outcome vs concept':42s} {summary['cos_outcome_vs_concept_mean']:.3f}  (chance {summary['chance_cos']:.3f})")
    print(f"  {'mean |cos| outcome vs readiness':42s} {summary['cos_outcome_vs_readiness_mean']:.3f}")
    json.dump({"per_seed": out, "summary": summary}, open("scripts/steerability_analysis.json", "w"), indent=1)
    print("\ndone -> scripts/steerability_analysis.json")


if __name__ == "__main__":
    main()
