"""Depth-matched probe BANK: the missing fair comparison in the multi-concept (scaling) eval.

bench_scaling compared ConceptGate's bank (heads on mid-layer taps, ~91% of the backbone) against a
probe bank whose heads sit on the FINAL layer (100% of the backbone) -- handing the probe a depth
advantage CG did not get. This fits the same logistic heads on CG's own taps, from the activations the
scaling run cached, so cost is identical and only the head differs. Pure CPU; no model is loaded.

Alignment checks: the prompt lists and RNG draws are rebuilt exactly as bench_scaling builds them, and
the script must reproduce the stored probe@final and CG numbers per concept before anything else is
believed.

Run: uv run --with datasets python scripts/eval_probe_tap_bank.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from eval_detection import (_beavertails_concepts, taps_for, BT_SAFE_POOL, BT_POS_PER_CAT, BT_TEST_CAP, _fit_cg)
from conceptgate.concept import Direction

TAP_FRACS = (0.5, 0.7, 0.85)   # what the scaling run used
SEEDS = (0, 1, 2); N_FIT = 32

def _probe(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
    return clf.decision_function((Xte - mu) / sd)

def run(model, stored):
    cats, trc, trsafe, tec, tesafe = _beavertails_concepts()
    _, n_layers = taps_for(model)
    taps = sorted({max(1, min(n_layers - 1, round(n_layers * f))) for f in TAP_FRACS})
    fin = n_layers - 1; layers = sorted(set(taps + [fin]))
    tap_i = [layers.index(t) for t in taps]; fin_i = layers.index(fin)
    trsafe_pool = trsafe[:BT_SAFE_POOL]; trpos = {c: trc[c][:BT_POS_PER_CAT] for c in cats}
    train_list = list(dict.fromkeys(trsafe_pool + [p for c in cats for p in trpos[c]]))
    tesafe_pool = tesafe[:BT_TEST_CAP]; tepos = {c: tec[c][:BT_TEST_CAP] for c in cats}
    test_list = list(dict.fromkeys(tesafe_pool + [p for c in cats for p in tepos[c]]))
    sig = "-".join(map(str, layers)); mk = model.replace("/", "__")
    A_tr = np.load(f"/tmp/cg_fullprobe__{mk}__btsc_train{len(train_list)}__L{sig}.npy")
    A_te = np.load(f"/tmp/cg_fullprobe__{mk}__btsc_test{len(test_list)}__L{sig}.npy")
    assert A_tr.shape[0] == len(train_list) and A_te.shape[0] == len(test_list), "cache/prompt-list size mismatch"
    itr = {p: i for i, p in enumerate(train_list)}; ite = {p: i for i, p in enumerate(test_list)}
    safe_tr = np.array([itr[p] for p in trsafe_pool]); safe_te = np.array([ite[p] for p in tesafe_pool])
    pos_tr = {c: np.array([itr[p] for p in trpos[c]]) for c in cats}
    pos_te = {c: np.array([ite[p] for p in tepos[c]]) for c in cats}
    print(f"\n### DEPTH-MATCHED PROBE BANK {model}  taps {taps} (+final {fin})  "
          f"train {len(train_list)} test {len(test_list)}  cache hit", flush=True)
    print(f"  {'category':48s} {'CG':>6s} {'p@fin':>6s} {'p@taps':>7s} {'p@top':>6s}   (stored CG / p@fin)", flush=True)
    rows = {}
    for c in cats:
        yte = np.r_[np.ones(len(pos_te[c])), np.zeros(len(safe_te))].astype(int)
        Xte_tap = np.concatenate([A_te[pos_te[c]][:, tap_i, :], A_te[safe_te][:, tap_i, :]], 0)
        Xte_fin = np.concatenate([A_te[pos_te[c]][:, fin_i, :], A_te[safe_te][:, fin_i, :]], 0)
        r = {"cg": [], "p_fin": [], "p_taps": [], "p_top": [], "p_each": [[] for _ in taps]}
        for s in SEEDS:
            rng = np.random.default_rng(4000 + s)
            npos = min(N_FIT, len(pos_tr[c]))
            ip = rng.choice(pos_tr[c], npos, replace=False)        # same call order as bench_scaling
            ineg = rng.choice(safe_tr, N_FIT, replace=False)
            gate = _fit_cg(A_tr[ip][:, tap_i, :], A_tr[ineg][:, tap_i, :], Direction.LOGISTIC)
            r["cg"].append(float(roc_auc_score(yte, gate.llr(Xte_tap))))
            ytr = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
            Xp, Xn = A_tr[ip], A_tr[ineg]
            r["p_fin"].append(float(roc_auc_score(yte, _probe(np.concatenate([Xp[:, fin_i], Xn[:, fin_i]]), ytr, Xte_fin))))
            # depth-matched: same logistic head, on CG's taps (concatenated), identical backbone cost
            r["p_taps"].append(float(roc_auc_score(yte, _probe(
                np.concatenate([Xp[:, tap_i].reshape(len(ip), -1), Xn[:, tap_i].reshape(len(ineg), -1)]), ytr,
                Xte_tap.reshape(len(Xte_tap), -1)))))
            for k, ti in enumerate(tap_i):
                r["p_each"][k].append(float(roc_auc_score(yte, _probe(np.concatenate([Xp[:, ti], Xn[:, ti]]), ytr, Xte_tap[:, k]))))
            r["p_top"].append(r["p_each"][-1][-1])
        m = {k: float(np.mean(v)) for k, v in r.items() if k != "p_each"}
        m["p_each"] = [float(np.mean(v)) for v in r["p_each"]]
        rows[c] = m
        st = stored.get(c, {})
        print(f"  {c[:48]:48s} {m['cg']:6.3f} {m['p_fin']:6.3f} {m['p_taps']:7.3f} {m['p_top']:6.3f}   "
              f"({st.get('cg', float('nan')):.3f} / {st.get('probe', float('nan')):.3f})", flush=True)
    mean = {k: float(np.mean([rows[c][k] for c in cats])) for k in ("cg", "p_fin", "p_taps", "p_top")}
    mean["p_each"] = [float(np.mean([rows[c]["p_each"][k] for c in cats])) for k in range(len(taps))]
    # alignment: reproduced numbers vs the stored scaling results
    dcg = max(abs(rows[c]["cg"] - stored[c]["cg"]) for c in cats if c in stored)
    dpf = max(abs(rows[c]["p_fin"] - stored[c]["probe"]) for c in cats if c in stored)
    print(f"\n  MEAN over {len(cats)}: CG {mean['cg']:.3f} | probe@final {mean['p_fin']:.3f} | "
          f"probe@CG-taps {mean['p_taps']:.3f} | probe@top-tap {mean['p_top']:.3f} | per-tap {[round(x,3) for x in mean['p_each']]}", flush=True)
    print(f"  ALIGNMENT vs stored scaling JSON: max |dCG| = {dcg:.4f}, max |dprobe@final| = {dpf:.4f}  "
          f"{'OK' if max(dcg, dpf) < 2e-3 else 'MISMATCH'}", flush=True)
    wins = sum(rows[c]["p_taps"] > rows[c]["cg"] for c in cats)
    print(f"  probe@CG-taps beats CG on {wins}/{len(cats)} categories; mean diff {mean['p_taps']-mean['cg']:+.3f}", flush=True)
    return {"model": model, "taps": taps, "final": fin, "n_fit": N_FIT, "seeds": list(SEEDS),
            "rows": rows, "mean": mean, "alignment": {"max_dcg": dcg, "max_dprobe_final": dpf}}

def main():
    sc = json.load(open("scripts/eval_scaling_results.json"))
    out = []
    for r in sc["results"]:
        a = r["auc"]
        if isinstance(a, dict) and "cg" in a and isinstance(a["cg"], dict):      # {"cg":{cat:auc}, "probe":{cat:auc}}
            stored = {c: {"cg": a["cg"][c], "probe": a["probe"][c]} for c in a["cg"]}
        else:                                                                    # {cat: {"cg":..,"probe":..}}
            stored = {c: {"cg": v.get("cg"), "probe": v.get("probe", v.get("pr"))} for c, v in a.items()}
        out.append(run(r["model"], stored))
    json.dump({"results": out}, open("scripts/eval_probe_tap_bank_results.json", "w"), indent=1)
    print("\ndone -> scripts/eval_probe_tap_bank_results.json")

if __name__ == "__main__":
    main()
