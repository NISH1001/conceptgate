"""The register x intent grid on real prompts: WildGuardMix test, plus OR-Bench hard.

WildGuardMix (allenai/wildguardmix) labels every prompt on two axes: prompt_harm_label (harmful /
unharmful) and adversarial (jailbreak-style framing / vanilla). Crossed, that is the same 2x2 as
eval_register_grid.py, on 1,725 real prompts that we did not write:

                     adversarial framing            vanilla
  harmful            A  (n=341, median 915 chars)   B  (n=413, 124 chars)
  unharmful          C  (n=455, 711 chars)          D  (n=490,  78 chars)

Two differences from the hand-written grid. "Adversarial" here is long roleplay, story and hypothetical
framing, not a short command to the model; so this tests a different surface axis. And the axis is
confounded with length (adversarial prompts are 6-9x longer), so the script also reports the Spearman
correlation of the gate score with prompt length inside each cell. A>C and B>D are the roughly
length-matched intent comparisons; C>D and A>B mix framing with length.

Cell E is OR-Bench hard (bench-llm/or-bench, or-bench-hard-1k): benign prompts that sound harmful. It
tests a third surface axis, harmful-sounding words with no harmful intent.

Four fits, each eight-shot, logistic, z=2, three resamples:
  0 dataset   jackhhao templates vs real benign (the register section's dataset-fitted gate)
  1 paper     FIT_POS vs FIT_NEG (the paper's gate)
  2 hw-bal    the hand-written grid's register-balanced fit (A+B vs C+D, 4 per cell), transferred
  3 wg-bal    the same balanced fit from WildGuardMix TRAIN (4 per cell): "fit from the traffic the gate
              will see", the fix proposed in future work

Run (needs pandas + pyarrow for the parquet files and datasets for jackhhao):
  uv run --with "datasets<4" --with "huggingface-hub>=1.5,<2" --with pandas --with pyarrow \\
      python scripts/eval_wildguard_grid.py [--model ...] [--dtype bfloat16] [--n-per-cell 128]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import spearmanr

from conceptgate import ConceptGate
from conceptgate.concept import Direction
from eval_gate import CHAR_CAP, FIT_NEG, FIT_POS, N_SHOT, SEEDS, _pools
from eval_register_grid import CELLS as HW_CELLS, auc

HF = os.path.expanduser("~/.cache/huggingface/hub")
LABEL = {"A": "harmful / adversarial", "B": "harmful / vanilla", "C": "unharmful / adversarial",
         "D": "unharmful / vanilla", "E": "OR-Bench hard (benign, sounds harmful)"}
PAIRS = {"A>D": ("A", "D"), "A>C (intent, framing fixed=adversarial)": ("A", "C"),
         "B>D (intent, framing fixed=vanilla)": ("B", "D"), "C>D (framing+length, no intent)": ("C", "D"),
         "A>B (framing+length, intent fixed)": ("A", "B"), "B>C (intent vs framing, the crux)": ("B", "C"),
         "A>E (attack vs harmful-sounding benign)": ("A", "E"), "E>D (harmful-sounding vs plain benign)": ("E", "D")}


def _snap(repo):
    paths = glob.glob(f"{HF}/datasets--{repo.replace('/', '--')}/snapshots/*/")
    if not paths:
        raise FileNotFoundError(f"{repo} not in the local HF cache")
    return paths[0]


def load_wildguard(split):
    import pandas as pd
    d = _snap("allenai/wildguardmix")
    df = pd.read_parquet(f"{d}/{split}/wildguard_{split}.parquet",
                         columns=["prompt", "prompt_harm_label", "adversarial"])
    cells = {}
    for key, harm, adv in (("A", "harmful", True), ("B", "harmful", False),
                           ("C", "unharmful", True), ("D", "unharmful", False)):
        g = df[(df.prompt_harm_label == harm) & (df.adversarial == adv)]
        cells[key] = [p.strip()[:CHAR_CAP] for p in g.prompt.tolist() if isinstance(p, str) and p.strip()]
    return cells


def load_orbench():
    import pandas as pd
    df = pd.read_csv(f"{_snap('bench-llm/or-bench')}/or-bench-hard-1k.csv")
    return [p.strip()[:CHAR_CAP] for p in df.prompt.tolist()]


def _score(cg, prompts):
    v = [cg.check(p) for p in prompts]
    return [x.fired for x in v], [x.score for x in v]


def run_arm(cg, name, fit_fn, test):
    """fit_fn(seed) -> (pos, neg). Scores every test cell per seed."""
    print(f"\n# {name}", flush=True)
    fire = {k: [] for k in test}
    llr = {k: [] for k in test}
    rho = {k: [] for k in test}
    rho_all = []
    for seed in SEEDS:
        pos, neg = fit_fn(seed)
        cg.learn("jailbreak", pos, neg, direction=Direction.LOGISTIC)
        cg.calibrate(z=2.0)
        row, pooled_s, pooled_l = [], [], []
        for k, prompts in test.items():
            f, s = _score(cg, prompts)
            fire[k].append(100 * float(np.mean(f)))
            llr[k].append(s)
            lens = [len(p) for p in prompts]
            rho[k].append(float(spearmanr(s, lens).correlation))
            pooled_s += s
            pooled_l += lens
            row.append(f"{k} {fire[k][-1]:5.1f}%")
        rho_all.append(float(spearmanr(pooled_s, pooled_l).correlation))
        print(f"  seed {seed}: fires  " + "   ".join(row), flush=True)
    out = {"fire": {k: {"mean": round(float(np.mean(v)), 1), "sd": round(float(np.std(v)), 1)}
                    for k, v in fire.items()},
           "auc": {nm: round(float(np.mean([auc(llr[p][i], llr[n][i]) for i in range(len(SEEDS))])), 3)
                   for nm, (p, n) in PAIRS.items() if p in test and n in test},
           "rho_len": {k: round(float(np.mean(v)), 2) for k, v in rho.items()},
           "rho_len_pooled": round(float(np.mean(rho_all)), 2)}
    print("  MEAN fire:  " + "   ".join(f"{k} {out['fire'][k]['mean']}+/-{out['fire'][k]['sd']}%" for k in test))
    for nm, v in out["auc"].items():
        print(f"    AUC {nm}: {v}")
    print("    Spearman(score, chars) per cell: " + "  ".join(f"{k} {v:+.2f}" for k, v in out["rho_len"].items())
          + f"   pooled {out['rho_len_pooled']:+.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--taps", default="")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--n-per-cell", type=int, default=128)
    a = ap.parse_args()
    if a.taps:
        taps = [int(t) for t in a.taps.split(",")]
    else:
        from eval_detection import taps_for
        taps, _ = taps_for(a.model)

    wg_test, wg_train = load_wildguard("test"), load_wildguard("train")
    rng = np.random.default_rng(0)        # one fixed test sample, shared by every arm and seed
    test = {k: [v[i] for i in rng.permutation(len(v))[:a.n_per_cell]] for k, v in wg_test.items()}
    orb = load_orbench()
    test["E"] = [orb[i] for i in rng.permutation(len(orb))[:a.n_per_cell]]
    print(f"### WILDGUARD GRID  {a.model}  taps {taps}  seeds {SEEDS}  n/cell {a.n_per_cell}")
    for k, v in test.items():
        print(f"  {k} {LABEL[k]:40s} n={len(v):3d}  median chars {int(np.median([len(p) for p in v]))}")

    ds_pos, ds_neg, _ = _pools()
    per = N_SHOT // 2

    def fit_pools(pos_pool, neg_pool):
        def f(seed):
            r = np.random.default_rng(seed)
            return ([pos_pool[i] for i in r.permutation(len(pos_pool))[:N_SHOT]],
                    [neg_pool[i] for i in r.permutation(len(neg_pool))[:N_SHOT]])
        return f

    def fit_balanced(cells, seed_offset):
        def f(seed):
            r = np.random.default_rng(seed_offset + seed)
            pick = {k: [cells[k][i] for i in r.permutation(len(cells[k]))[:per]] for k in "ABCD"}
            return pick["A"] + pick["B"], pick["C"] + pick["D"]
        return f

    cg = ConceptGate.from_pretrained(a.model, layers=taps, device=a.device, chat_template=True,
                                     dtype=(a.dtype or None))
    res = {"model": a.model, "taps": list(taps), "n_per_cell": a.n_per_cell,
           "median_chars": {k: int(np.median([len(p) for p in v])) for k, v in test.items()}}
    res["dataset"] = run_arm(cg, "Arm 0: dataset-fitted gate (jackhhao templates vs real benign)",
                             fit_pools(ds_pos, ds_neg), test)
    res["paper"] = run_arm(cg, "Arm 1: paper gate (FIT_POS vs FIT_NEG)", fit_pools(FIT_POS, FIT_NEG), test)
    res["hw_balanced"] = run_arm(cg, "Arm 2: hand-written balanced fit (grid A+B vs C+D), transferred",
                                 fit_balanced(HW_CELLS, 100), test)
    res["wg_balanced"] = run_arm(cg, "Arm 3: WildGuard-train balanced fit (A+B vs C+D, 4 per cell)",
                                 fit_balanced(wg_train, 200), test)
    cg.unload()
    path = os.path.join(os.path.dirname(__file__), "eval_wildguard_grid_results.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres[a.model] = res
    json.dump(allres, open(path, "w"), indent=2)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
