"""Paper figures for the ConceptGate preprint, rebuilt from the stored result files.

Style: vector PDF, LaTeX-rendered text to match the manuscript, Okabe-Ito colours (validated
colour-blind safe), and every series also carries its own marker and line style so each figure
still reads in grayscale. No titles inside the figures; the LaTeX captions carry them.

Run from the repo root:  uv run python docs/preprint/figures/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "scripts"))
sys.path.insert(0, SCRIPTS)
import analyze_behaviour_dose as A  # noqa: E402

# Okabe-Ito
BLUE, VERM, GREEN, PINK, ORANGE = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"
GREY, INK = "#6b6b6b", "#1a1a1a"

TEXTWIDTH = 6.5  # inches: letter paper with 1in margins

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_json(name):
    return json.load(open(os.path.join(SCRIPTS, name)))


def short(model):
    return {"Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "google/gemma-2-2b-it": "gemma-2-2b",
            "gpt2": "GPT-2"}.get(model, model)


def panel_label(ax, s):
    ax.text(-0.14, 1.06, rf"$\mathbf{{({s})}}$", transform=ax.transAxes, ha="left", va="bottom")


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- depth
def fig_depth():
    res = load_json("eval_efficiency_results.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1), sharey=True)
    for ax, r, lab in zip(axes, res, "ab"):
        single = [c for c in r["configs"] if c["config"].startswith("1tap")]
        multi = [c for c in r["configs"] if not c["config"].startswith("1tap")]
        x = [c["depth_frac"] for c in single]
        ax.axhline(r["linear_probe"]["auc"], color=GREY, ls="--", lw=0.9,
                   label="full-model probe (final layer)")
        ax.plot(x, [c["auc_log"] for c in single], color=BLUE, marker="o", label="ConceptGate, logistic")
        ax.plot(x, [c["auc_probe_tap"] for c in single], color=GREEN, ls=":", marker="x", ms=5,
                label="depth-matched probe")
        ax.plot(x, [c["auc_diff"] for c in single], color=VERM, ls="-.", marker="v",
                label="ConceptGate, diff-of-means")
        for c in multi:
            ax.plot(c["depth_frac"], c["auc_log"], color=BLUE, marker="s", ls="none", mfc="white", mew=1.0)
            ax.plot(c["depth_frac"], c["auc_probe_tap"], color=GREEN, marker="+", ls="none", ms=6, mew=1.0)
        ax.plot([], [], color=BLUE, marker="s", ls="none", mfc="white", label="3/5-tap fusion")
        ax.set_xlabel("fraction of network depth run")
        ax.set_xlim(0.2, 1.02)
        ax.set_title(short(r["model"]) + rf" ($N={r['N']}$, 262 held-out prompts)")
        panel_label(ax, lab)
    axes[0].set_ylabel("held-out AUC")
    axes[0].set_ylim(0.9, 0.995)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_depth.pdf")


# --------------------------------------------------------------------------- sample efficiency
def fig_nshot():
    res = load_json("eval_detection_results.json")["results"]
    series = (("conceptgate_log", "ConceptGate, logistic", BLUE, "o", "-"),
              ("logistic", "logistic regression, same taps", GREEN, "x", ":"),
              ("svm_linear", "linear SVM, same taps", GREY, "s", "--"),
              ("conceptgate", "ConceptGate, diff-of-means", VERM, "v", "-."),
              ("best_layer", "best single layer, diff-of-means", ORANGE, "^", (0, (1, 1.5))))
    names = {"HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B"}
    fig, axes = plt.subplots(1, len(res), figsize=(TEXTWIDTH, 2.0), sharey=True)
    for ax, r, lab in zip(axes, res, "abcd"):
        ns = sorted(int(n) for n in r["rows"])
        for key, name, col, mk, ls in series:
            ax.errorbar(ns, [r["rows"][str(n)][key]["auc"] for n in ns],
                        yerr=[r["rows"][str(n)][key]["auc_std"] for n in ns],
                        color=col, marker=mk, ls=ls, ms=3.5, capsize=1.5, elinewidth=0.6, label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("examples per class $N$")
        ax.set_title(names.get(r["model"], short(r["model"])))
        panel_label(ax, lab)
    axes[0].set_ylabel("held-out AUC")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_nshot.pdf")


# --------------------------------------------------------------------------- steering
def fig_steer():
    res = {short(r["model"]): r for r in load_json("eval_steering_results.json")["results"]}
    styles = {"food": (BLUE, "o", "-"), "nature": (GREEN, "s", "--"), "technology": (VERM, "^", ":")}
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 3.2), sharex=True)
    for j, m in enumerate(["GPT-2", "Qwen2.5-0.5B"]):
        r = res[m]
        for c, (col, mk, ls) in styles.items():
            v = r["concepts"][c]
            axes[0, j].plot(v["frac"], v["lex"], color=col, marker=mk, ls=ls, label=c)
            axes[1, j].plot(v["frac"], v["ppl"], color=col, marker=mk, ls=ls, label=c)
        axes[0, j].set_title(m)
        for ax in axes[:, j]:
            ax.axvline(0, color=GREY, lw=0.6, ls="--")
        axes[1, j].set_xlabel(r"steering fraction $\alpha$ (of residual norm)")
    axes[0, 0].set_ylabel("concept keywords (%)")
    axes[1, 0].set_ylabel("perplexity")
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 0].set_ylim(2.5, 10)
    for ax, lab in zip(axes.flat, "abcd"):
        panel_label(ax, lab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_steer.pdf")


# --------------------------------------------------------------------------- held-out category
CAT = {
    "animal_abuse": "animal abuse", "child_abuse": "child abuse",
    "controversial_topics,politics": "controversial/politics",
    "discrimination,stereotype,injustice": "discrimination",
    "drug_abuse,weapons,banned_substance": "drugs/weapons",
    "financial_crime,property_crime,theft": "financial crime",
    "hate_speech,offensive_language": "hate speech",
    "misinformation_regarding_ethics,laws_and_safety": "misinformation",
    "non_violent_unethical_behavior": "non-violent unethical",
    "privacy_violation": "privacy", "self_harm": "self-harm",
    "sexually_explicit,adult_content": "sexual content",
    "terrorism,organized_crime": "terrorism", "violence,aiding_and_abetting,incitement": "violence",
}


def fig_ood():
    res = load_json("eval_ood_results.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.0), sharey=True)
    order = None
    for ax, r, lab in zip(axes, res, "ab"):
        rows = r["rows"]
        if order is None:
            order = sorted(rows, key=lambda c: rows[c]["cg_ood"])
        y = np.arange(len(order))
        off = 0.17
        for k, (key_in, key_ood, col, mk, name, dy) in enumerate((
                ("cg_in", "cg_ood", BLUE, "o", "ConceptGate", off),
                ("pr_in", "pr_ood", VERM, "s", "full-model probe", -off))):
            xin = [rows[c][key_in] for c in order]
            xo = [rows[c][key_ood] for c in order]
            for yi, a, b in zip(y + dy, xin, xo):
                ax.plot([a, b], [yi, yi], color=col, lw=0.6, alpha=0.6)
            ax.plot(xin, y + dy, ls="none", marker=mk, mfc="white", mec=col, mew=0.9,
                    label=f"{name}, in-distribution")
            ax.plot(xo, y + dy, ls="none", marker=mk, color=col, label=f"{name}, held out")
        ax.axvline(0.5, color=GREY, lw=0.6, ls="--")
        s = r["summary"]
        ax.set_title(rf"{short(r['model'])}: held-out mean {s['cg_ood']:.3f} vs {s['pr_ood']:.3f}")
        ax.set_xlabel("AUC")
        ax.set_xlim(0.3, 1.0)
        panel_label(ax, lab)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([CAT.get(c, c) for c in order])
    axes[0].tick_params(axis="y", length=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 0.0))
    save(fig, "fig_ood.pdf")


# --------------------------------------------------------------------------- behavioural dose
def load_dose(model, extra=()):
    tag = A.tag_of(model)
    meta = json.load(open(os.path.join(SCRIPTS, f"behaviour_dose_results__{tag}.json")))
    for p in extra:
        meta = A.merge_runs(meta, json.load(open(os.path.join(SCRIPTS, p))))
    return meta


def scatter_fit(ax, x, y, jx=0.0, jy=0.0, seed=0, col=BLUE):
    """rho and the fit use the raw values; jitter (doses are multiples of 1/K) only moves the dots."""
    rng = np.random.default_rng(seed)
    ax.scatter(x + rng.uniform(-jx, jx, len(x)), y + rng.uniform(-jy, jy, len(y)),
               s=7, color=col, alpha=0.55, lw=0)
    b, a = np.polyfit(x, y, 1)
    xs = np.array([np.min(x), np.max(x)])
    ax.plot(xs, a + b * xs, color=INK, lw=0.9)
    ax.text(0.04, 0.96, rf"$\rho = {A.sp(x, y):+.2f}$", transform=ax.transAxes, va="top")


def fig_instrument():
    meta = load_dose("Qwen/Qwen2.5-0.5B-Instruct")
    rows, k = meta["rows"], int(meta["k"])
    atk = np.array([r["kind"] != "benign" for r in rows])
    Do = A.dose(A.rates(rows, "clf", list(range(1, k, 2))))[0][atk]
    De = A.dose(A.rates(rows, "clf", list(range(0, k, 2))))[0][atk]
    Dl = A.dose(A.rates(rows, "lex", list(range(k))))[0][atk]
    Dc = A.dose(A.rates(rows, "clf", list(range(k))))[0][atk]
    proxy = A.proxy_lever(rows)[atk]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.0))
    j = 0.008
    scatter_fit(axes[0], Do, De, j, j)
    axes[0].set_xlabel("dose, odd samples"); axes[0].set_ylabel("dose, even samples")
    scatter_fit(axes[1], Dl, Dc, j, j)
    axes[1].set_xlabel("dose, refusal lexicon"); axes[1].set_ylabel("dose, rejection classifier")
    scatter_fit(axes[2], proxy, Dc, 0.0, j)
    axes[2].set_xlabel("first-token proxy lever"); axes[2].set_ylabel("sampled behavioural dose")
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, lab)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_instrument.pdf")


def fig_dose():
    an = load_json("behaviour_dose_analysis.json")
    q = load_dose("Qwen/Qwen2.5-0.5B-Instruct")
    rows, k = q["rows"], int(q["k"])
    atk = np.array([r["kind"] != "benign" for r in rows])
    P = A.rates(rows, "clf", list(range(k)))
    D, _ = A.dose(P)
    y = D[atk]
    pred = np.array(an["Qwen/Qwen2.5-0.5B-Instruct"]["stage2"]["pred_oof_grouped"])
    dP = (P["plus"] - P["none"])[atk]

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.0))
    scatter_fit(axes[0], pred, y, 0.0, 0.008)
    axes[0].set_xlabel("predicted dose (out of fold)"); axes[0].set_ylabel("measured dose")

    covs = [0.10, 0.25, 0.50, 0.75, 0.90]
    gate, rmean, rsd = [], [], []
    rng2 = np.random.default_rng(0)
    for cov in covs:
        n = max(1, int(round(cov * len(dP))))
        m = pred >= np.sort(pred)[::-1][n - 1]
        gate.append(float(dP[m].mean()))
        draws = np.array([dP[rng2.permutation(len(dP))[:int(m.sum())]].mean() for _ in range(500)])
        rmean.append(float(draws.mean())); rsd.append(float(draws.std()))
    rmean, rsd = np.array(rmean), np.array(rsd)
    ax = axes[1]
    band = ax.fill_between(covs, rmean - 2 * rsd, rmean + 2 * rsd, color=GREY, alpha=0.18, lw=0)
    (rl,) = ax.plot(covs, rmean, color=GREY, ls="--", marker="x")
    (gl,) = ax.plot(covs, gate, color=BLUE, marker="o")
    ax.set_xlabel(r"share of attacks written"); ax.set_ylabel("refusal gained per write")
    ax.set_xticks(covs); ax.set_xticklabels([f"{int(c*100)}%" for c in covs])
    ax.set_ylim(0, 0.32)
    ax.legend([gl, (band, rl)], ["predicted-dose gate", r"random subset ($\pm2$ s.d.)"],
              loc="upper right")

    g = load_dose("google/gemma-2-2b-it", extra=["behaviour_dose_results__google__gemma-2-2b-it__seed1.json"])
    ax = axes[2]
    for meta, col, mk, ls, name in ((q, BLUE, "o", "-", "Qwen2.5-0.5B ($K=16$)"),
                                     (g, VERM, "s", "--", "gemma-2-2b ($K=32$)")):
        rr, kk = meta["rows"], int(meta["k"])
        a = np.array([r["kind"] != "benign" for r in rr])
        Pm = A.rates(rr, "clf", list(range(kk)))
        vals = [np.nanmean(Pm[arm][a]) for arm in ("minus", "none", "plus")]
        ax.plot([0, 1, 2], vals, color=col, marker=mk, ls=ls, label=name)
        print(f"  {name}: refusal minus/none/plus = {[round(v, 3) for v in vals]}")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"$-\alpha$", "none", r"$+\alpha$"])
    ax.set_xlabel("write"); ax.set_ylabel("refusal rate on attacks")
    ax.set_ylim(0.2, 0.75)
    ax.legend(loc="lower right")
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, lab)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_dose.pdf")
    print(f"  gate {np.round(gate, 3)} random {np.round(rmean, 3)}")


if __name__ == "__main__":
    fig_depth()
    fig_nshot()
    fig_steer()
    fig_ood()
    fig_instrument()
    fig_dose()
