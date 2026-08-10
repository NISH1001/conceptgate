"""Real-data check: MixtureConceptGate on GPT-2 activations (weapons concept).

Two regimes on data/concepts/weapons.json:

  A. strict few-shot (last-token rep, ~12 prompts/class) — expects the BIC few-shot
     collapse to J=1 per class, held-out detection matching the single-Gaussian
     ConceptGate (it IS that gate at J=1), and high LLR rank agreement.
  B. per-token fit (hundreds of samples) — informational: reports whether GPT-2's
     token activations for this concept are multimodal (J>1). NOTE: per-token
     fitting dilutes d' (math.md section 10), so held-out detection in this regime
     is expected to be poor for BOTH gates; it is printed, not gated on.

Run:  uv run python scripts/mixture_gpt2_check.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from conceptgate import data as D  # noqa: E402
from conceptgate.gate import ConceptGate, MixtureConceptGate, recall_fpr  # noqa: E402

MODEL = "gpt2"
CONCEPT_PATH = "data/concepts/weapons.json"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    device = pick_device()
    print(f"device={device} model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    concept = D.load_concept(CONCEPT_PATH)
    layers = concept["layers_gpt2"]

    A_pos, _ = D.extract_token_activations(model, tok, concept["positives"], layers, device, last_only=True)
    A_neg, _ = D.extract_token_activations(model, tok, concept["negatives"], layers, device, last_only=True)
    Tp, _ = D.extract_token_activations(model, tok, concept["test_positives"], layers, device, last_only=True)
    Tn, _ = D.extract_token_activations(model, tok, concept["test_negatives"], layers, device, last_only=True)

    # ---- Regime A: strict few-shot (last-token) ----
    print(f"\n== Regime A: few-shot last-token (pos {A_pos.shape[0]}, neg {A_neg.shape[0]} samples) ==")
    mg = MixtureConceptGate(name=concept["name"]).fit(A_pos, A_neg)
    mg.calibrate_z(3.0)
    J = (mg.gmm_pos.n_components, mg.gmm_neg.n_components)
    print(f"selected J (pos, neg): {J}   (expect (1, 1): BIC few-shot collapse)")
    recall, fpr = recall_fpr(mg.fire(Tp), mg.fire(Tn))
    print(f"mixture gate held-out: recall={recall:.2f} FPR={fpr:.2f}")

    cg = ConceptGate(name=concept["name"], filter_method="fisher").fit(A_pos, A_neg)
    cg.calibrate_z(3.0)
    r0, f0 = recall_fpr(cg.fire(Tp), cg.fire(Tn))
    print(f"single-Gaussian gate  : recall={r0:.2f} FPR={f0:.2f}")

    lm = np.concatenate([mg.llr(Tp), mg.llr(Tn)])
    lc = np.concatenate([cg.llr(Tp), cg.llr(Tn)])
    agree = float(np.corrcoef(lm.argsort().argsort(), lc.argsort().argsort())[0, 1])
    print(f"LLR rank agreement mixture-vs-single: {agree:.3f}")

    # ---- Regime B: per-token (informational) ----
    P_pos, _ = D.extract_token_activations(model, tok, concept["positives"], layers, device, last_only=False)
    P_neg, _ = D.extract_token_activations(model, tok, concept["negatives"], layers, device, last_only=False)
    print(f"\n== Regime B: per-token fit ({P_pos.shape[0]} pos / {P_neg.shape[0]} neg samples) — informational ==")
    mg2 = MixtureConceptGate(name=concept["name"]).fit(P_pos, P_neg)
    J2 = (mg2.gmm_pos.n_components, mg2.gmm_neg.n_components)
    print(f"selected J (pos, neg): {J2}   (J>1 would mean multimodal token activations)")
    mg2.calibrate_z(3.0)
    r2, f2 = recall_fpr(mg2.fire(Tp), mg2.fire(Tn))
    print(f"held-out (last-token) detection from per-token fit: recall={r2:.2f} FPR={f2:.2f} "
          f"(expected poor: per-token dilution, math.md sec. 10)")

    ok = J == (1, 1) and recall >= 0.8 and fpr <= 0.2 and agree >= 0.9
    print("\nMIXTURE GPT-2 CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
