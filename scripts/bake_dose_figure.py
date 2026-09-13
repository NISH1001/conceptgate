"""Bake the §4.11 behavioural-dose figures (static inline SVG, report style) from the measured runs.

Three panels, left to right:
  A  out-of-fold predicted dose vs measured dose on Qwen2.5-0.5B -- the finding
  B  gain per write against coverage, gate vs a size-matched random null -- the payoff
  C  refusal rate per arm, Qwen vs gemma -- why the second model has nothing to predict

And a second figure, the instrument itself:
  A  split-half reliability of the dose -- it is measurable at all
  B  the two independent scorers against each other -- they agree
  C  the first-token proxy against the sampled dose -- the withdrawn claim, rehabilitated

Writes scripts/fig_dose.svg and scripts/fig_instrument.svg. Render both and LOOK at them before
they go anywhere (numeric checks have missed label collisions in this project before).

Run: uv run python scripts/bake_dose_figure.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_behaviour_dose as A  # noqa: E402

INK = "currentColor"          # theme-adaptive, as the report's other figures use
AXIS = "#d8d5c8"
MUTED = "#889"
TEAL = "#26A99D"
DARK = "#1c7d74"
WARM = "#c2703d"
W, H = 760, 286


def panel_geom(x0):
    """plot box for a panel whose left edge is x0"""
    return dict(x0=x0, x1=x0 + 176, y0=196, y1=74)   # y0 = bottom (SVG y grows down)


def sx(g, t):
    return g["x0"] + t * (g["x1"] - g["x0"])


def sy(g, t):
    return g["y0"] + t * (g["y1"] - g["y0"])


def frame(g, xlab, ylab):
    out = [f'<line x1="{g["x0"]}" y1="{g["y0"]}" x2="{g["x1"]}" y2="{g["y0"]}" stroke="{AXIS}"/>',
           f'<line x1="{g["x0"]}" y1="{g["y0"]}" x2="{g["x0"]}" y2="{g["y1"]}" stroke="{AXIS}"/>',
           f'<text x="{(g["x0"]+g["x1"])/2:.0f}" y="{g["y0"]+17}" text-anchor="middle" font-size="9" fill="{MUTED}">{xlab}</text>']
    cy = (g["y0"] + g["y1"]) / 2
    out.append(f'<text x="{g["x0"]-30}" y="{cy:.0f}" text-anchor="middle" font-size="9" fill="{MUTED}" '
               f'transform="rotate(-90 {g["x0"]-30} {cy:.0f})">{ylab}</text>')
    return out


def scatter(g, xv, yv, note=None, fit=True, color=TEAL, shared=True):
    """Points with a least-squares guide. `shared=True` puts both axes on one scale, which is right only
    when they carry the SAME unit (dose vs dose); with different units (a log-odds lever against a
    probability) a shared scale squashes one axis into a band, so scale them independently."""
    out = []
    if shared:
        lo = hi = None
        lo = float(min(np.min(xv), np.min(yv))); hi = float(max(np.max(xv), np.max(yv)))
        rx = ry = (hi - lo) or 1.0
        lox = loy = lo
    else:
        lox, hix = float(np.min(xv)), float(np.max(xv))
        loy, hiy = float(np.min(yv)), float(np.max(yv))
        rx, ry = (hix - lox) or 1.0, (hiy - loy) or 1.0
    nx_ = lambda v: (v - lox) / rx  # noqa: E731
    ny_ = lambda v: (v - loy) / ry  # noqa: E731
    for a_, b_ in zip(xv, yv):
        out.append(f'<circle cx="{sx(g, nx_(a_)):.1f}" cy="{sy(g, ny_(b_)):.1f}" r="2" fill="{color}" opacity="0.5"/>')
    if fit:
        m, c = np.polyfit(xv, yv, 1)
        x1v, x2v = float(np.min(xv)), float(np.max(xv))
        out.append(f'<line x1="{sx(g, nx_(x1v)):.1f}" y1="{sy(g, ny_(m*x1v+c)):.1f}" '
                   f'x2="{sx(g, nx_(x2v)):.1f}" y2="{sy(g, ny_(m*x2v+c)):.1f}" stroke="{DARK}" stroke-width="1.2"/>')
    if note:
        out.append(f'<text x="{g["x0"]+4}" y="{g["y1"]-6}" font-size="8" fill="{MUTED}">{note}</text>')
    return out


def head(g, title, stat, color=DARK):
    cx = (g["x0"] + g["x1"]) / 2
    return [f'<text x="{cx:.0f}" y="46" text-anchor="middle" font-size="10.5" fill="{INK}">{title}</text>',
            f'<text x="{cx:.0f}" y="61" text-anchor="middle" font-size="11" font-weight="600" fill="{color}">{stat}</text>']


def main():
    meta, acts, wraw = A.load("Qwen/Qwen2.5-0.5B-Instruct", out_dir=HERE)
    rows, k = meta["rows"], int(meta["k"])
    kind = np.array([r["kind"] for r in rows]); atk = kind != "benign"
    P = A.rates(rows, "clf", list(range(k))); D, _ = A.dose(P)
    o2 = A.stage2(meta, acts, wraw, "clf", seed=0, nperm=50)
    pred, y = np.array(o2["pred_oof_grouped"]), D[atk]
    dP = (P["plus"] - P["none"])[atk]

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
         f'aria-label="Per-prompt behavioural steering dose: predicted vs measured, gate selection, and the second model" '
         f'font-family="ui-sans-serif,system-ui,sans-serif">',
         f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="{INK}">'
         f'how far will a write move THIS prompt? measured on behaviour, predicted before generation</text>',
         f'<text x="{W/2:.0f}" y="32" text-anchor="middle" font-size="9" fill="{MUTED}">'
         f'Qwen2.5-0.5B, 164 held-out attacks, K=16 sampled continuations per arm, dose = ½(P₊ − P₋)</text>']

    # ---------------- A: predicted vs measured
    gA = panel_geom(60)
    s += head(gA, "ridge on 3 taps, out of fold", "ρ = +0.58")
    s += frame(gA, "predicted dose", "measured dose")
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    rng = hi - lo or 1.0
    nx = lambda v: (v - lo) / rng  # noqa: E731
    for px, py in zip(pred, y):
        s.append(f'<circle cx="{sx(gA, nx(px)):.1f}" cy="{sy(gA, nx(py)):.1f}" r="2" fill="{TEAL}" opacity="0.5"/>')
    # least-squares guide
    b, a = np.polyfit(pred, y, 1)
    x1v, x2v = float(pred.min()), float(pred.max())
    s.append(f'<line x1="{sx(gA, nx(x1v)):.1f}" y1="{sy(gA, nx(a + b*x1v)):.1f}" '
             f'x2="{sx(gA, nx(x2v)):.1f}" y2="{sy(gA, nx(a + b*x2v)):.1f}" stroke="{DARK}" stroke-width="1.2"/>')
    s.append(f'<text x="{gA["x0"]+4}" y="{gA["y1"]-6}" font-size="8.5" fill="{MUTED}">'
             f'the gate&#8217;s own LLR on the same target: ρ = −0.30</text>')

    # ---------------- B: coverage sweep
    gB = panel_geom(300)
    s += head(gB, "writing only where it is predicted to work", "beats random at every point")
    s += frame(gB, "share of attacks written", "refusal gained per write")
    covs = [0.10, 0.25, 0.50, 0.75, 0.90]
    gate, rmean, rsd = [], [], []
    rng2 = np.random.default_rng(0)
    for cov in covs:
        n = max(1, int(round(cov * len(dP))))
        tau = np.sort(pred)[::-1][n - 1]
        m = pred >= tau
        gate.append(float(dP[m].mean()))
        draws = np.array([dP[rng2.permutation(len(dP))[:int(m.sum())]].mean() for _ in range(400)])
        rmean.append(float(draws.mean())); rsd.append(float(draws.std()))
    ymax = max(gate) * 1.15
    ny = lambda v: v / ymax  # noqa: E731
    band_top = " ".join(f"{sx(gB, c):.1f},{sy(gB, ny(m + 2*sd)):.1f}" for c, m, sd in zip(covs, rmean, rsd))
    band_bot = " ".join(f"{sx(gB, c):.1f},{sy(gB, ny(m - 2*sd)):.1f}" for c, m, sd in reversed(list(zip(covs, rmean, rsd))))
    s.append(f'<polygon points="{band_top} {band_bot}" fill="{MUTED}" opacity="0.18"/>')
    s.append('<polyline points="' + " ".join(f"{sx(gB, c):.1f},{sy(gB, ny(m)):.1f}" for c, m in zip(covs, rmean))
             + f'" fill="none" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 2"/>')
    s.append('<polyline points="' + " ".join(f"{sx(gB, c):.1f},{sy(gB, ny(v)):.1f}" for c, v in zip(covs, gate))
             + f'" fill="none" stroke="{DARK}" stroke-width="1.6"/>')
    for c, v in zip(covs, gate):
        s.append(f'<circle cx="{sx(gB, c):.1f}" cy="{sy(gB, ny(v)):.1f}" r="2.6" fill="{DARK}"/>')
    s.append(f'<text x="{sx(gB, 0.10):.0f}" y="{sy(gB, ny(gate[0]))-8:.0f}" text-anchor="start" font-size="9" '
             f'font-weight="600" fill="{DARK}">+0.24</text>')
    s.append(f'<text x="{sx(gB, 0.90):.0f}" y="{sy(gB, ny(gate[-1]))-8:.0f}" text-anchor="end" font-size="9" '
             f'fill="{DARK}">+0.11</text>')
    s.append(f'<text x="{sx(gB, 0.52):.0f}" y="{sy(gB, ny(rmean[2]))+13:.0f}" text-anchor="middle" font-size="8.5" '
             f'fill="{MUTED}">size-matched random ±2sd</text>')
    for t, lab in ((0.10, "10%"), (0.50, "50%"), (0.90, "90%")):
        s.append(f'<text x="{sx(gB, t):.0f}" y="{gB["y0"]+9}" text-anchor="middle" font-size="8" fill="{MUTED}">{lab}</text>')

    # ---------------- C: the second model
    gC = panel_geom(542)
    s += head(gC, "the same write on a second model", "gemma-2-2b: ρ = +0.08, n.s.", WARM)
    s += frame(gC, "write direction", "refusal rate on attacks")
    qw = [0.30, 0.54, 0.64]
    gm = [0.46, 0.51, 0.53]
    xs = [0.06, 0.5, 0.94]
    nyc = lambda v: v / 0.75  # noqa: E731
    # labels go at the LEFT end, where the two lines are furthest apart (0.30 vs 0.46);
    # at the right end they converge and any label is struck through by its own line.
    for series, col, lab, dy in ((qw, DARK, "Qwen2.5-0.5B", 15), (gm, WARM, "gemma-2-2b", -9)):
        s.append('<polyline points="' + " ".join(f"{sx(gC, x):.1f},{sy(gC, nyc(v)):.1f}" for x, v in zip(xs, series))
                 + f'" fill="none" stroke="{col}" stroke-width="1.6"/>')
        for x, v in zip(xs, series):
            s.append(f'<circle cx="{sx(gC, x):.1f}" cy="{sy(gC, nyc(v)):.1f}" r="2.6" fill="{col}"/>')
        s.append(f'<text x="{sx(gC, xs[0])+6:.0f}" y="{sy(gC, nyc(series[0]))+dy:.0f}" text-anchor="start" '
                 f'font-size="8.5" font-weight="600" fill="{col}">{lab}</text>')
    for x, lab in zip(xs, ["−α", "none", "+α"]):
        s.append(f'<text x="{sx(gC, x):.0f}" y="{gC["y0"]+9}" text-anchor="middle" font-size="8" fill="{MUTED}">{lab}</text>')
    s.append(f'<text x="{(gC["x0"]+gC["x1"])/2:.0f}" y="{gC["y0"]+31}" text-anchor="middle" font-size="8.5" fill="{MUTED}">'
             f'the write barely moves it — no spread to predict</text>')

    s.append("</svg>")
    out = os.path.join(HERE, "fig_dose.svg")
    open(out, "w").write("\n".join(s) + "\n")
    print(f"wrote {out}  ({len(s)} elements)")
    print(f"panel A: {len(pred)} points, ρ={A.sp(pred, y):+.3f}")
    print(f"panel B: gate {[round(v,3) for v in gate]} vs random {[round(v,3) for v in rmean]}")
    bake_instrument(meta, rows, k, atk)


def bake_instrument(meta, rows, k, atk):
    """Figure: the instrument. Why the withdrawn per-prompt claim was the measurement's fault."""
    odd, even = list(range(1, k, 2)), list(range(0, k, 2))
    Do = A.dose(A.rates(rows, "clf", odd))[0][atk]
    De = A.dose(A.rates(rows, "clf", even))[0][atk]
    Dl = A.dose(A.rates(rows, "lex", list(range(k))))[0][atk]
    Dc = A.dose(A.rates(rows, "clf", list(range(k))))[0][atk]
    proxy = A.proxy_lever(rows)[atk]

    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
         f'aria-label="The sampled instrument: split-half reliability, agreement between two scorers, and the proxy rehabilitated" '
         f'font-family="ui-sans-serif,system-ui,sans-serif">',
         f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="{INK}">'
         f'the withdrawn claim was the instrument&#8217;s fault, not the quantity&#8217;s</text>',
         f'<text x="{W/2:.0f}" y="32" text-anchor="middle" font-size="9" fill="{MUTED}">'
         f'Qwen2.5-0.5B, 164 held-out attacks; sampling the outcome instead of decoding it greedily</text>']

    gA = panel_geom(60)
    s += head(gA, "the dose measured twice", f"&#961; = {A.sp(Do, De):+.2f}")
    s += frame(gA, "dose from odd samples", "dose from even samples")
    s += scatter(gA, Do, De, note="Spearman&#8211;Brown at K=16: +0.82")

    gB = panel_geom(300)
    s += head(gB, "two independent scorers", f"&#961; = {A.sp(Dl, Dc):+.2f}")
    s += frame(gB, "dose, refusal lexicon", "dose, rejection classifier")
    s += scatter(gB, Dl, Dc, note="a word list vs a trained classifier")

    gC = panel_geom(542)
    s += head(gC, "the first-token proxy, re-judged", f"&#961; = {A.sp(proxy, Dc):+.2f}")
    s += frame(gC, "first-token proxy lever", "sampled behavioural dose")
    s += scatter(gC, proxy, Dc, note="the greedy rulers agreed at &#8722;0.01", shared=False)
    s.append(f'<text x="{(gC["x0"]+gC["x1"])/2:.0f}" y="{gC["y0"]+31}" text-anchor="middle" font-size="8.5" fill="{MUTED}">'
             f'the proxy was fine; the ruler it was checked with was not</text>')
    s.append("</svg>")
    out = os.path.join(HERE, "fig_instrument.svg")
    open(out, "w").write("\n".join(s) + "\n")
    print(f"wrote {out}")
    print(f"  split-half {A.sp(Do, De):+.3f} | lex vs clf {A.sp(Dl, Dc):+.3f} | proxy vs dose {A.sp(proxy, Dc):+.3f}")


if __name__ == "__main__":
    main()
