#!/usr/bin/env python3
"""Generate the handbook's figures as dependency-free SVG.

Every figure is a *calculated model curve*, parameterized by the measured
anchor points stated in the chapters — not raw data. Re-run to regenerate:

    python3 scripts/make_figures.py

Output: docs/assets/*.svg  (embedded by the chapters).

Pure standard library (math only) so the docs build has no plotting
dependency and the curves are reproducible and reviewable.
"""
import math, os

W, H = 760, 460
ML, MR, MT, MB = 74, 34, 56, 58           # plot margins
PW, PH = W - ML - MR, H - MT - MB
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# palette (readable on both light and dark pages; figures sit on a light card)
C = dict(ink="#1f2937", sub="#475569", grid="#e6e8ec", axis="#334155",
         card="#fcfcfb", edge="#e5e7eb",
         indigo="#4f46e5", teal="#0d9488", amber="#d97706",
         rose="#e11d48", slate="#64748b", green="#15803d",
         breakeven="#dc2626")

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Fig:
    def __init__(self, xlim, ylim, title, xlabel, ylabel,
                 xticks, yticks, xfmt=None, yfmt=None):
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.parts = []
        self.title = title
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xticks, self.yticks = xticks, yticks
        self.xfmt = xfmt or (lambda v: f"{v:g}")
        self.yfmt = yfmt or (lambda v: f"{v:g}")

    def sx(self, x):
        return ML + (x - self.x0) / (self.x1 - self.x0) * PW

    def sy(self, y):
        return MT + (1 - (y - self.y0) / (self.y1 - self.y0)) * PH

    # ---- primitives -------------------------------------------------
    def line(self, x1, y1, x2, y2, color, w=1.5, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{w}"{d} opacity="{opacity}" '
            f'stroke-linecap="round"/>')

    def curve(self, pts, color, w=2.4, dash=None):
        d = "M" + " L".join(f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"'
            f'{da} stroke-linejoin="round" stroke-linecap="round"/>')

    def rect(self, x1, y1, x2, y2, fill, opacity=0.14, stroke=None):
        X, Y = self.sx(min(x1, x2)), self.sy(max(y1, y2))
        wd, ht = abs(self.sx(x2) - self.sx(x1)), abs(self.sy(y2) - self.sy(y1))
        st = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.parts.append(
            f'<rect x="{X:.1f}" y="{Y:.1f}" width="{wd:.1f}" height="{ht:.1f}" '
            f'fill="{fill}" opacity="{opacity}"{st}/>')

    def dot(self, x, y, color, r=4.5):
        self.parts.append(
            f'<circle cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{r}" '
            f'fill="{color}" stroke="white" stroke-width="1.4"/>')

    def text(self, x, y, s, color=None, size=13, anchor="start",
             weight="normal", px=False, italic=False):
        color = color or C["ink"]
        X = x if px else self.sx(x)
        Y = y if px else self.sy(y)
        st = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{X:.1f}" y="{Y:.1f}" fill="{color}" font-size="{size}" '
            f'font-family="{FONT}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}>{esc(s)}</text>')

    def vspan_label(self, x, s, color):
        self.line(self.sx(x), MT, self.sx(x), MT + PH, color, 1.2, dash="3,4",
                  opacity=0.8)

    # ---- frame ------------------------------------------------------
    def frame(self):
        p = self.parts
        # gridlines
        for xt in self.xticks:
            X = self.sx(xt)
            self.line(X, MT, X, MT + PH, C["grid"], 1)
        for yt in self.yticks:
            Y = self.sy(yt)
            self.line(ML, Y, ML + PW, Y, C["grid"], 1)
        # axes
        self.line(ML, MT + PH, ML + PW, MT + PH, C["axis"], 1.4)
        self.line(ML, MT, ML, MT + PH, C["axis"], 1.4)
        # ticks + labels
        for xt in self.xticks:
            X = self.sx(xt)
            self.line(X, MT + PH, X, MT + PH + 5, C["axis"], 1.2)
            self.text(X, MT + PH + 20, self.xfmt(xt), C["sub"], 12, "middle", px=True)
        for yt in self.yticks:
            Y = self.sy(yt)
            self.line(ML - 5, Y, ML, Y, C["axis"], 1.2)
            self.text(ML - 9, Y + 4, self.yfmt(yt), C["sub"], 12, "end", px=True)
        # titles
        self.text(ML, 26, self.title, C["ink"], 16, "start", "600", px=True)
        self.text(ML + PW / 2, H - 16, self.xlabel, C["sub"], 13, "middle", px=True)
        # rotated y label
        self.parts.append(
            f'<text x="18" y="{MT + PH / 2:.1f}" fill="{C["sub"]}" font-size="13" '
            f'font-family="{FONT}" text-anchor="middle" '
            f'transform="rotate(-90 18 {MT + PH / 2:.1f})">{esc(self.ylabel)}</text>')

    def legend(self, items, x, y):
        # items: list of (label, color, dash|None)
        yy = y
        for label, color, dash in items:
            da = f' stroke-dasharray="{dash}"' if dash else ""
            self.parts.append(
                f'<line x1="{x}" y1="{yy}" x2="{x+26}" y2="{yy}" stroke="{color}" '
                f'stroke-width="2.6"{da} stroke-linecap="round"/>')
            self.text(x + 33, yy + 4, label, C["ink"], 12.5, "start", px=True)
            yy += 20

    def svg(self):
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" role="img">\n'
            f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="10" '
            f'fill="{C["card"]}" stroke="{C["edge"]}"/>\n{body}\n</svg>\n')


def save(fig, name):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(fig.svg())
    print("wrote", name)


# ====================================================================
# 1. Prefill/decode: end-to-end M5 speedup vs prefill time-share (Amdahl)
#    S(f) = 1 / (f/Sp + (1-f)/Sd), anchored to measured Sp≈3.5, Sd≈1.2
# ====================================================================
def fig_phase_amdahl():
    Sp, Sd = 3.5, 1.2
    f = Fig((0, 1), (1.0, 3.6),
            "End-to-end Neural-Accelerator speedup vs how prefill-heavy the job is",
            "fraction of baseline time spent in prefill  (0 = pure decode, 1 = pure prefill)",
            "end-to-end speedup  (×)",
            [0, .2, .4, .6, .8, 1.0], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}×")
    f.frame()
    S = lambda x: 1.0 / (x / Sp + (1 - x) / Sd)
    pts = [(x / 100, S(x / 100)) for x in range(0, 101)]
    f.curve(pts, C["indigo"], 2.8)
    # asymptote guides
    f.line(f.sx(0), f.sy(Sd), f.sx(1), f.sy(Sd), C["slate"], 1.2, dash="4,4", opacity=.7)
    f.line(f.sx(0), f.sy(Sp), f.sx(1), f.sy(Sp), C["slate"], 1.2, dash="4,4", opacity=.7)
    f.text(0.99, Sd + 0.09, "decode-only ≈ 1.2×  (bandwidth-bound)", C["slate"], 12, "end")
    f.text(0.50, Sp - 0.12, "prefill-only ≈ 3.5×  (compute-bound)", C["slate"], 12, "middle")
    # workload markers
    for xf, lab in [(0.30, "chat / short prompt"), (0.55, "coding agent"),
                    (0.82, "long-context RAG")]:
        f.dot(xf, S(xf), C["rose"])
        f.text(f.sx(xf), f.sy(S(xf)) - 13, lab, C["ink"], 11.5, "middle", px=True)
        f.text(f.sx(xf), f.sy(S(xf)) + 22, f"{S(xf):.1f}×", C["rose"], 11.5,
               "middle", "600", px=True)
    save(f, "phase-amdahl.svg")


# ====================================================================
# 2. Speculative decoding: speedup vs acceptance rate (crossover at 1×)
# ====================================================================
def fig_spec_crossover():
    f = Fig((0, 1), (0.5, 3.2),
            "Speculative / MTP speedup vs draft acceptance rate",
            "per-token draft acceptance rate  α",
            "decode speedup  (×)",
            [0, .2, .4, .6, .8, 1.0], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}×")
    f.frame()
    # ideal ceiling: E[tokens/cycle] = (1-a^(g+1))/(1-a), zero overhead
    def ideal(a, g):
        if a >= 1: return g + 1
        return (1 - a ** (g + 1)) / (1 - a)
    for g, col in [(1, C["teal"]), (2, C["amber"]), (4, C["green"])]:
        pts = [(a / 200, ideal(a / 200, g)) for a in range(0, 201)]
        f.curve(pts, col, 1.9, dash="5,4")
    # realistic self-MTP (γ=1) with per-cycle overhead c: S=(1+a)/(1+c)
    c = 0.18
    pts = [(a / 200, (1 + a / 200) / (1 + c)) for a in range(0, 201)]
    f.curve(pts, C["indigo"], 2.8)
    # break-even
    f.line(f.sx(0), f.sy(1.0), f.sx(1), f.sy(1.0), C["breakeven"], 1.6, dash="2,4")
    f.text(f.sx(0.30), f.sy(1.0) - 6,
           "break-even 1.0× — below this, speculation is a net loss",
           C["breakeven"], 11.5, "start", px=True)
    # measured operating boxes (labels sit in the open band below y=1)
    f.rect(0.05, 1.03, 0.11, 1.11, C["rose"], 0.20, C["rose"])
    f.line(f.sx(0.08), f.sy(1.03), f.sx(0.08), f.sy(0.80), C["rose"], 1, dash="2,3", opacity=.7)
    f.text(f.sx(0.02), f.sy(0.74), "MoE: α≈5–11%", C["ink"], 11.5, "start", px=True)
    f.text(f.sx(0.02), f.sy(0.63), "→ 1.03–1.11×", C["rose"], 11.5, "start", "600", px=True)
    f.rect(0.85, 1.32, 0.88, 1.57, C["indigo"], 0.22, C["indigo"])
    f.line(f.sx(0.865), f.sy(1.32), f.sx(0.865), f.sy(0.80), C["indigo"], 1, dash="2,3", opacity=.7)
    f.text(f.sx(0.62), f.sy(0.74), "dense 27B: α≈85–88%", C["ink"], 11.5, "start", px=True)
    f.text(f.sx(0.62), f.sy(0.63), "→ 1.32–1.57× (measured)", C["indigo"], 11.5, "start", "600", px=True)
    f.legend([("ideal ceiling γ=1 (1 draft tok)", C["teal"], "5,4"),
              ("ideal ceiling γ=2", C["amber"], "5,4"),
              ("ideal ceiling γ=4", C["green"], "5,4"),
              ("self-MTP with overhead (γ=1)", C["indigo"], None)],
             ML + 14, MT + 14)
    save(f, "spec-crossover.svg")


# ====================================================================
# 3. Prompt-lookup decoding: speedup vs copy fraction (crossover)
# ====================================================================
def fig_pld_crossover():
    f = Fig((0, 1), (0.5, 2.1),
            "Prompt-lookup decoding: speedup vs how copy-heavy the output is",
            "fraction of output tokens that are verbatim runs from context  f",
            "decode speedup  (×)",
            [0, .2, .4, .6, .8, 1.0], [0.5, 1.0, 1.5, 2.0],
            xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}×")
    f.frame()
    # S(f) = (1 + g*f)/(1+o): o=0.43 (fixed proposal tax) -> f=0 gives 0.70;
    # g=1.72 -> f=1 gives ~1.90; crossover at f≈0.25
    o, g = 0.43, 1.72
    S = lambda x: (1 + g * x) / (1 + o)
    pts = [(x / 100, S(x / 100)) for x in range(0, 101)]
    f.curve(pts, C["indigo"], 2.8)
    f.line(f.sx(0), f.sy(1.0), f.sx(1), f.sy(1.0), C["breakeven"], 1.6, dash="2,4")
    # crossover point
    xc = (1 + o - 1) / g  # where S=1  -> (1+g f)=(1+o) -> f=o/g
    xc = o / g
    f.dot(xc, 1.0, C["breakeven"])
    f.text(f.sx(xc) + 6, f.sy(1.0) - 8, f"crossover ≈ {xc*100:.0f}% copy", C["breakeven"], 11.5, "start", px=True)
    # endpoints
    f.dot(0.0, S(0.0), C["rose"]); f.text(f.sx(0.0)+8, f.sy(0.70)+4, "free-form prose ≈ 0.70×", C["ink"], 11.5, "start", px=True)
    f.dot(1.0, S(1.0), C["teal"]); f.text(f.sx(1.0)-8, f.sy(1.90)+4, "copy / retrieval ≈ 1.9×", C["ink"], 11.5, "end", px=True)
    f.rect(0.0, 0.5, xc, 2.1, C["rose"], 0.06)
    f.rect(xc, 0.5, 1.0, 2.1, C["teal"], 0.06)
    f.text(f.sx(xc/2), MT + 16, "net loss", C["rose"], 12, "middle", "600", px=True)
    f.text(f.sx((xc+1)/2), MT + 16, "net win", C["green"], 12, "middle", "600", px=True)
    save(f, "pld-crossover.svg")


# ====================================================================
# 4. KV cache size vs context, full vs windowed, wired-budget crossover
# ====================================================================
def fig_kv_budget():
    # representative dense model: L=48, kv_heads=8, head_dim=128, bf16
    L, Hkv, d, elt = 48, 8, 128, 2
    per_tok = 2 * L * Hkv * d * elt          # bytes/token (keys+values)
    gib = lambda toks: per_tok * toks / 1024**3
    win = 32768
    budget = 18.0                             # GiB left for KV after weights+OS
    xmax = 262144
    f = Fig((0, xmax), (0, 100),
            "KV-cache memory vs context length  (illustrative 48-layer model, bf16)",
            "context length  (tokens)",
            "KV-cache size  (GiB)",
            [0, 65536, 131072, 196608, 262144], [0, 20, 40, 60, 80, 100],
            xfmt=lambda v: f"{v//1024}K" if v else "0",
            yfmt=lambda v: f"{v:g}")
    f.frame()
    # full attention (linear)
    f.curve([(t, gib(t)) for t in range(0, xmax + 1, 4096)], C["rose"], 2.8)
    # windowed / rotating (ramp then flat)
    wpts = [(t, gib(min(t, win))) for t in range(0, xmax + 1, 4096)]
    f.curve(wpts, C["teal"], 2.8)
    # budget line
    f.line(f.sx(0), f.sy(budget), f.sx(xmax), f.sy(budget), C["breakeven"], 1.6, dash="2,4")
    f.text(f.sx(xmax) - 6, f.sy(budget) - 8, f"wired-memory budget ≈ {budget:.0f} GiB",
           C["breakeven"], 11.5, "end", px=True)
    # crossover: full crosses budget
    t_cross = budget * 1024**3 / per_tok
    f.dot(t_cross, budget, C["breakeven"])
    f.text(f.sx(t_cross) + 8, f.sy(budget) + 20,
           f"full attention hits the cap at ≈{t_cross/1024:.0f}K tokens", C["ink"], 11.5, "start", px=True)
    f.text(f.sx(xmax) - 6, f.sy(gib(xmax)) - 8, f"full: {gib(xmax):.0f} GiB @ 256K", C["rose"], 11.5, "end", px=True)
    f.text(f.sx(xmax) - 6, f.sy(gib(win)) + 18, f"windowed @ 32K: flat {gib(win):.0f} GiB", C["teal"], 11.5, "end", px=True)
    f.legend([("full attention (grows with context)", C["rose"], None),
              ("rotating / windowed cache (32K)", C["teal"], None)],
             ML + 14, MT + 14)
    save(f, "kv-budget.svg")


# ====================================================================
# 5. Quantization: decode t/s vs bits-per-weight (bandwidth gain vs dequant tax)
# ====================================================================
def fig_quant_bpw():
    f = Fig((2, 16), (0, 2.6),
            "Why fewer bits ≠ proportionally faster decode",
            "weight precision  (bits per weight)",
            "relative decode throughput  (4-bit real = 1.0)",
            [2, 4, 6, 8, 10, 12, 14, 16], [0, 0.5, 1.0, 1.5, 2.0, 2.5],
            xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}×")
    f.frame()
    # decode time/token = c1*bytes(∝bpw) [+ c2 fixed dequant/weight].
    # throughput ∝ 1/time. ideal ignores dequant (c2=0); real includes it.
    # Normalize both so real(4-bit) = 1.0.
    c1, c2 = 1.0, 1.0
    ref = 1.0 / (4 * c1 + c2)
    real = lambda b: (1.0 / (b * c1 + c2)) / ref
    ideal = lambda b: (1.0 / (b * c1)) / ref
    ip = [(b / 2, ideal(b / 2)) for b in range(4, 33)]
    rp = [(b / 2, real(b / 2)) for b in range(4, 33)]
    f.curve(ip, C["slate"], 2.0, dash="5,4")
    f.curve(rp, C["indigo"], 2.8)
    for b, col in [(2, C["rose"]), (3, C["amber"]), (4, C["teal"]),
                   (8, C["slate"]), (16, C["slate"])]:
        f.dot(b, real(b), col)
    f.text(f.sx(4) + 8, f.sy(real(4)) + 4, "4-bit (DWQ sweet spot)",
           C["teal"], 11.5, "start", "600", px=True)
    # the widening gap = the dequant tax; brace it at 2-bit
    f.line(f.sx(2.12), f.sy(real(2)), f.sx(2.12), f.sy(ideal(2)), C["amber"], 1.6)
    f.text(f.sx(2.35), f.sy((real(2) + ideal(2)) / 2), "gap = dequant tax",
           C["amber"], 11.5, "start", "600", px=True)
    f.text(f.sx(8), f.sy(ideal(8)) - 10, "ideal ∝ 1/bits", C["slate"], 11, "start", px=True)
    f.legend([("ideal: bandwidth only (∝ 1/bits)", C["slate"], "5,4"),
              ("real: bandwidth minus dequant overhead", C["indigo"], None)],
             ML + 14, MT + 14)
    f.text(ML + 14, MT + PH - 10,
           "Illustrative — M5 Neural Accelerators can shift the optimum; measure on your model.",
           C["sub"], 11, "start", px=True, italic=True)
    save(f, "quant-bpw.svg")


# ====================================================================
# 6. Measurement: wall-clock vs output tokens regression (TTFT + decode slope)
# ====================================================================
def fig_ttft_regression():
    import random
    random.seed(7)
    ttft, tpot = 0.42, 1.0 / 70.0            # 0.42 s fixed, 70 tok/s decode
    f = Fig((0, 1200), (0, 20),
            "Separating TTFT from decode: wall-clock vs output tokens",
            "output tokens generated",
            "end-to-end wall-clock  (s)",
            [0, 300, 600, 900, 1200], [0, 5, 10, 15, 20],
            xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    f.frame()
    # scatter
    for _ in range(46):
        n = random.uniform(30, 1150)
        wall = ttft + tpot * n + random.gauss(0, 0.55) + (0.6 if random.random() < .12 else 0)
        f.dot(n, max(0.05, wall), C["slate"], 3.2)
    # fit line
    f.curve([(0, ttft), (1200, ttft + tpot * 1200)], C["indigo"], 2.8)
    # intercept + slope annotations
    f.dot(0, ttft, C["rose"])
    f.line(f.sx(0), f.sy(ttft), f.sx(90), f.sy(3.1), C["rose"], 1, dash="3,3")
    f.text(f.sx(95), f.sy(3.2), "intercept = TTFT ≈ 0.42 s", C["rose"], 11.5, "start", px=True)
    # slope triangle
    x1, x2 = 600, 900
    f.line(f.sx(x1), f.sy(ttft + tpot*x1), f.sx(x2), f.sy(ttft + tpot*x1), C["teal"], 1.6)
    f.line(f.sx(x2), f.sy(ttft + tpot*x1), f.sx(x2), f.sy(ttft + tpot*x2), C["teal"], 1.6)
    f.text(f.sx(x2) + 8, f.sy(ttft + tpot*(x1+x2)/2), "slope = 1/decode-t/s", C["teal"], 11.5, "start", px=True)
    f.text(f.sx(x2) + 8, f.sy(ttft + tpot*(x1+x2)/2) + 16, "→ decode ≈ 70 tok/s", C["teal"], 11.5, "start", "600", px=True)
    f.text(ML + 14, MT + PH - 10,
           "Each dot is one request. The blend of TTFT + decode is only separable across many outputs.",
           C["sub"], 11, "start", px=True, italic=True)
    save(f, "ttft-regression.svg")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_phase_amdahl()
    fig_spec_crossover()
    fig_pld_crossover()
    fig_kv_budget()
    fig_quant_bpw()
    fig_ttft_regression()
    print("done ->", os.path.normpath(OUT))
