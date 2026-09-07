# Apple Silicon Inference Tuning

A measured field guide to serving LLMs fast on Apple Silicon (M-series).

Most "run LLMs on your Mac" write-ups stop at *install the runtime, pick a
quant, watch tokens scroll*. This handbook is about the next order of
magnitude: the handful of levers that actually move throughput and latency on
M-series hardware — each backed by a measurement and placed against the prior
art, so you know whether we're confirming published work or contradicting it —
plus the levers that *look* promising and do nothing.

Written **MLX-first** (where the strongest Apple-silicon numbers are today),
with the hardware and methodology framed to generalize to `llama.cpp`,
vLLM-on-Metal, and anything else living inside unified memory and the Metal
command queue.

## The levers, ranked by magnitude

| # | Lever | Typical magnitude | Applies to |
|---|---|---|---|
| 1 | Accelerator-capable runtime (reach the M5 matrix units) | ~3.3–4× | Prefill / TTFT |
| 2 | Exact prefix caching (APC) | up to ~2.4× multi-turn | Repeated / shared context |
| 3 | Megakernel decode lane | ~1.68–1.81× | Short-context decode |
| 4 | Speculative decoding / native MTP | dense ~1.3–1.6×; MoE ~1.03–1.11× | Single-user decode |
| 5 | Weight quantization | ∝ bytes-per-weight saved | Decode / bandwidth-bound |
| 6 | Prompt-lookup decoding | ~1.9× copy, ~0.7× prose | Retrieval / copy workloads |
| 7 | Compiled decode replay | ~1.1× | Decode w/ shape-stable caches |
| 8 | Wired-memory limit + KV sizing | removes stalls | Large models / long contexts |

Full detail, prior-art attribution, and caveats are in the chapters.

## Contents

- **The hardware model** — what bounds each phase, and why prefill and decode need different levers.
- **Environment & build** — reaching the matrix units, and not silently losing them on a rebuild.
- **Runtime & OS knobs** — wired-memory limits, KV sizing, and the knobs that do nothing.
- **Quantization** — weight and KV quant that helps, and the "fewer bits ≠ faster" trap.
- **Speculative decoding & MTP** — when speculation pays, when it hurts, the single-user roofline.
- **Serving-level techniques** — prefix caching, compiled decode replay, megakernels.
- **Measuring it right** — decode-vs-TTFT, thermal settling, GPU tracing, A/B discipline.

## Build the site locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # static site into ./site
```

## House rule

Every figure came off a real machine. Where a claim is an extrapolation, a
vendor number we couldn't reproduce, or an open question, it says so. Each
technique carries a **Prior art / How we differ / Our finding** block stating
whether our measurement was *consistent with*, *extends*, or *diverges from*
the published work.

---

*A public distillation of internal serving work. It documents techniques and
measurements; it is not affiliated with or endorsed by Apple or any model
vendor. Measurements were taken primarily on an M5-generation Max part
(macOS 26); numbers scale with the specific part's memory bandwidth and GPU
core count.*
