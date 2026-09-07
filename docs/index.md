# Apple Silicon Inference Tuning

A measured field guide to serving LLMs fast on Apple Silicon.

Most "run LLMs on your Mac" write-ups stop at *install the runtime, pick a
quant, watch tokens scroll*. This guide is about the next order of magnitude:
the handful of levers that actually move throughput and latency on M-series
hardware — each one backed by a measurement and placed against the prior art,
so you know whether we're confirming what others found or contradicting it —
and, just as important, the levers that *look* promising and do nothing.

It is written **MLX-first**, because that is where the strongest Apple-silicon
numbers come from today, but the hardware and methodology chapters generalize
to `llama.cpp`, vLLM-on-Metal, and anything else that has to live inside
unified memory and the Metal command queue.

!!! note "House rule: numbers or it didn't happen"
    Every figure in this guide came off a real machine. Where a claim is an
    extrapolation, a vendor number we couldn't reproduce, or an open question,
    it says so in those words. Each technique carries a **Prior art / How we
    differ / Our finding** block stating whether our measurement was
    *consistent with*, *extends*, or *diverges from* the published work.

## The levers, ranked by magnitude

Read this table top to bottom — that is roughly the order in which the levers
pay off. The magnitudes are **peak, and conditional on the workload named in
the last column**: a prefill lever is invisible to a decode-bound job, and a
prefix-cache lever is worth nothing on a cold, unique prompt. Match the lever
to the phase you're actually bound by (see [The hardware model](hardware.md))
before you reach for it.

| # | Lever | Typical magnitude | Applies to | Chapter |
|---|---|---|---|---|
| 1 | **Accelerator-capable runtime** (reach the M5 matrix units) | **~3.3–4×** | Prefill / TTFT; long prompts, RAG, agents | [Environment & build](environment.md) |
| 2 | **Exact prefix caching (APC)** | up to **~2.4× multi-turn** (e.g. ~33.5 s → 14 s) | Repeated / shared context, multi-turn chat | [Serving-level techniques](serving-techniques.md) |
| 3 | **Megakernel decode lane** (where it applies) | **~1.68–1.81×** | Short-context decode on supported models | [Serving-level techniques](serving-techniques.md) |
| 4 | **Speculative decoding / native MTP** | dense **~1.3–1.6×**; MoE **~1.03–1.11×** | Single-user decode, dense models most | [Speculative decoding & MTP](speculative-decoding.md) |
| 5 | **Weight quantization** | proportional to bytes-per-weight saved | Decode / bandwidth-bound; also fits bigger models | [Quantization](quantization.md) |
| 6 | **Prompt-lookup decoding (PLD)** | **~1.9× copy**, but **~0.7× on prose** | Retrieval / copy-heavy workloads only | [Speculative decoding & MTP](speculative-decoding.md) |
| 7 | **Compiled decode replay** | **~1.1×** | Decode with shape-stable caches | [Serving-level techniques](serving-techniques.md) |
| 8 | **Wired-memory limit + KV sizing** | removes stalls (prevents a cliff, not a multiplier) | Large models / long contexts | [Runtime & OS knobs](runtime-knobs.md) |

Two things that are **not** in the table because they *cost* throughput or do
nothing, and knowing that saves you a weekend:

- **KV-cache quantization** is exact but regressed short-output long-context
  throughput ~2× in our tests — it is a *memory* lever, not a speed one. See
  [Quantization](quantization.md).
- **CPU core-pinning / OMP thread tuning / fixed-clock cooldowns** showed no
  benefit on a GPU-bound workload. See [Runtime & OS knobs](runtime-knobs.md).

If you are diagnosing a machine rather than reading front to back, use the
[Quick reference / cheat sheet](cheatsheet.md). It turns this ranking into a
short decision tree and keeps the commands in one place. The
[Glossary](glossary.md) defines the terms used as gates in that tree.

## The one thing most people get wrong

Levers 1–8 assume you already know **which phase you're bound by**. Prefill is
compute-bound and loves the matrix units (~3.3–4× on M5); decode is
memory-bandwidth-bound and barely feels them (~1.2×). Long prompts → chase
prefill; long generations → chase bandwidth and dispatch overhead. Almost every
"why didn't that speed up?" surprise on Apple silicon is a lever applied to the
wrong phase. Start with [The hardware model](hardware.md).

### A five-minute first pass

Before changing a flag, capture four values from one representative request:

1. prompt tokens;
2. time to first token (TTFT);
3. output tokens; and
4. steady-state time per output token, excluding prefill.

Then repeat with a longer prompt and again with a longer generation. If only
TTFT grows, work the prefill column of the table. If per-token decode time grows
with context, budget KV traffic and memory. If decode is slow even at short
context while GPU intervals contain gaps, investigate dispatch. This small
context/output ladder prevents the common mistake of optimizing a blended
request-time number whose bottleneck you cannot name.

!!! tip "Apply one lever at a time"
    Record the baseline, assert that the intended code path ran, change one
    lever, and rerun the same workload. Only after a lever passes independently
    should you test it in combination. Prefix caching, speculation, and compiled
    replay all alter state ownership; their interactions deserve their own gate.

## Who this is for

Engineers running local or single-tenant inference on Apple Silicon who have
outgrown the defaults: people serving a model behind an API, building agents on
top of a local endpoint, or benchmarking hardware and quant formats and wanting
their numbers to mean something.

## Hardware scope

The measurements here were taken primarily on an **M5-generation Max** part
(40-core GPU with per-core Neural Accelerators, 128 GB unified memory,
macOS 26). Where a finding is M5-specific — anything touching the Neural
Accelerators — it is flagged, because **M4 and earlier have no GPU matrix
datapath at all** and behave differently. Bandwidth-bound decode findings and
the entire methodology transfer down the line, with the obvious caveat that
absolute numbers scale with the specific part's memory bandwidth and GPU core
count.

## How the guide is organized

| Chapter | What it answers |
|---|---|
| [The hardware model](hardware.md) | What bounds each phase of inference, and why prefill and decode need different levers. |
| [Environment & build](environment.md) | Getting a runtime that reaches the matrix units — and not silently losing it on the next rebuild. |
| [Runtime & OS knobs](runtime-knobs.md) | Wired-memory limits, KV sizing, double-buffering, and the knobs that do nothing. |
| [Quantization](quantization.md) | Weight and KV quantization that helps, and the "fewer bits ≠ faster" trap. |
| [Speculative decoding & MTP](speculative-decoding.md) | When speculation pays, when it hurts, and the single-user roofline. |
| [Heterogeneous GPU + ANE inference](heterogeneous-inference.md) | When the separate Apple Neural Engine can overlap a saturated GPU—and why ready work, compact handoffs, and batch preservation determine whether it helps. |
| [Serving-level techniques](serving-techniques.md) | Prefix caching, compiled decode replay, and megakernels. |
| [Measuring it right](measurement.md) | Decode-vs-TTFT, thermal settling, GPU tracing, and A/B discipline. |
| [Glossary](glossary.md) | Precise definitions of the terms that carry the performance argument. |
| [Quick reference / cheat sheet](cheatsheet.md) | Ranked levers, diagnostic questions, commands, and the don't-bother list. |

---

*This handbook is maintained by the lab as a public distillation of internal
serving work. It documents techniques and measurements; it is not affiliated
with or endorsed by Apple or any model vendor.*
