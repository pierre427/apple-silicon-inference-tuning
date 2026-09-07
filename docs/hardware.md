# The hardware model

You cannot tune what you don't have a mental model for. This chapter is the
model: what physically bounds each phase of LLM inference on Apple Silicon, and
why **prefill and decode respond to completely different levers**. Almost every
"why didn't that speed up?" surprise on Apple silicon comes from applying a
prefill lever to a decode-bound workload, or vice versa.

## Two phases, two bottlenecks

A single request has two very different regimes:

- **Prefill** (a.k.a. the prompt / context phase, and the source of
  time-to-first-token): the whole prompt is processed in one shot. Every layer
  is a big matrix multiply over `sequence_length × hidden` — hundreds or
  thousands of tokens at once. This is **compute-bound**: arithmetic throughput
  (FLOP/s) is the ceiling.
- **Decode** (the generation phase): tokens are produced one at a time, each
  conditioned on all the KV state so far. Each step multiplies a *single* row
  by the weight matrices. There is almost no arithmetic intensity — the GPU
  spends its time *reading the weights and the KV cache out of memory*. This is
  **memory-bandwidth-bound**.

This split is not an Apple-silicon quirk; it is true on every accelerator. What
*is* Apple-specific is how sharply the current hardware rewards one phase and
not the other.

## The M5 Neural Accelerators: a prefill lever, mostly

The M5 generation is the first Apple GPU with a real matrix datapath. Each GPU
shader core embeds a **Neural Accelerator** — dedicated matrix-multiply
hardware (on the order of ~1,024 FP16 FMAs per cycle per core) reached through
Metal 4 tensor operations. On a 40-core Max part that is 40 of them.

What that buys you, from Apple's own base-M5-vs-M4 measurements with the
accelerators enabled:

| Phase | M5 vs M4 speedup | Why |
|---|---|---|
| **Prefill / TTFT** | **3.33× – 4.06×** | Compute-bound — exactly what the matrix units accelerate. |
| **Decode** | **1.19× – 1.27×** | Bandwidth-bound — the matrix units barely help; the small gain tracks the memory-bandwidth uplift. |

(Apple's published points: prefill 3.33× on a 20B MoE, 3.52× on a 30B MoE,
4.06× on a 14B 4-bit dense; decode 1.19×–1.27×. Apple's own stated mechanism:
"the first token is compute-bound and takes full advantage of the Neural
Accelerators; subsequent tokens are bounded by memory bandwidth.")

Two consequences fall straight out of that table:

!!! warning "The accelerators do almost nothing for your decode t/s"
    If your workload is long *generations* off short prompts, the M5 matrix
    hardware is close to irrelevant — you are bandwidth-bound and the lever you
    want is quantization (fewer bytes per weight to read) and dispatch
    reduction, not the accelerators. If your workload is long *prompts* /
    large agent contexts / RAG, prefill dominates TTFT and the accelerators are
    the single biggest win available.

!!! note "Max-tier decode gains are smaller than Apple's base-tier numbers"
    Apple benchmarked the *base* M5 (memory bandwidth ~153 vs ~120 GB/s, about
    +28%). The Max-tier bandwidth uplift over the previous Max is smaller
    (~12–15%), so **decode gains on a Max part are more modest than the 1.2×**
    Apple quotes, while the compute-bound prefill gains should largely hold.
    Apple did not benchmark the Max tier — treat its exact decode figures as
    unproven above the base part.

We verified the compute side directly with a pure-GEMM microbenchmark (GPU
otherwise idle), comparing a runtime *without* the accelerator kernels against
one *with* them:

| Workload | no-NA runtime | NA runtime | Speedup |
|---|---|---|---|
| fp16, n=4096 | 452 TFLOP/s | 1711 TFLOP/s | **3.8×** |
| bf16, n=4096 | 433 TFLOP/s | 1672 TFLOP/s | **3.9×** |
| fp16, n=8192 | 450 TFLOP/s | 1624 TFLOP/s | 3.6× |
| bf16, n=2048 | 388 TFLOP/s | 760 TFLOP/s | 2.0× |

~3.6–3.9× on large GEMMs — squarely in Apple's claimed prefill band — and less
on smaller matrices, because a small matmul can't fill the accelerators. This
is why the number-one lever in the next chapter is simply *making sure your
runtime actually dispatches to these kernels*: on the exact same hardware, the
difference between reaching them and not is this table.

> **Prior art.** Apple ML Research, *Exploring LLMs with MLX and the Neural
> Accelerators in the M5 GPU* (prefill 3.33–4.06×, decode 1.19–1.27×); the
> Metal tensor-compute path reverse-engineered in arXiv:2606.12765.
> **How we differ.** Independent GEMM microbenchmark on a *Max* part, and we
> profiled decode on the Max tier Apple did not benchmark.
> **Our finding.** *Consistent with prior art* on prefill (3.6–3.9× matches
> Apple's band); *diverges* on Max-tier decode — the smaller Max bandwidth
> uplift means decode gains should fall **below** Apple's base-tier 1.2×.

!!! danger "M4 and earlier have no matrix datapath"
    On M4 and earlier the GPU has no matrix-multiply hardware — fp8 matmul on
    M4 is *emulated* and runs at ~0.94× of fp16, i.e. slightly slower. Advice
    that assumes matrix acceleration (or that lower-precision matmul is
    faster) is simply wrong on pre-M5 parts. Know which generation you're on
    before you copy a tuning recipe.

## Unified memory is the decode ceiling

Apple Silicon shares one pool of memory between CPU and GPU. Two practical
implications dominate serving:

- **Decode speed is roughly `memory_bandwidth / bytes_read_per_token`.** The
  bytes read per token are dominated by the model weights (every decode step
  streams the active parameters) plus the KV cache. This is the physics behind
  two levers elsewhere in the guide: quantization lowers the numerator's
  divisor (fewer bytes per weight → proportionally faster decode, *if* dequant
  is cheap — see [Quantization](quantization.md)), and MoE models read only
  their *active* experts per token, which is why a 35B-A3B MoE decodes far
  faster than its parameter count suggests.
- **You can run out of "wired" GPU memory long before you run out of RAM.**
  macOS caps how much unified memory the GPU may pin. Large models plus a
  growing KV cache push against that cap and the symptom is not an
  out-of-memory error — it is *stalls*, as the system thrashes memory
  pressure. Raising the cap is a one-line lever covered in
  [Runtime & OS knobs](runtime-knobs.md).

Because weights and KV live in the same pool the GPU computes from, there is no
host↔device copy to optimize (a real cost on discrete GPUs). That removes one
class of tuning entirely and puts all the weight on *bandwidth* and *how many
bytes each token has to touch*.

## Dispatch overhead: the third bottleneck nobody mentions

Decode is not only bandwidth-bound; at short context it is also **launch-bound**.
Each decode step is a chain of many small GPU kernels, and each dispatch has
fixed overhead. When the per-kernel work is tiny (batch size 1, one row), that
fixed cost stops being negligible: a meaningful fraction of a decode step can
be *bubbles between kernels* rather than compute.

This is why two techniques in [Serving-level techniques](serving-techniques.md)
exist at all — compiled decode replay (freeze the graph so you stop paying to
rebuild and re-dispatch it) and megakernels (collapse a token's work toward a
single dispatch). It is also why naïvely splitting a fused kernel back into
"clean" segments can be *bit-identical yet 3× slower*: a GPU dispatch is not
free, and paying it per-op per-token is a real, measurable tax. Keep dispatch
count in your mental model alongside FLOPs and bytes.

> **Prior art.** Launch-bound small-batch decode is well known on discrete
> GPUs (the motivation for CUDA graphs and kernel fusion generally).
> **How we differ.** We quantified the per-dispatch cost on Apple Metal and
> what it does to a single-token decode step.
> **Our finding.** *Consistent with prior art* — dispatch overhead is a
> first-order decode cost here too; the fixes ([compiled replay and
> megakernels](serving-techniques.md)) are the Metal analogues of graph
> capture and fusion.

## Putting the model to work

When you're about to tune something, ask three questions in order:

1. **Which phase am I bound by?** Profile prefill and decode *separately*.
   A change to prefill is invisible to a decode-bound workload and vice versa.
2. **If decode: am I bandwidth-bound or launch-bound?** Long context →
   bandwidth (fewer bytes: quantize weights/KV, or use MoE). Short context →
   launch (fewer dispatches: compile/replay, megakernel).
3. **If prefill: am I actually reaching the matrix units?** If not, nothing
   else you do to prefill matters until you are — see the next chapter.

Everything downstream in this guide is an application of these three questions.

## A worked roofline check

You do not need a full profiler to make a useful first classification. For a
single-stream decode, estimate the minimum weight traffic per step, then compare
it with elapsed time. The calculation is deliberately a *bound*, not a claim
that every runtime reads every byte exactly once:

```python
def decode_bandwidth_gib_s(active_weight_gib, tokens_per_second,
                           kv_gib_per_token=0.0):
    """Illustrative lower-bound traffic estimate for one decode stream."""
    bytes_per_step_gib = active_weight_gib + kv_gib_per_token
    return bytes_per_step_gib * tokens_per_second

# Fill these from the model metadata and your measurement.
print(decode_bandwidth_gib_s(
    active_weight_gib=ACTIVE_WEIGHT_GIB,
    tokens_per_second=MEASURED_DECODE_TPS,
    kv_gib_per_token=ESTIMATED_KV_TRAFFIC_GIB,
))
```

For a dense model, `active_weight_gib` is approximately the quantized weight
footprint. For an MoE, use the weights actually selected per token plus the
always-active layers, not total parameters. Compare the result with a measured
memory-bandwidth baseline from the same machine, not a marketing peak. A high
fraction points toward a bandwidth lever. A low fraction does **not** prove
compute-bound: at width one it often means launch gaps, synchronization, or
poor kernel occupancy, which requires a trace to separate.

For prefill, reverse the question. Run a fixed prompt-length ladder and report
prompt tokens divided by prefill time. If throughput rises with prompt length
before flattening, the small cases were not filling the matrix path. If large
cases remain unexpectedly slow, run the GEMM smoke test in
[Environment & build](environment.md) before touching model code.

### How to apply the model

- **Interactive chat with a long system prompt:** measure TTFT first; verify the
  accelerator path, then look for exact prefix reuse.
- **Short prompt with a long completion:** measure steady decode; try weight
  quantization and dispatch reduction, not a prefill-only kernel.
- **Long-context agent with short replies:** split prefill from decode and track
  KV bytes. Prefix caching may remove repeated prefill, while cache sizing may
  prevent a later memory cliff.
- **Many concurrent requests:** measure the scheduler at the intended batch and
  context mix. Batch width can turn launch slack into useful work, but it also
  increases KV residency and changes the compute knee.

### When not to use a roofline estimate

Do not use the estimate as a substitute for profiling when sampling, tool calls,
tokenization, network time, or queueing dominate wall time. It describes the
model forward path. It also cannot identify a correctness regression: a faster
kernel that changes the accepted exactness class still fails the gate.

!!! note "Transfer to llama.cpp and vLLM-on-Metal"
    The names of the counters and cache objects differ, but the classification
    survives intact. In `llama.cpp`, separate prompt evaluation from token
    generation and inspect Metal-offload timing. In a Metal backend for a
    vLLM-style server, separate scheduler/queue time, prefill, and decode, then
    compare active KV blocks and batch width. Unified memory removes an explicit
    host-to-device copy; it does not remove bandwidth, launch, or residency
    limits.

## Sources

- Apple ML Research, *Exploring LLMs with MLX and the Neural Accelerators in
  the M5 GPU* — <https://machinelearning.apple.com/research/exploring-llms-mlx-m5>
- Apple Developer Tech Talk 111432, *Accelerate your ML workloads with the M5
  and A19 GPUs*
- Reverse-engineering of the Metal tensor compute path on Apple GPUs —
  arXiv:2606.12765
- *Production-Grade Local LLM Inference on Apple Silicon* — arXiv:2511.05502
- MLX documentation — lazy evaluation, Metal execution, compilation, and
  performance guidance: <https://ml-explore.github.io/mlx/build/html/index.html>
- MLX-LM — reference generation, cache, quantization, and serving
  implementations: <https://github.com/ml-explore/mlx-lm>
