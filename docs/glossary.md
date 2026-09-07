# Glossary

These are the load-bearing terms in this handbook. Definitions describe the
mechanism; any performance figure elsewhere remains tied to the machine,
runtime, model, and workload that produced it.

## Active parameters

The weights a model reads for one token. For a dense model this is close to the
whole model. For a mixture-of-experts (MoE) model it includes the always-active
layers plus the experts selected for that token. Active parameters, rather than
total parameters, are the useful first estimate for decode traffic and for
whether a draft model is truly cheap. See [Unified memory is the decode
ceiling](hardware.md#unified-memory-is-the-decode-ceiling).

## Arithmetic intensity

The amount of arithmetic performed per byte moved from memory. High arithmetic
intensity gives compute units enough work to hide data movement; low arithmetic
intensity exposes memory bandwidth. Batched prompt processing has much higher
arithmetic intensity than a single-token, single-stream matrix-vector-style
decode step. See [Two phases, two bottlenecks](hardware.md#two-phases-two-bottlenecks).

## Automatic prefix caching (APC)

A serving policy that stores KV state for token prefixes and restores the
longest compatible prefix on a later request. A repeat hit reuses the whole
prompt; an extension hit restores an earlier conversation and prefills only the
new suffix; shared-prefix hits reuse common instructions across sessions. A
correct cache key binds the token sequence, model revision, positional
semantics, and cache representation. APC saves prefill work; it does not make a
cache miss faster. See [Prefix caching](serving-techniques.md#prefix-caching-apc).

## Compiled decode replay

Tracing a shape-stable decode step once and replaying the captured graph for
later tokens. The lever removes repeated host graph construction and dispatch
setup. It requires fixed array shapes and in-graph state such as the cache write
position; growing caches and Python data-dependent branches cause retracing or
fallback. See [Compiled decode
replay](serving-techniques.md#compiled-decode-replay).

## Compute-bound

A regime in which arithmetic throughput is the limiting resource: making data
movement cheaper does little unless it also reduces compute, while a faster
matrix path can move the result substantially. Long-prompt prefill is the main
compute-bound regime in this handbook. Diagnose it with shape sweeps and kernel
throughput, not CPU utilization. Contrast
[memory-bandwidth-bound](#memory-bandwidth-bound). See [The M5 Neural
Accelerators](hardware.md#the-m5-neural-accelerators-a-prefill-lever-mostly).

## Continuous batching

A scheduler that forms GPU batches from tokens belonging to different live
requests, admitting and retiring sequences as they become ready. It increases
aggregate throughput by supplying width across users, but can increase queueing
latency and KV residency. It attacks some of the same unused-width opportunity
as single-stream speculation, so the gains do not automatically stack. See
[Batching vs single-user](serving-techniques.md#batching-vs-single-user-changes-the-calculus).

## Decode

The autoregressive generation phase after the prompt has been processed. Each
step consumes the latest token plus cached state and produces logits for the
next token. At small batch on Apple Silicon it usually has low arithmetic
intensity and is limited by weight/KV bandwidth or kernel launch overhead.
Measure decode separately from prefill, preferably as time per output token or
steady-state output tokens per second. See [Two phases, two
bottlenecks](hardware.md#two-phases-two-bottlenecks) and [Measuring decode and
TTFT](measurement.md#decode-ts-and-ttft-are-two-different-numbers).

## Dispatch-bound / launch-bound

A regime in which fixed work required to enqueue, schedule, or synchronize GPU
kernels is a meaningful share of the critical path. It is common at batch one
and short context, where individual kernels do too little work to amortize each
launch. Fewer dispatches, compiled replay, batching, or fusion may help. Low
observed bandwidth alone does not prove launch-bound behavior; confirm gaps with
a trace. See [Dispatch overhead](hardware.md#dispatch-overhead-the-third-bottleneck-nobody-mentions).

## Distilled Weight Quant (DWQ)

A learned-quantization method that begins with quantized weights and optimizes
the associated non-quantized parameters, such as scales and biases, against a
teacher model. The stored bit width and kernel format remain the same, so its
purpose is better quality at an equal representation, not a new decode kernel.
Calibration and evaluation must be separate. See [DWQ](quantization.md#dwq-distilled-weight-quant).

## GDN / recurrent state

State updated token by token by a gated-delta-net or another recurrent mixer.
Unlike an attention KV cache, it cannot necessarily be corrected by trimming a
sequence axis. Speculative rejection must restore both KV and recurrent state,
then replay only committed tokens. Any cache-reuse gate must validate their
shared invariant. See [The recurrent-state rollback
trap](speculative-decoding.md#the-recurrent-state-rollback-trap-gdn-qwen3-next).

## KV cache

The stored attention keys and values for tokens already processed. It avoids
recomputing the whole prefix at every decode step. Its size grows with cached
tokens, layers, KV heads, head dimension, element size, and concurrent
sequences. A rotating cache caps growth by discarding old positions, which
changes the accessible context. KV state is both a memory-residency cost and a
bandwidth cost at long context. See [Size the KV
cache](runtime-knobs.md#size-the-kv-cache-deliberately).

## Megakernel

A persistent GPU kernel that schedules most or all of a token's model work
inside one dispatch, rather than launching an ordinary kernel for each operator.
Its purpose is to remove a launch floor. Because threadgroups may coordinate
through a device-wide barrier, resident occupancy becomes a correctness
condition, and device completion must be explicitly acknowledged before state
is committed. It is much more invasive than ordinary operator fusion. See
[Megakernels](serving-techniques.md#megakernels).

## Memory-bandwidth-bound

A regime in which performance is limited by how quickly weights, KV state, and
other tensors can be read or written, rather than by peak arithmetic. Single-
stream decode commonly falls here because each step reuses little work per byte
of model state. Weight quantization can help by reducing bytes, provided its
dequantization cost does not replace the bottleneck. See [Unified memory is the
decode ceiling](hardware.md#unified-memory-is-the-decode-ceiling).

## Multi-token prediction (MTP)

A model head trained to propose more than one future token from the target
model's internal state. In self-speculation, the target's own MTP head drafts
candidates and the target verifies them, avoiding a separate draft model. MTP is
not synonymous with ordinary speculative decoding: it requires a compatible,
trained head and its acceptance can depend strongly on model architecture. See
[Native MTP self-spec](speculative-decoding.md#native-mtp-self-spec-no-draft-model).

## Neural Accelerator (NA)

Dedicated matrix-multiply hardware embedded in each M5-generation GPU shader
core and reached through the relevant Metal tensor operations and runtime
kernels. In this handbook, **NA** does not mean the separate Apple Neural Engine.
The NA path principally accelerates sufficiently large, compute-bound matrix
work such as prefill; it does not remove the memory-bandwidth ceiling on decode.
See [The M5 Neural
Accelerators](hardware.md#the-m5-neural-accelerators-a-prefill-lever-mostly)
and [Prove the accelerators are live](environment.md#prove-the-accelerators-are-live).

## Apple Neural Engine (ANE)

The separate machine-learning accelerator on Apple SoCs, exposed to application
workloads through Core ML and lower-level system interfaces. It is not one of
the M5 GPU's per-core Neural Accelerators: ANE can execute concurrently with a
Metal target, while NA/NAX work shares the GPU's resource budget. The practical
limits are static graph support, placement, submission cost, state residency,
and the size and dtype of every GPU/ANE handoff. See [Heterogeneous GPU + ANE
inference](heterogeneous-inference.md).

## Paged KV / PagedAttention

A memory-management design that stores KV cache in non-contiguous blocks and
maps logical sequence positions to those blocks. It reduces allocation waste
and supports flexible sharing and eviction. It is a foundation for efficient
serving, not automatically prefix caching: APC additionally needs a compatible
content key and a reuse policy. See [Prefix
caching](serving-techniques.md#prefix-caching-apc).

## Prefill

The prompt-processing phase that computes model state for all input tokens
before autoregressive generation begins. Prefill uses matrix operations with a
sequence dimension and is usually compute-bound once prompts are large enough
to fill the hardware. It dominates TTFT for long prompts. Prefix caching avoids
repeating some prefill; it does not accelerate a cold unique prompt. See [Two
phases, two bottlenecks](hardware.md#two-phases-two-bottlenecks).

## Prompt-lookup decoding (PLD)

A speculative method that proposes continuations by finding repeated token
n-grams in the existing prompt or context. It needs no trained draft model and
is lossless when the target verifies proposals under the original decoding
rule. It is useful when output copies or transforms input and becomes overhead
when the continuation has little prompt overlap. See [Prompt-lookup
decoding](speculative-decoding.md#prompt-lookup-decoding-pld).

## Quantization

Representing model state with fewer bits or a lower-cost numerical format. Two
forms must be named separately:

- **Weight quantization** reduces model-weight storage and traffic. It can make
  a larger model fit and can accelerate bandwidth-bound decode, but
  dequantization and kernel choice determine the real speed.
- **KV-cache quantization** reduces attention-cache storage and traffic as
  context grows. It uses a different path, has a different quality surface, and
  can add attention/dequantization work even when answers remain unchanged.

Never report merely “quantized” when the operational question depends on which
state changed. See [Weight quantization](quantization.md#weight-quantization)
and [KV-cache quantization](quantization.md#kv-cache-quantization).

## Speculative decoding

An exact draft-and-verify procedure: a cheaper process proposes several tokens,
the target scores them together, and a rejection rule commits only a valid
prefix plus the target-selected continuation. “Exact” means it preserves the
target sampling distribution when the algorithm is implemented correctly; it
does not mean every seeded run must use identical floating-point reductions.
Speed depends on draft cost, verify cost, committed tokens per round, and state
management—not acceptance rate alone. See [Speculative decoding, MTP &
prompt-lookup](speculative-decoding.md).

## Time per output token (TPOT)

The average steady-state decode time attributable to each generated token,
normally excluding the prompt and the interval already counted as TTFT. Its
reciprocal is steady-state decode tokens per second when units and token counts
match. TPOT is an average; streaming services should also inspect the
distribution of inter-token gaps. See [Also report time per output
token](measurement.md#also-report-time-per-output-token).

## Time to first token (TTFT)

Elapsed time from an accepted request boundary to the first generated token
becoming available. State the boundary: a server-level TTFT may include queueing,
tokenization, cache lookup, prompt prefill, sampling, and transport, whereas a
model-only TTFT may start after tokenization. For a fixed harness, TTFT is the
intercept in a regression of wall time against output-token count. See [Decode
t/s and TTFT](measurement.md#decode-ts-and-ttft-are-two-different-numbers).

## Unified memory

The physical memory pool shared by CPU and GPU on Apple Silicon. Model weights
and caches can remain in one addressable pool, avoiding the explicit
host-to-device copy typical of a discrete accelerator. The pool is still finite
and subject to bandwidth, allocation, compression, and GPU-wiring constraints.
“Unified” does not mean all resident bytes are equally cheap or always GPU-
accessible. See [Unified memory is the decode
ceiling](hardware.md#unified-memory-is-the-decode-ceiling).

## Wired memory

Pages pinned for GPU access so the operating system cannot freely page or
reclaim them. macOS limits how much unified memory can be wired for the GPU. A
large model plus growing caches can hit that policy boundary before total RAM is
exhausted, causing pressure and stalls. Raising the limit prevents that cliff
only when the workload is actually constrained by it, and must leave system
headroom. See [Raise the wired-memory
limit](runtime-knobs.md#raise-the-wired-memory-limit).

## Further public references

- Apple ML Research, *Exploring LLMs with MLX and the Neural Accelerators in
  the M5 GPU*: <https://machinelearning.apple.com/research/exploring-llms-mlx-m5>
- MLX documentation: <https://ml-explore.github.io/mlx/build/html/index.html>
- MLX-LM: <https://github.com/ml-explore/mlx-lm>
- Leviathan et al., *Fast Inference from Transformers via Speculative
  Decoding* — ICML 2023, arXiv:2211.17192.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention* — SOSP 2023, arXiv:2309.06180.
