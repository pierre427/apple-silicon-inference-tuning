# Quick reference / cheat sheet

Use this page at the machine. The full chapters contain the evidence,
limitations, and exactness gates; this is the short route from symptom to next
measurement.

## First: name the bottleneck

**Question zero: which phase am I bound by?** Separate prompt processing from
steady generation before choosing a lever.

| Ask | Evidence to collect | If yes, start with |
|---|---|---|
| Is the request dominated by prompt processing / TTFT? | Fixed-output prompt-length ladder | Accelerator path, then APC for repeated prefixes |
| Is steady decode slow while estimated weight traffic approaches measured bandwidth? | TPOT plus active weight/KV bytes | Weight quantization; reduce KV traffic or active bytes |
| Is short-context, width-one decode slow with gaps between GPU encoders? | Short encoder-interval trace | Compiled replay, batching, or a qualified megakernel |
| Does latency worsen as context grows? | Context ladder, active/peak memory, KV estimate | KV sizing, wired-memory headroom, attention path |
| Are many requests queued? | Queue delay, batch width, throughput knee, KV residency | Continuous batching and two-ceiling admission |
| Does output substantially copy the prompt? | N-gram overlap and PLD committed tokens/cycle | Prompt-lookup decoding |

If you cannot answer the first question, do not tune yet. Measure TTFT and TPOT
separately; one blended request throughput cannot identify the phase.

## Ranked levers

This repeats the front-page order. Magnitudes are the lab's measurements or the
explicitly attributed public measurements described in the linked chapters;
they are conditional on the workload in the fourth column.

| # | Lever | Typical magnitude | Applies to | Go to |
|---|---|---|---|---|
| 1 | **Accelerator-capable runtime** | **~3.3–4×** | Prefill / TTFT; long prompts | [Environment & build](environment.md) |
| 2 | **Exact prefix caching (APC)** | up to **~2.4× multi-turn** (e.g. ~33.5 s → 14 s) | Repeated / shared context | [Serving-level techniques](serving-techniques.md#prefix-caching-apc) |
| 3 | **Megakernel decode lane** | **~1.68–1.81×** | Supported short-context decode paths | [Megakernels](serving-techniques.md#megakernels) |
| 4 | **Speculative decoding / native MTP** | dense **~1.3–1.6×**; MoE **~1.03–1.11×** | Single-user decode; dense models most | [Speculative decoding](speculative-decoding.md) |
| 5 | **Weight quantization** | proportional to bytes-per-weight saved | Bandwidth-bound decode; model fit | [Quantization](quantization.md) |
| 6 | **Prompt-lookup decoding (PLD)** | **~1.9× copy**, but **~0.7× on prose** | Retrieval / copy-heavy output only | [PLD](speculative-decoding.md#prompt-lookup-decoding-pld) |
| 7 | **Compiled decode replay** | **~1.1×** | Shape-stable, width-one decode | [Compiled replay](serving-techniques.md#compiled-decode-replay) |
| 8 | **Wired-memory limit + KV sizing** | removes stalls; prevents a cliff | Large models and long contexts | [Runtime & OS knobs](runtime-knobs.md) |

Do not multiply the rows together. Levers overlap, and some are mutually
exclusive. Qualify each alone, then test the interaction on the production
configuration.

## Key commands

### Environment identity

```bash
python - <<'PY'
import platform
import sys
import mlx.core as mx

print("python=", sys.executable)
print("macos=", platform.mac_ver()[0])
print("mlx=", mx.__version__)
print("metal=", mx.metal.is_available())
PY
```

### Neural Accelerator smoke test

Run the `na_check.py` from [Environment & build](environment.md#prove-the-accelerators-are-live)
on an idle GPU:

```bash
python na_check.py
```

Use a floor qualified on the same machine and deployment environment. A model
request is too confounded to prove the matrix path.

### Wired-memory limit

```bash
# inspect
sysctl -n iogpu.wired_limit_mb

# set: choose MB from a model + cache budget and leave system headroom
sudo sysctl iogpu.wired_limit_mb=114688

# revert to the OS default
sudo sysctl iogpu.wired_limit_mb=0
```

This removes memory-pressure stalls only when the workload hits the wiring cap.
It is not a throughput multiplier for a model that already fits.

### Bound the rotating KV cache

```bash
mlx_lm.generate --model <model> --max-kv-size 4096 --prompt <prompt>
```

Treat the value as an application semantics choice, not merely a memory flag:
older context falls out of the attention window. A rotating cache and quantized
KV do not currently compose in the MLX path described by this handbook.

### Capture a short GPU trace

Use Instruments or `xctrace` only after the model reaches steady decode. Keep
the recording window short, put the command under an external process-group
watchdog, and count exported `metal-gpu-intervals` rows before calling the trace
valid. See [GPU-trace capture](measurement.md#gpu-trace-capture-on-macos-short-windows-count-the-rows)
for the required safety and interpretation details.

## Minimum benchmark record

For every arm, retain:

- source and model revision;
- interpreter, OS, MLX/runner version, and backend path;
- weight format, KV format, cache/window, batch, and concurrency;
- exact prompt-set identity and prompt/output token counts;
- sampler and seed or deterministic comparison method;
- TTFT, TPOT, raw per-request observations, and settle state;
- mechanism counters proving the requested path ran; and
- output exactness class and its gate.

For speculation, also retain proposed width, accepted prefix length, committed
tokens per cycle, and draft/verify/rollback time. For APC, retain
`cached_tokens`, match kind, cache format, and publish-back state.

## Fast decision rules

### Long prompt, short answer

1. Verify the accelerator-capable runtime with `na_check.py`.
2. Measure model-only TTFT separately from queue and tokenization.
3. If prefixes repeat, assert APC extension and shared-prefix hits.
4. If every prompt is unique, optimize cold prefill; APC has nothing to reuse.

### Short prompt, long answer

1. Measure TPOT after warmup.
2. Estimate active weight bytes per step and compare with measured bandwidth.
3. If bandwidth-bound, test weight quantization at a quality gate.
4. If launch-bound, test replay/fusion or batch width.
5. Try speculation only after timing bare verify widths and the full cycle.

### Long context, short answer

1. Estimate KV bytes for context × concurrency.
2. Measure active/peak memory and context-dependent TPOT.
3. Choose among a rotating window, lower concurrency, smaller weights, or
   targeted KV quantization.
4. Re-test cold, warm, trim, and rollback paths.

### Many concurrent users

1. Sweep offered load, batch width, and context mix.
2. Cap admission by both resident-memory budget and measured throughput knee.
3. Report queue latency separately from model execution.
4. Compare speculation with batching at equal offered load; width does not
   automatically stack.

## Do not bother—unless evidence changes the premise

- **CPU core pinning, P-core affinity, or OpenMP thread tuning** for a workload
  already proven GPU-bound.
- **A source build merely because it is a source build.** Use it for a specific
  feature or fix, then run the same smoke tests as a wheel.
- **Lower bits on faith.** Dequantization and kernel choice can erase the bytes
  saved. Measure equal-memory and equal-bit controls.
- **KV quantization as a default speed lever.** It is a targeted memory trade
  and must pass long-context throughput and correctness gates.
- **Prompt lookup on free-form prose.** It needs reusable prompt spans.
- **An external draft near the target's active size**, especially for MoE where
  verification can touch a wider union of experts.
- **A fixed cooldown sleep.** Settle on a two-sided functional throughput band.
- **Long shader-profiler recordings.** Short windows plus nonzero row counts are
  the dependable path.
- **A megakernel before proving launch gaps are material.** It is a specialized
  runtime and correctness burden.
- **Combining individually fast levers without a composition matrix.** Shared
  cache state is where silent fallbacks and rollback bugs hide.
- **One blended tokens/s number.** It cannot tell TTFT from decode or queueing.

## Stop conditions

Stop and investigate instead of accepting a speed number when:

- the candidate's mechanism counter is zero;
- the accelerator smoke test falls into the slow cluster;
- the thermal-settle ceiling fires;
- a trace exports no usable rows;
- greedy tokens or the predeclared fidelity gate fail;
- a cache hit cannot prove how many tokens were restored;
- a recurrent rollback restores KV but not recurrent state;
- the candidate retraces inside the measured decode region; or
- a win disappears on the production-config multi-turn or loaded-service test.

The longer explanations live in [The hardware model](hardware.md),
[Measuring it right](measurement.md), and the [Glossary](glossary.md).
