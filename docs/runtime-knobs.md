# Runtime & OS knobs

With an accelerator-capable runtime in place ([Environment & build](environment.md)),
a small set of OS and runtime settings clear the most common stalls and give
you real control over the memory/quality/throughput trade-off. This chapter is
the short list of knobs that *matter* — and, at the end, the ones that look
like knobs but do nothing, so you can stop turning them.

## Raise the wired-memory limit

Symptom: a large model — especially a big MoE — that runs fine at short context
starts *stalling* as generation proceeds, with no out-of-memory error. That is
memory-pressure thrashing against the cap on how much unified memory the GPU is
allowed to pin ("wire").

The lever is a single sysctl:

```bash
# raise the Metal wired-memory cap (value in MB).
# pick something between your model size and total RAM, leaving headroom
# for macOS and anything else running. Example: 112 GB on a 128 GB machine.
sudo sysctl iogpu.wired_limit_mb=114688
# revert to the default (no explicit cap):
sudo sysctl iogpu.wired_limit_mb=0
```

Guidance:

- Set it **between your resident model size and total RAM**, leaving headroom
  for the OS, a browser, Docker, or a concurrent job. On a 128 GB machine,
  `114688` (112 GB) leaves ~16 GB for everything else under a typical
  serve-plus-desktop load; go to `122880` (120 GB) only if the machine is
  dedicated to a single model.
- The change removes the memory-pressure decode stalls on large models. It does
  not make a well-fitting model faster — it prevents a badly-fitting one from
  falling off a cliff.

Read the effective value back after setting it; a command that printed no error
is weaker evidence than the kernel's current value:

```bash
sysctl -n iogpu.wired_limit_mb
```

Watch both active memory and system pressure during a context ladder. If latency
stays flat and the model already fits with headroom, raising the cap further has
no performance mechanism to exploit.

!!! note "Make it survive reboot"
    `sysctl` is not persistent. To pin it across reboots, install a
    `LaunchDaemon` that runs the `sysctl` at load (`RunAtLoad`). Revert by
    setting the value back to `0` and unloading/removing the daemon. Keep the
    plist under source control so the machine's tuning is reproducible rather
    than a thing someone typed once.

For custom generation loops (your own speculative or batched decode code rather
than the stock generator), set MLX's process-level wired limit before allocating
the model and caches:

```python
import mlx.core as mx

# Bytes, not MB. Derive this from a model + cache budget and leave OS headroom.
process_budget_bytes = int(PROCESS_BUDGET_GIB * 1024**3)
mx.metal.set_wired_limit(process_budget_bytes)

# Load weights and allocate caches only after the limit is in place.
model, tokenizer = load_model()
```

Keep the OS-level cap and process-level request consistent. A process cannot
make memory available that the OS refuses to wire, and an oversized request can
starve the rest of the machine. Check the exact API against your installed MLX;
memory-control names can move between releases.

> **Prior art.** The `iogpu.wired_limit_mb` sysctl and the memory-pressure
> behaviour are documented in the mlx-lm README's memory-limit notes.
> **How we differ.** We turned it into concrete headroom guidance, made it
> persistent across reboots, and scoped the pin inside custom decode loops.
> **Our finding.** *Consistent with prior art, extended* — the knob does what
> the docs say; the value picking, persistence, and per-loop scoping are the
> operational parts the docs leave to you.

## Size the KV cache deliberately

The KV cache grows with context and is, after the weights, the second big
consumer of both bandwidth and wired memory. Two controls:

- **`--max-kv-size N`** (or the equivalent in your serving code) installs a
  **rotating, fixed-size** KV cache. Small values (e.g. `512`) use little
  memory but drop distant context (worse quality on long inputs); larger values
  (`4096`+) keep more context at more memory. Choose based on how much of the
  history your workload actually needs to attend to.
- For long agent contexts on a full-attention model, cap the full-attention KV
  with a rotating cache rather than letting it grow unbounded. The cap is the
  difference between a session that stays responsive and one that slowly
  strangles itself on wired memory.

### Estimate the cache before choosing the cap

For a conventional full-attention transformer, unquantized KV storage is
approximately:

```
2 × layers × kv_heads × head_dim × cached_tokens × bytes_per_element × batch
```

The leading two is for keys and values. Grouped-query attention uses
`kv_heads`, not query heads. This small calculator makes the trade explicit:

```python
def kv_cache_gib(*, layers, kv_heads, head_dim, tokens,
                 bytes_per_element=2, batch=1):
    total = (2 * layers * kv_heads * head_dim * tokens
             * bytes_per_element * batch)
    return total / 1024**3

print(kv_cache_gib(
    layers=NUM_LAYERS,
    kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    tokens=MAX_KV_SIZE,
    batch=MAX_CONCURRENT_SEQUENCES,
))
```

Add the quantized weight footprint, runtime working memory, prefix-cache
entries, and a real OS reserve. Then validate the estimate against MLX active
and peak memory while walking the intended context and concurrency range.
Hybrid recurrent models do not fit this simple formula exactly; inventory their
fixed recurrent state and full-attention layers separately.

**How to apply.** Start from the longest context whose information the
application genuinely needs, multiply by peak concurrent sequences, and see if
the total budget fits. If it does not, choose explicitly among a shorter window,
lower concurrency, smaller weights, or KV quantization. Do not let the first
production traffic spike make that choice for you.

**When not to cap.** A rotating cache changes model semantics once tokens fall
out of the window. Do not use it for tasks that require exact access to the
entire history, or for a model whose cache implementation does not define safe
rotation. Memory stability is not a substitute for answer fidelity.

!!! warning "Rotating cache and KV quantization don't compose"
    A rotating (windowed) KV cache **cannot** be combined with KV-cache
    quantization — the quantized-cache conversion is unimplemented for the
    rotating cache. So "windowed *and* quantized KV" is not currently a valid
    combination; you pick one. See [Quantization](quantization.md) for the
    KV-quant trade-offs.

## Double-buffer custom decode loops

The stock generator overlaps GPU work with the Python-side setup of the next
step by evaluating asynchronously. If you write your own decode loop (common
for speculative decoding or a custom server), you lose that overlap unless you
add it back:

```python
# pipeline committed-token graphs so the GPU isn't idle while Python
# prepares the next step
mx.async_eval(next_tokens)   # kick off without blocking
...                          # do host-side work for the following step
mx.eval(next_tokens)         # materialize when you actually need the values
```

In measured A/B this was throughput-neutral-to-positive and
correctness-identical — the point is not to *lose* the overlap the stock path
already gives you. If your custom loop is slower than `stream_generate` at the
same settings, missing `async_eval` double-buffering is the first thing to
check.

A minimal timing check should compare the whole committed-token loop, not the
latency of `mx.async_eval` itself (which only enqueues work). Materialize the
same boundary in both arms and assert identical committed token ids. Reading a
GPU-produced scalar with `.item()` inside every iteration forces a
synchronization and can erase the overlap you just added.

> **Prior art.** `mx.async_eval` and the async/pipelined evaluation pattern are
> documented in MLX and used by the stock `stream_generate`.
> **How we differ.** We retrofitted it into custom speculative/serving decode
> loops that had lost the overlap.
> **Our finding.** *Consistent with prior art* — the win is not new
> throughput, it's *not forfeiting* the overlap the stock generator already
> has; a hand-rolled loop without it leaves that on the floor.

## The non-levers

Time you don't spend here is time saved. On a GPU-bound inference workload, the
following showed **no** measured benefit and are safe to ignore:

- **CPU core-pinning / P-core affinity / OpenMP thread env vars.** The work is
  on the GPU; pinning CPU threads or tuning `OMP_NUM_THREADS` does nothing for
  decode or prefill throughput here. There is no evidence of benefit.
- **Chasing exotic quant formats to "save bandwidth."** Codebook/IQ-style quant
  concerns are a `llama.cpp` problem; MLX already uses block/affine quant, and
  lower bits-per-weight is not automatically faster anyway (dequant overhead —
  see [Quantization](quantization.md)).
- **Fixed-clock cooldowns between benchmark phases.** A `sleep 420` is the
  wrong instrument — simultaneously too long on a cold machine and too short on
  a hot one. Settle on a measured throughput baseline instead; see
  [Measuring it right](measurement.md).

!!! note "Transfer to llama.cpp and vLLM-on-Metal"
    The memory budget and KV formula are runtime-independent. Map
    `--max-kv-size` to the backend's context/window control, and map
    `async_eval` to whatever queueing primitive lets CPU scheduling overlap GPU
    execution. A vLLM-style paged cache changes allocation granularity and
    fragmentation; it does not make KV bytes free. In `llama.cpp`, also include
    the configured batch and unified-memory offload in the same budget.

## Checklist

- [ ] `iogpu.wired_limit_mb` set with headroom, and made persistent via a
      `LaunchDaemon` if this is a served machine.
- [ ] KV cache sized (`--max-kv-size`) for the workload; long agent contexts
      capped with a rotating cache.
- [ ] Remember: rotating cache **xor** KV quantization, not both.
- [ ] Custom decode loops double-buffer with `async_eval` and scope the
      wired-limit pin.
- [ ] Not wasting time on CPU affinity / OMP knobs.

## Sources

- mlx-lm README — memory limits (`iogpu.wired_limit_mb`), `--max-kv-size`,
  rotating KV cache: <https://github.com/ml-explore/mlx-lm>
- MLX core documentation — `async_eval`, streams, and memory management:
  <https://ml-explore.github.io/mlx/build/html/index.html>
