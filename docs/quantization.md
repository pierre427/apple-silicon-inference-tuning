# Quantization

Quantization is the most-reached-for lever in local serving, and the one most
often misunderstood. The intuition "fewer bits per weight means less memory
traffic, therefore faster" is only half true on Apple Silicon: dequantization is
compute, and on a machine where decode is bandwidth-bound but small matmuls can
still be launch- or dequant-bound, the arithmetic of the trade does not always
come out in your favor.

This chapter covers three things the lab has measured or verified against
primary sources: weight quantization (DWQ and `dynamic_quant`, block/affine vs
codebook formats), KV-cache quantization, and the central principle that lower
bits-per-weight does **not** automatically mean faster. See also
[the hardware model](hardware.md) for why prefill and decode respond so
differently, and [measurement methodology](measurement.md) for how these numbers
were taken.

!!! note "Evidence discipline"
    Every figure below comes from a measured run or a cited primary source. Where
    a claim is a transfer from another platform (e.g. a llama.cpp result on M2
    Ultra) rather than something tested on M5, it is labelled as such. Treat the
    unlabelled numbers as MLX-on-M5 measurements and the labelled ones as
    hypotheses.

---

## The central principle: lower bpw ≠ faster

On Apple Silicon the two phases of inference have opposite bottlenecks. Prefill
is compute-bound and, on M5, runs on the Neural Accelerators; decode is
memory-bandwidth-bound. Quantization changes the amount of weight *data* moved
per token, which helps the bandwidth-bound phase — but the dequant step is
itself work, and for the matmul shapes that dominate decode the dequant and
launch overhead can eat the bandwidth saving.

The clearest statement of this comes from the cross-format literature:
**block/affine quantization outperforms codebook/IQ-style quantization on Apple
Silicon** because the codebook lookup adds dequant overhead that dominates the
bandwidth it saves.

!!! warning "This is a transfer, not an M5 measurement"
    That block/affine-beats-codebook result is a **llama.cpp finding on M2
    Ultra**, not an MLX or M5 measurement. MLX already uses block/affine
    quantization natively, so on the MLX stack the comparison is moot in
    practice. It is included here as a *transferable principle only* — lower bpw
    does not imply faster, and dequant overhead can dominate bandwidth savings.
    The M5 Neural Accelerators change the compute/bandwidth balance and **may
    change which format is optimal**; that is an open question the lab has not
    resolved.

> **Prior art.** Benazir & Lin, *Profiling LLM Inference on Apple Silicon: A
> Quantization Perspective*, arXiv:2508.08531 — the block/affine-beats-codebook
> result, a llama.cpp finding on M2 Ultra.
> **How we differ.** MLX already uses block/affine natively; we did **not** re-run
> this comparison on M5.
> **Our finding.** *Diverges from prior art* in scope — the comparison is not
> M5-tested, and the M5 Neural Accelerators may change the optimal format (open
> question). Transfer-only principle.

The practical consequence: choose a quant format for **quality at a fixed
memory budget**, and treat any speed change as something to measure, not
assume. The DWQ result below is a clean example — a strictly better-quality
4-bit model at *identical* decode speed.

### Run a four-arm experiment, not a bit-width race

For each candidate, keep model revision, prompt set, cache policy, sampling,
batch, and output lengths fixed. Compare at least:

| arm | question it answers |
|---|---|
| full precision | What quality and numerics are we trying to preserve? |
| current shipped quant | What is the real operational baseline? |
| candidate at equal bit width | Did the method improve quality without buying more bytes? |
| candidate at equal memory budget | Is it the best use of the memory the service can afford? |

Record weight bytes, peak active memory, prefill rate, decode rate, and a
teacher-forced fidelity metric. A smaller file that decodes slower is not a
speed win; a faster model that crosses the application's quality gate is not a
deployable win.

**When not to quantize further.** Stop lowering the bit width when the model
already fits with the required concurrency and the next format fails quality or
slows the target phase. Quantization is a means to fit and move fewer bytes,
not a score to minimize.

> **Prior art.** arXiv:2508.08531 — dequant overhead can dominate the bandwidth
> a lower-bpw format saves.
> **How we differ.** Reproduction of the principle.
> **Our finding.** *Consistent with prior art* in principle; we extend it with
> the caution that M5 matrix hardware changes the arithmetic.

---

## Weight quantization

### Establish a plain RTN baseline first

Before testing learned or mixed schemes, create a standard affine quant at the
same group size. That control separates a better calibration method from simply
spending more scale metadata:

```bash
# Check flag names against the installed mlx-lm release.
python -m mlx_lm.convert \
    --hf-path <source-model> \
    --mlx-path <rtn-output> \
    --quantize \
    --q-bits 4 \
    --q-group-size 32
```

Load the saved artifact in a fresh process before benchmarking it. Testing only
the in-memory converted object misses serialization metadata and reload-path
defects. Store the source revision, conversion command, and tokenizer files with
the result.

### DWQ (Distilled Weight Quant)

DWQ fine-tunes the non-quantized parameters (the scales and biases) of an
already-quantized model against the full-precision teacher's outputs. The
on-disk format is unchanged — it is still, say, a 4-bit model — so **decode
speed is unaffected**; only quality moves.

Properties (from the MLX `LEARNED_QUANTS.md` documentation, verified in lab
runs):

- **Best at 2–4 bit.** A 4-bit DWQ model reaches roughly the quality of a
  standard 4.6-bit quantization.
- **Ineffective at 6/8-bit** — there is no meaningful quality headroom to
  recover above 4-bit, so do not spend effort DWQ-ing an 8-bit model.
- **Can distill from an 8-bit teacher** rather than the full bf16/fp16 model,
  halving the memory needed to run the distillation.
- **Supports group size 32.**

A lab run distilled a small model (bf16 teacher → 4-bit **group-size 32**
student, ~128 iterations, a few hundred calibration samples) and evaluated
perplexity on a **held-out** slice never seen during distillation:

| model | perplexity |
|---|---|
| bf16 teacher | 6.270 ± 0.123 |
| RTN 4-bit gs64 (standard ship) | 6.965 ± 0.138 |
| RTN 4-bit gs32 (control) | 6.632 ± 0.126 |
| **DWQ 4-bit gs32** | **6.148 ± 0.117** |

- **DWQ vs standard RTN-4bit (gs64): −11.7 % perplexity.**
- **DWQ vs the RTN-gs32 control: −7.3 %** — this isolates the distillation
  contribution from the group-size change. Shrinking the group from 64 to 32
  alone recovers about half the RTN→bf16 gap; DWQ closes most of the rest.
- **Decode speed unchanged:** the RTN-gs32 and DWQ-gs32 models ran at 594 vs
  608 tok/s — the same 4-bit on-disk format, so DWQ is a free quality upgrade at
  equal bits and equal speed.

!!! note "Honest caveat on the perplexity table"
    In that run the calibration and evaluation sets came from the same
    distribution, which flatters DWQ — that is why it edges *below* the bf16
    teacher here. On a general corpus expect DWQ to land **between** RTN and
    bf16, not above it. The transferable, load-bearing result is the RTN→DWQ
    gain **at equal bits and equal speed**, not the specific margin over bf16.

> **Prior art.** MLX `mlx-lm` `LEARNED_QUANTS.md` / `mlx-lm` docs.
> **How we differ.** Independently confirmed the 2–4 bit sweet spot on our own
> models.
> **Our finding.** *Consistent with prior art* — 4-bit DWQ ≈ 4.6-bit standard,
> with nothing worth recovering above 4-bit.

Conceptual invocation (check the exact flags against your installed `mlx-lm`,
which evolves):

```bash
# Distill a 4-bit, group-size-32 student against a full-precision teacher.
# Conceptual — verify flag names against your mlx-lm version.
python -m mlx_lm.dwq \
    --model <teacher-repo-or-path> \
    --bits 4 \
    --group-size 32 \
    --data <calibration-set> \
    --mlx-path <output-path>
```

Use a calibration set matched to your serving domain, and evaluate on a held-out
slice of the same domain — not on the calibration data itself.

### `dynamic_quant`: mixed bits per layer

`mlx_lm.dynamic_quant` assigns **5 bits to the layers it deems sensitive and 4
bits to the rest**, producing a model with a mixed effective bits-per-weight in
the **[4.5, 5.5]** range. It is the natural next step when a flat 4-bit model
loses too much quality but a flat 5- or 6-bit model costs more memory than you
want to spend.

```bash
# Produce a mixed-precision (mostly-4-bit, 5-bit for sensitive layers) model.
# Conceptual — verify flag names against your mlx-lm version.
python -m mlx_lm.dynamic_quant \
    --model <repo-or-path> \
    --mlx-path <output-path>
```

DWQ and `dynamic_quant` are complementary: `dynamic_quant` decides *where* to
spend bits; DWQ improves quality *at* whatever bit width a layer landed on.

> **Prior art.** MLX `mlx-lm` `LEARNED_QUANTS.md`.
> **How we differ.** Reproduction only.
> **Our finding.** *Consistent with prior art* — mixed 5/4-bit lands in the
> [4.5, 5.5] BPW range as documented.

### Ranking quant formats: use teacher-forced reference agreement

When comparing candidate formats, do **not** rank them on a single downstream
benchmark score, and be wary of ranking purely on reconstruction error either —
proxy metrics such as per-tensor reconstruction error can *dissociate* from how
faithfully the quantized model reproduces the full-precision model's behavior.
Rank formats on **teacher-forced agreement with the full-precision model**:
reference perplexity or KL divergence against the fp16/bf16 outputs on held-out
text. That measures the thing you actually care about — how close the quantized
model stays to the original — rather than a proxy that can move the other way.

Here is a compact MLX-oriented KL probe. It compares next-token distributions
on identical teacher-forced input; wrap it in a corpus loop and report a
document-level interval rather than treating one string as a gate:

```python
import mlx.core as mx
from mlx_lm import load

teacher, tokenizer = load("<teacher>")
student, _ = load("<quantized-candidate>")

ids = tokenizer.encode("<held-out evaluation text>")
x = mx.array(ids)[None, :]

teacher_logits = teacher(x)[:, :-1].astype(mx.float32)
student_logits = student(x)[:, :-1].astype(mx.float32)

teacher_logp = teacher_logits - mx.logsumexp(
    teacher_logits, axis=-1, keepdims=True
)
student_logp = student_logits - mx.logsumexp(
    student_logits, axis=-1, keepdims=True
)
teacher_p = mx.exp(teacher_logp)
kl = mx.mean(mx.sum(teacher_p * (teacher_logp - student_logp), axis=-1))
mx.eval(kl)
print("mean_teacher_to_student_kl=", kl.item())
```

Use the same tokenizer and exact token sequence for both arms. Also compute
task-facing quality: teacher agreement can detect fidelity loss, but it cannot
tell you whether a divergence matters to the application.

---

## KV-cache quantization

Quantizing the KV cache (`kv_bits=8`, or 4) trades cache memory for compute in
exactly the way the central principle warns about. In the lab's serving battery
it was shipped **default-off** and deliberately **not promoted to a default**,
for a measured reason.

**What holds:** enabling `kv_bits=8` preserved **exact answers on 4k, 16k, and
32k context probes** — the quantized cache did not corrupt outputs on those
runs, and it composes safely with prefix caching (warm restore stays
token-exact under 8-bit and 4-bit KV).

**What breaks the "free win" story:** **short-output, long-context throughput
regressed by roughly 2×.** Quantized KV is not free — it is a memory-for-speed
trade, and on short-output long-context serving the speed side of that trade is
badly negative.

!!! warning "Quantized KV is a targeted tool, not a default"
    Recommend `kv_bits=8` only where its specific tradeoff fits — for example a
    workload that is genuinely cache-memory-constrained and tolerant of the
    long-context throughput hit. For general short-output serving it is a net
    loss. Measure on *your* output-length and context distribution before
    enabling it.

### A deployment gate for KV quantization

Exercise the corners the service will actually see, not only a short prompt:

1. cold and warm prefix-cache paths;
2. minimum and maximum context;
3. short and long output lengths;
4. each supported batch/concurrency level; and
5. rollback or trimming, if speculation can reject tokens.

For greedy decoding, compare token streams against the unquantized-cache arm.
For sampled decoding, replay fixed random draws or compare logits before the
sampler so randomness does not hide a cache defect. Record cache bytes and
served latency together; otherwise the result cannot express the trade.

> **Prior art.** `mlx-lm` KV-quant support and the general KV-quant literature,
> commonly presented as a near-free memory win.
> **How we differ.** We measured served throughput, not just memory.
> **Our finding.** *Diverges from prior art* — answer-exact on 4k/16k/32k, but
> short-output long-context throughput regresses ~2×; recommend only where the
> trade fits.

### Why quantized KV costs throughput (the mechanism)

It is tempting to assume the cost is a separate dequant launch for K and V that
could be "fused away." That premise is wrong. When the KV cache is quantized,
MLX takes a **separate decomposed attention path**: `Q·Kᵀ` via
`quantized_matmul`, an explicit `softmax`, then `scores·V` via a second
`quantized_matmul`. Dequant is **already fused inside `quantized_matmul`**
(per-tile), so there is no standalone fp16 K/V dequant launch to eliminate.

The real cost is that this decomposed path **materializes the full scores matrix
to DRAM and forfeits flash-attention** — the fused SDPA kernel keeps the online
softmax in-register and never writes scores out. There is no quantized-SDPA
kernel in MLX core. Closing this gap means authoring a quantized
flash-attention kernel (dequant K/V on the fly inside the per-key-block loop,
reuse the online softmax, handle GQA, add a Neural-Accelerator path) — a
multi-week, core-level effort, not a local patch. Until that exists, quantized
KV pays the score-materialization tax.

### Constraint: windowed + quantized KV do not combine

**A `RotatingKVCache` cannot be combined with KV quantization** — its
`toQuantized` is unimplemented. If you rely on a rotating/windowed fixed-size KV
cache (e.g. `--max-kv-size`) for long contexts, you **cannot also quantize that
cache**. This rules out the otherwise-appealing "windowed *and* quantized KV"
combination; you pick one.

> **Prior art.** None — an implementation constraint.
> **How we differ.** Documented the interaction rather than inheriting it from
> prior work.
> **Our finding.** *Extends prior art* — the two do not compose (`toQuantized`
> is unimplemented for the rotating cache).

!!! warning "Verify quantized kernels on your MLX version"
    Quantized paths have had real, silent correctness bugs. Two public examples
    worth knowing: a strided-dequantize defect that corrupted quantized-KV warm
    reads on sliced (non-last-axis) cache views (mlx issue #4370, fixed upstream
    by #4381 in mlx 0.32.2), and a `gather_qmm` kernel bug that corrupted
    quantized MoE expert matmuls for large flattened row counts (fixed by mlx
    #3922). The lesson is not "avoid quantization" but "pin a known-good MLX
    version and keep a regression test" — a silent numeric bug in a quant kernel
    is invisible unless you check outputs against a full-precision reference.

---

## Putting it together

- **Weight quant is where the reliable wins are.** DWQ at 4-bit gives a
  measurable quality gain at zero speed cost; `dynamic_quant` buys quality with a
  little memory. Both keep decode speed governed by the on-disk bit width, not by
  anything exotic.
- **Do not assume fewer bits is faster.** Dequant is compute. On the
  bandwidth-bound decode phase, a lower-bpw format helps only if its dequant
  overhead stays below the bandwidth it saves — and codebook-style formats often
  do not clear that bar (a llama.cpp/M2-Ultra transfer; M5 Neural Accelerators
  may shift the answer).
- **KV-cache quantization is a targeted tool, not a default.** It can be
  answer-exact, but short-output long-context throughput can halve, and it will
  not combine with a rotating/windowed cache. Enable it only where its tradeoff
  fits your workload.
- **Rank formats on teacher-forced fidelity** to the full-precision model, not
  on a single downstream score or on a reconstruction-error proxy.

!!! note "Transfer to llama.cpp and vLLM-on-Metal"
    The format names differ, but the experiment does not. `llama.cpp` readers
    should compare quant families at equal resident bytes and include their
    dequant kernel in timing. A vLLM-style Metal backend should report weight
    format separately from KV-cache dtype and paged-cache capacity. Never fold
    the two kinds of quantization into one label: they save different bytes and
    can move latency in opposite directions.

---

## Sources

Public primary references for the material in this chapter:

- MLX `mlx-lm` documentation — `LEARNED_QUANTS.md` (DWQ and `dynamic_quant`
  behavior, bit-width guidance, group size, teacher options) and the `mlx-lm`
  README (KV-cache memory limits, `--max-kv-size`, `iogpu.wired_limit_mb`):
  <https://github.com/ml-explore/mlx-lm>
- Benazir & Lin, *Profiling LLM Inference on Apple Silicon: A Quantization
  Perspective* — arXiv:2508.08531.
- Apple ML Research, *Exploring LLMs with MLX and the Neural Accelerators in the
  M5 GPU* — the prefill-is-compute-bound / decode-is-bandwidth-bound framing that
  underlies the central principle.
- Public MLX correctness references cited above: mlx issues/PRs #4370 / #4381
  (strided quantized dequantize) and #3922 (`gather_qmm` large-row corruption).

The DWQ perplexity table, the ~2× KV-quant throughput regression, and the
`dynamic_quant` bit ranges are lab measurements taken on M5-class hardware with
the MLX stack; reproduce them on your own model and workload before relying on
the exact figures.
