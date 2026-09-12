# Serving-level techniques

Most tuning advice targets the model: the quantization format, the kernel, the
attention variant. But a large share of end-to-end latency lives in the *serving
loop* — the code that owns the KV cache across turns, decides what to recompute,
and dispatches work to the GPU. This chapter covers four levers that live there:
**prefix caching**, **compiled decode replay**, **megakernels**, and the way
**batching versus single-user** service changes which of them is worth adopting.

The recurring lesson: a serving lever is only as good as the correctness gate
that qualifies it, and levers that each win alone can lose — or silently
misbehave — when composed. Numbers below come from a lab serving stack on an
Apple M5 Max (40-core GPU, 128 GB unified memory); see
[the hardware model](hardware.md) for why decode is bandwidth-bound and prefill
is compute-bound on this class of machine, and [measurement](measurement.md) for
the gating discipline these results depend on.

---

## Prefix caching (APC)

Automatic prefix caching (APC) reuses the attention KV state of a shared prompt
prefix across requests, so a multi-turn conversation or a repeated system prompt
does not re-run prefill from scratch. On Apple Silicon, where prefill is the
compute-bound half of the workload, a cache hit is one of the largest
whole-conversation wins available.

The correctness question is whether a *warm restore* — decoding from the cached
state — produces the same tokens as a cold prefill of the same text. Measured
carefully, it does:

- **Warm restore is attended-state faithful.** In one probe the warm arm held
  the top-1 token for 202 consecutive positions before forking at a sub-nat
  near-tie, with no change to the typed output.
- **Zero leakage from discarded branches.** When speculative rollback trims the
  cache to the last accepted token and zeroes the ragged tail, KV state left in
  the pool from a discarded branch is *byte-identical* to a clean warm run — not
  "small drift," exactly zero. Content-addressed prefix caches that never
  re-inject removed tokens are the class the KV-leakage literature explicitly
  exempts.

!!! warning "Exactness can be repeat-only — assert the cache hit, not the stream"
    A subtle defect: one runtime folded a hash of the text embeddings into the
    exact-APC cache key. Byte-identical *repeats* still hit, so a test that only
    checked "warm turns still stream" passed — but a multi-turn *extension* or a
    new session never hit at all, silently falling back to cold prefill. Never
    salt an APC key on anything the token ids already determine, and make the
    gate assert `cached_tokens > 0` on new-session, extension, **and** repeat —
    not just that output was produced.

### Make the cache contract observable

A prefix-cache key must bind every property that changes the meaning or layout
of stored state, while avoiding values already determined by the token ids. A
simple content-addressed key can start like this:

```python
import hashlib
import json

def prefix_key(*, model_revision, cache_format, position_config, token_ids):
    payload = {
        "model_revision": model_revision,
        "cache_format": cache_format,          # dtype, layout, quant settings
        "position_config": position_config,    # RoPE/scaling semantics
        "token_ids": token_ids,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
```

Production caches usually hash blocks so they can choose the longest matching
prefix, but the invariant is the same: equal keys imply cache-compatible model
state. Do not include request ids, timestamps, raw embedding hashes, or other
per-request salt; those destroy cross-request reuse without making the state
safer.

Every response should expose enough internal metrics to prove the mechanism:

```text
prompt_tokens       total tokenized prompt length
cached_tokens       tokens restored rather than prefetched
cache_match_kind    miss | repeat | extension | shared-prefix
cache_format        layout/dtype compatibility version
cache_publish       whether committed state was stored for a later turn
```

Test all four match kinds. For an extension, assert that the restored prefix
ends at the expected token boundary and only the suffix is prefetched. For a
miss, assert `cached_tokens == 0`; false hits are correctness bugs, while false
misses are performance bugs.

### APCv2: make every model declare its cache layout

An attention-only APC entry can look like one object even when it contains one
KV cache per layer. That abstraction stops being sufficient once a model mixes
attention, recurrent state, rotating windows, QSA summaries, and a persistent
MTP head. Those pieces have different token geometry, rollback behavior, and
eviction value. Treating them as one anonymous object either forces expensive
copies or invites partial restores that combine incompatible generations.

The safer migration pattern is a separate **APCv2** contract:

- Leave legacy APC unchanged for models that do not opt in.
- Require each v2 model family to declare a versioned layout name. A layout is
  code plus an invariant, not merely a string stamped on old cache objects.
- Keep one atomic, committed token boundary for correctness, but own and account
  for storage by `plane / layer / segment`.
- Split ordinary attention KV and QSA summaries into immutable prefix and short
  mutable-tail segments; describe rotating attention as a window and recurrent
  GDN state as a state segment.
- Store persistent MTP state as its own plane. A stale required target segment
  rejects the whole boundary; a stale MTP segment disables speculation while
  retaining a valid target restore.

This separates *storage granularity* from *semantic atomicity*. Segments may be
shared copy-on-write, evicted, compressed, or placed differently, but a request
never assembles a target state from different committed generations.

#### Capture the retry boundary before speculation starts

Continuous self-MTP needs more than the completion cache. For a prompt of
length `P`, capture a retry checkpoint immediately before processing the final
prompt token: target coverage `P-1`, draft coverage `P-2`, plus the hidden seed
and RNG position that will predict token `P-1`. This is the last boundary at
which no proposal is in flight. Publish a later completion checkpoint only
after the speculative transaction commits or rolls back.

```python
# Conceptual APCv2 publication protocol.
retry = checkpoint(
    target_coverage=P - 1,
    mtp_coverage=P - 2,
    hidden_seed=hidden,
    rng_state=rng_state,
    committed=True,
)
apc_v2.publish(retry)

proposal = mtp.propose(retry)
accepted, correction = target.verify(proposal)
restore(retry)                    # return to the committed generation
replay(accepted, correction)      # advance only durable tokens
discard(proposal[len(accepted):]) # rejected drafts never enter APC
apc_v2.publish(checkpoint(committed=True))
```

Do not prune an exact MTP checkpoint merely because the target KV cache can be
trimmed to the same nominal token count: recurrent/draft state is generally
exact-boundary-only. A publication API should reject any checkpoint not
explicitly marked committed.

#### What the first Qwen4-class gate showed

At a 16,378-token repeated prompt and 256-token greedy completion, the broken
self-MTP path restored zero prompt tokens: **11.304 s TTFT, 16.911 s wall, and
45.489 decode tok/s**. APCv2 restored **16,377 / 16,378** prompt tokens and ran
at **0.088 s TTFT, 5.573 s wall, and 46.486 decode tok/s**. The cache held 172
segments across two retained boundaries: GDN state, attention KV, QSA summary,
and MTP draft planes. Each measured request proposed 203 draft tokens, accepted
153, and discarded 50. A forced-zero-acceptance test separately proved that a
rejected proposal could not mutate the stored retry checkpoint.

The segmented descriptor-COW snapshot is the default for this v2 boundary,
with independent deep copy as a compatibility fallback. Snapshot bookkeeping
was sub-millisecond in the measured bracket and did not create a wall-time or
decode-throughput regression. These numbers repair a false cache miss; they do
not show that segmentation makes a cache hit intrinsically faster than an
already-correct monolithic restore.

!!! warning "Segmentation is not zero-copy consumption yet"
    The current v2 gate makes ownership, accounting, and invalidation
    canonical, but reconstructs the model's native concrete cache graph on
    restore. A later consumer can read physically discontiguous segments
    directly only after its attention/recurrent kernels are explicitly wired
    for that representation. Until then, measure coalescing/materialization
    cost and do not advertise descriptor-COW as a free physical branch.

**Migration gate per model family:** declare the layout; enumerate every state
plane and token geometry; test miss, repeat, extension, trim, rejection at
positions zero through draft width, and completion publication; then run cold,
warm, live-continuation, and multi-turn performance brackets. Opt in only after
the model passes—never infer compatibility from class names or cache offsets.

### Classify the divergence honestly

Warm-versus-cold greedy output can differ without any restore fault. Two benign
causes dominate, and both must be ruled out before calling a divergence a bug:

- **Kernel-shape numerics.** The fused attention kernel picks its Metal variant
  and block count from the key-sequence length. A restored state that is one
  chunk-width different from a fresh prefill (e.g. a 2047- vs 2048-wide final
  chunk) reorders the softmax reduction and can flip a token at a near-tie —
  even on a bit-identical cache. Forward *width* alone (decoding one token vs
  verifying three) can flip a token on a completely fresh cache.
- **Quantized logprobs.** Returned logprobs are often bf16-quantized, so an
  absolute tolerance below ~0.125 nats at logit magnitude ~16 is unreachable by
  construction. "1.375-nat drift" turned out to be six of ~1000 values, all at
  ~1e-8 probability.

The rule: compare warm restore against *its own store input* first (that must be
bit-exact), and only then against a fresh prefill — where seam-position
near-ties are expected, not defects. The warm arm is the stable one across
engine changes; when a digest moves, it is usually the cold arm that moved.

> **Prior art.** Kwon et al., *Efficient Memory Management for Large Language
> Model Serving with PagedAttention* (SOSP 2023, arXiv:2309.06180), established
> paged KV management for serving; vLLM's automatic prefix caching adds
> content-addressed sharing of matching prefix blocks. Paging and prefix reuse
> are related memory mechanisms, but they are not the same feature.
> **How we differ.** We require exact warm-restore (token-identical) with verified zero cross-request leakage, and classify where exactness holds vs a salt/text-only caveat.
> **Our finding.** *Extends prior art* — most APC implementations optimize hit-rate; we add a bit-exactness + leakage-safety classification.

---

## Compiled decode replay

Each decode step normally rebuilds its compute graph on the host every token.
Framework tracing (MLX's `mx.compile`) can trace that step **once** and *replay*
the traced graph on every subsequent token, deleting the per-token host graph
construction. On a large MoE model at batch size M=1 the replayed step ran
**~1.109x faster** than the pipelined rebuilt graph (roughly 129 → 143 tok/s at
1K context), emitting **one trace per completion**.

The precondition is a **shape-stable cache**. Every array the step touches must
keep its shape: KV stored as a preallocated ring buffer with the write position
as an *array* input (not a Python int), fixed-size ledgers, and no `.item()` or
shape-branching inside the step. A cache that grows by concatenation in blocks
breaks the trace and forces a rebuild.

```python
import mlx.core as mx

# Shape-stable cache: preallocated ring, position is an ARRAY, never a Python int.
class RingKVCache:
    def __init__(self, n_layers, n_kv_heads, head_dim, capacity):
        shape = (n_layers, 1, n_kv_heads, capacity, head_dim)
        self.keys   = mx.zeros(shape)
        self.values = mx.zeros(shape)
        self.offset = mx.array(0)          # in-graph, so the trace stays valid

# The step is a pure function of (token, state) -> (logits, state).
# mx.compile traces it once; later calls replay the traced graph.
@mx.compile
def decode_step(token, keys, values, offset):
    logits, keys, values = _forward(token, keys, values, offset)
    return logits, keys, values, offset + 1   # thread state as return values
```

In real code, treat `(model revision, dtype, batch, cache-capacity bucket)` as
the trace identity. Warm each intended bucket, count trace creation, and fail a
benchmark that retraces inside the measured region. Choose the smallest bucket
that safely holds the request rather than one maximum slab for every context.

**How to apply.** First make the eager step a pure state transition, then add a
fixed cache, then compile. Compare each intermediate arm; otherwise a regression
from padding can be mistaken for a compiler regression. Return mutated arrays
from the compiled function and commit them only after successful evaluation.

**When not to.** Skip replay for highly dynamic shapes, wide speculative verify,
or contexts where the padded attention cost already exceeds saved host work.
Do not put Python-side sampling, logging, or request objects inside the compiled
step merely to enlarge the captured region.

!!! danger "A fast-math kernel can make replay numerically diverge"
    Early replay diverged from stock decode. The cause was *not* the replay
    machinery — it was the compiler fusing `sigmoid` down to a fast-exp
    approximation. On 3,840 real activations the fused form sat **1.52x further
    from an fp64 reference** than the eager op (max 4 ULP), enough to flip greedy
    tokens by ~40 layers deep. The fix was a `precise::exp` variant expressed
    through a custom Metal kernel that is *opaque to fusion*, restoring
    bit-identical arithmetic. After it, 128 greedy steps were **bit-identical to
    the stock cache**, with one trace serving all 128 replays. Lesson: pick a
    statistic that *separates the variants you have* — a pooled RMS said 1.0012
    and would have called this a benign reorder, because the fast exp only bites
    at large magnitude and RMS dilutes it.

> **Prior art.** MLX `mx.compile` graph capture/replay over shape-stable KV caches (e.g. RingKVCache).
> **How we differ.** We applied compile+replay to the single-token (M=1) decode step with a shape-stable cache and hit a compiled fast-exp/sigmoid defect that a precise-exp kernel fixed.
> **Our finding.** *Extends prior art* — ~1.109x at M=1, bit-identical after the exp fix; documents a real correctness pitfall of naive compilation.

### The gain is short-context; the ring buffer is a cost

The host work compiled replay removes is roughly *fixed*, while GPU work grows
with context, so the win narrows as the prompt lengthens. Over a context ladder
the whole-conversation gain was **~1.04–1.10x, bit-identical to stock** up to
about 16K, then crossed over: ~1.003x at 128K and ~0.967x at 256K, where the
padded ring slab costs more than the host time it saves.

Two caveats worth internalizing:

- **The ring cache is itself a tax.** An explicit `[1,1,N,capacity]` mask is a
  slower attention path than a plain `mask=None`, so the shape-stable cache adds
  a few percent on its own; compiled replay has to earn that back before it nets
  out positive. Oversized capacity buckets are worse — a slab sized to 32768 for
  ~16K live columns tripped a pathological attention block-selection path (+37%
  in one case) that had nothing to do with the ring itself.
- **M>1 does not benefit.** At verify width 3 the lever is a wash at short
  context and a loss at long — it is a single-user, M=1 lever, not a
  speculative-verify one. See [speculative decoding](speculative-decoding.md)
  for why the wide-verify path has a different cost structure.

### Composing replay with prefix caching

Compiled replay and APC want the same thing — to skip work between turns — but
naively they don't stack: an APC hit returns a *private* deepcopy of the cache,
and if the replay path only recognizes caches it "owns," it declines on every
hit and reverts to the slow rebuilt graph. In one six-turn agent measurement the
default configuration actually *regressed* to 0.729x because the compiled path
never published its state back and so never hit. The fix is to let the hit reuse
the compiled step and publish the plain (non-ring) KV form back into the cache —
never store a ring buffer in APC, which doubles slab bytes. That recovered
**1.077x whole-conversation with identical output text**.

!!! note "A single-request gate cannot see a multi-turn regression"
    The 0.729x regression above was invisible to every single-request
    qualification — it only appears across turns. A **multi-turn harness**
    (here, six turns) has to be part of the gate for any lever that touches
    cross-turn state.

---

## Submit each layer as you build it

A lazy array framework builds a whole graph before it runs anything. For a
48-layer decoder that means the host spends its time building nodes while the
GPU has no work, then the GPU spends its time executing while the host has
nothing left to do. The two costs alternate when they could overlap.

The fix is one line inside the layer loop: after each layer, issue a
**non-blocking** evaluation of the running hidden state so the work built so far
is submitted while the host builds the next layer.

```python
for layer, layer_cache in zip(self.layers, cache):
    hidden = layer(hidden, ...)
    if pipeline_submits:          # decode/verify widths only
        mx.async_eval(hidden)     # submit, do not block
```

The blocking variant (`eval`) destroys the benefit: it waits, which is the
serialization you are trying to remove. The non-blocking variant forces *when* a
value is computed, never *what* it is, so the token stream is bit-identical.
Verify that with a digest, not by inspection.

**Measured, 4-bit MoE model with a native MTP head, medians of three settled
repetitions, every arm bit-identical:**

| Context | Plain decode | Self-speculative decode |
|---|---:|---:|
| 1K | **+19.9%** | **+20.2%** |
| 16K | **+20.8%** | **+20.6%** |
| 64K | **+24.0%** | **+15.9%** |

Three things make this worth its own section.

**Gate it by row count.** Apply it only when the forward is narrow — a
single-token decode step or a small speculative verify slab. A prefill slab of a
few thousand rows already saturates the GPU, and its host cost is amortized over
those rows, so there is no idle to fill and only submission overhead to add. A
threshold on `batch × sequence` around 64 is a reasonable default.

**Submitting more often was better, not worse.** The obvious worry is command
buffer overhead: 48 submissions per token instead of one. A variant submitting
every fourth layer, 12 per token, was **worse in all eight cells** (+10% to +16%
against +15% to +24%). On this hardware the GPU is starved harder than the extra
submissions cost, at every context tested. Do not assume the opposite without
measuring it.

**It composes with speculation rather than competing with it.** The gain appears
on both plain and speculative decode. Speculative rounds have host-side
serialization points (an accept boundary, draft-chain control) that break
cross-token pipelining, and this fills exactly that idle.

!!! warning "This is the lever most likely to be mis-screened"
    Our own first screen of this change recorded a large *regression* on
    long-context plain decode and shelved it. That screen booted a fresh server
    per arm. See rule 6 in [A/B discipline](measurement.md#ab-discipline): the
    result reversed completely under same-boot toggling.

## Composed levers need their interactions tested

The clearest cautionary tale is a **prompt-lookup / cache-hit interaction bug**.
An adaptive prompt-lookup decoder with a short warm-up window (8 cycles) worked
fine cold — ~50 cycles, 86% acceptance, ~250 tok/s on a retrieval workload — but
when the *same* request was served from an APC hit it latched **off** after two
cycles and ran the rest as plain decode at ~113 tok/s, a **0.45x** hit-vs-cold
ratio. Root cause: the acceptance rate-gate's wall-clock window opened *before*
the first verify forward, so the cache-restore and kernel warm-up were divided
into the first few tokens and read as a low rate. Arming the window *after* the
first speculative cycle fixed it (hit/cold back to **0.992**).

Neither lever was wrong alone. The bug lived entirely in their composition — and
only a test that ran prompt-lookup *on a cache hit* could find it. When you stack
serving levers, test the cross-product of their states, not each lever in
isolation.

A small composition matrix catches most state bugs before a load test:

| dimension | values to cover |
|---|---|
| prefix state | cold, repeat hit, extension hit, shared-prefix hit |
| decode lane | plain, compiled, speculative / prompt-lookup |
| cache representation | ordinary, rotating, quantized where supported, segmented/shared-suffix |
| cohort shape | uniform, ragged, preflight decline, B2-to-B1 survivor |
| outcome | full accept, early reject, trim/rollback, cancellation, next real forward |

You do not need every Cartesian-product case for every release. You do need
pairwise coverage of any two mechanisms that read or mutate the same cache, plus
one production-config multi-turn scenario. Instrument mechanism counters so a
green test cannot be a silent fallback to plain decode.

> **Prior art.** None directly.
> **How we differ.** We composed a short-warmup prompt-lookup decoder with an APC hit, where the cache-hit mis-latched the warm-up path off.
> **Our finding.** *Extends prior art* — the mis-latch ran 0.45x→0.99x after the fix; composed levers need their interactions tested, not just each alone.

### Field report: a Qwen4 composition intake

A September 2026 intake across Qwen4-class MLX implementations sharpened the
composition rule. The important result was not a new headline multiplier; it
was learning which evidence lets a mechanism enter a production stack.

| Candidate | Evidence | Decision |
|---|---|---|
| Split immutable QSA prefix with private suffix | Uniform B2 storage tests passed, but ragged `[3, 1]`, forced preflight decline, and B2-to-B1 next-forward paths initially failed | Repair accepted by an independent CPU review; GPU and memory-performance qualification still required |
| [oMLX #3553](https://github.com/jundot/omlx/pull/3553) bit-exact core | Fused GDN verify prework, grouped same-signature projections, parked-head folding, and narrow QSA were composed on a clean current tree; the author reports equal-acceptance gains from +1.2% at 16K to +14.1% at 210K | Keep as one GPU qualification arm; do not claim the external numbers locally before measuring them |
| oMLX #3553 tolerance kernels | Gathered indexed-KV attention differs around `1e-4`; NAX QSA scores around `2e-5` | Keep each explicit opt-in and test separately: a small score delta can change a discrete top-k page set and the greedy trajectory |
| [oMLX #3548](https://github.com/jundot/omlx/pull/3548) A8 prefill | The author reports 391.8→518.5 prompt tok/s (+32.3%) on a Qwen3.8-27B 4K prefill; the local eligible affine-Q4/Q5-g64 operator was only **0.78x** the incumbent at `M=2048` | Fail the local component gate; this does not refute the different whole-model operating point |
| [vLLM #52244](https://github.com/vllm-project/vllm/pull/52244) page-aligned hybrid prefix fix | A local APCv2 target `P-1` / draft `P-2` sweep passed at 2,047/2,048/2,049, so no direct CPU analogue was found | Retain a real-model APCv2 × MTP × recurrent-state boundary/churn oracle before closing the risk |
| [mlx-lm #1871](https://github.com/ml-explore/mlx-lm/issues/1871) metadata graph bound | The equivalent `mx.depends` graph-bounding mechanism was already present locally | Complementary to host mirrors that avoid membership-transition readbacks; neither substitutes for the other |

The implementation receipts at this boundary were unified repair `42250f1`
(198 authoring-suite passes plus an independent 73/73 focused CPU review),
clean Rapid-current subset `8daf2fb`, and clean oMLX composition `9e2ce29`.
The unified repair was published to the lab's private and public mirrors after
that review, but remains performance-unqualified on GPU; the other two were
build/test qualified without touching the live model.

The shared-QSA failure is the reusable lesson. A split cache is not qualified
when its storage objects merely clone, detach, and report the right offsets.
Every *consumer* must either understand the split representation or build an
exact temporary physical view. Then the test must execute the next attention
forward after each membership transition. The repair's dense fallback is a
correctness escape hatch, not a speed claim: materializing a full base per row
and layer may be expensive and needs its own GPU/memory gate.

The clean-current port to a second MLX serving stack intentionally took only
the already-independent pieces—short-forward eager dispatch and resident-cache
admission metadata—and left APCv2/shared-QSA out. That is the right kind of
partial port: transfer a mechanism only when its ownership contract exists in
the destination scheduler.

Finally, path receipts mattered as much as the score. A 100-cell realistic
sampling run scored **87/100 mechanically and 100/100 under manual semantic
review**, but all 160 harness turns failed closed to ordinary decode because
the transformed sampler was unsupported by the speculative verifier. Its
5.496 completion tok/s was also measured under heavy concurrent 16K–82K
traffic, so it is valid plain-decode quality evidence and **not** an isolated
throughput comparison or an MTP quality result. Always bind a benchmark claim
to the mechanism receipt that proves which lane actually ran.

#### Late qualification addendum: memory is part of the test gate

The subsequent GPU window established the platform boundary, but not the
requested full-model comparison. Keeping the live model protected made loading
a second roughly 100 GB model unsafe. A two-second guard stopped the benchmark
process group before any timing sample when swap headroom fell to **187.6 MiB**;
the live server stayed healthy and memory recovered. Consequently, the oMLX
#3553 bit-exact arms, unified shared-QSA performance test, and full-model
APCv2 × MTP × GDN boundary oracle are **memory-blocked under this deployment
constraint, not failed**. Do not turn a safety abort before measurement into a
performance verdict.

The already-running server did provide a useful deterministic route control.
After one warm-up, a unique `temperature=0` request completed 32 tokens in
**0.799140 s**, or **40.0431 end-to-end completion tok/s**. It restored zero
tokens from APC and accepted **20/20** self-MTP draft tokens. This proves that
the live segmented self-MTP route executed for that prompt; perfect acceptance
on a counting string is not a broad quality claim.

One bounded use of the same Flash-Next model as an inference-tuning subagent
completed 512 tokens in **12.4476 s** (**41.1324 tok/s**) with zero cached prompt
tokens and **304/413 accepted drafts (73.61%)**. Independent review scored the
answer **3.5/10** for usefulness: it was competent but truncated, mislabeled the
guarded memory abort once, and contributed no new mechanism. That result favors
tightly schemaed summary work over autonomous tuning judgment.

Two negative boundaries remain important. First, the earlier very slow control
was contaminated by cold-page rewarming after the aborted second-model load and
must not be compared with the warm request. Second, transformed-sampling MTP is
still untested: both live controls used deterministic sampling and reported no
transformed verifier. A future qualification window needs either the protected
model absent or enough reserved memory for the second model, then separate
bit-exact, tolerance-changing, APC boundary, and transformed-sampling arms.


#### September 12 continuation: gates reached, limits preserved

The later exclusive window removed the memory-blocked condition above by
quiescing the actual model LaunchAgents, not merely killing their child PIDs.
The following are measured local results on the same Qwen4-class 4-bit
checkpoint; they supersede the earlier pending status, not the historical
measurements.

| Mechanism | New local evidence | Disposition |
|---|---|---|
| Unified APCv2 × MTP × GDN | Six boundaries around 2048/4096, 18 checks per boundary including membership churn; consumer-visible target/draft/cache state exact | Correctness gate passed at tested boundaries |
| Shared immutable QSA prefix/private suffix | All five 16K lifecycle scenarios exact after preserving indexed versus dense fallback dispatch | Keep a separate serving-policy decision: one branch-to-first-commit bracket improved 1.384× but decode was 0.940× |
| oMLX RMS-compatible full GDN verify | 18 exact component cases and actual full-model rollback parity; exact core with grouped/narrow paths passed 4/4 raw-state arms | Narrow B1/S3–4 path enabled locally by operator choice; no convincing end-to-end speed attribution |
| Parked speculative-head folding | Native park/fold/reentry/rollback lifecycle reached with 38 actual tokens and fold blocks `[4,4,1]`; matched-history deferred control exact | Correctness qualified; forced scheduling and hashing are not performance evidence |
| Native indexed gathered attention | 32 same-operand probes, maximum relative error 0.0063694 against a predeclared 0.001 bound | Rejected; stays off even though this prompt's 64 tokens and log-probability probes matched |
| NAX sparse scores | 29 probes across eight context buckets; relative score error below 3.90e-7, but one selected block set changed | Rejected: output, MTP acceptance and model log-probabilities then diverged |
| Rapid safe file-backed PLE + eager dispatch | 384 resident PLE tensors removed before evaluation; sampled real-row BF16 parity; 16K/64-token eager ABBA exact | One repetitive-prompt bracket measured +17.80% with 3.30% control drift; no broad working-set or live multi-lane claim |

The exact oMLX core's later 256-token ABBA had a nominal +2.60% median decode
change and 6.37% baseline drift. Do not promote that as a throughput win.
A harness field saying the output/mechanism gate passed is not itself a
thermal qualification. New publication ladders must satisfy functional
recovery and a predeclared control-drift bound independently.

The two GDN implementations also illustrate why a mechanical code copy is
unsafe: native unified/Rapid Qwen4 use direct-L2 normalization; the deployed
oMLX superclass uses RMS arithmetic. The local port retains oMLX's RMS
contract and concrete cache-advance/rollback semantics. It does not assert
that replacing another stack's normalization with RMS is correct.

For parked heads, compare the same retained history with the same fold-block
grouping, moved earlier in time versus deferred until reentry. Turning
retention off is a different algorithm: draft history and acceptance can
change while target outputs remain correct. A fixed-depth run that never
parks says nothing about this feature.

For PLE offload, qualify the loader before throughput: bind the actual
checkpoint path, validate geometry and source identity, replace tensors
before evaluation, reproduce dequantization rounding, and cap the row cache.
A successful offload run is not a resident-versus-offloaded full-model parity
test unless both representations were actually compared. Measure diverse
working sets and eviction separately before claiming general I/O gains.

Validate the checkpoint's normalization storage convention before treating
token parity as useful inference evidence. In one Rapid Qwen4 qualification,
the runtime implemented zero-centered RMSNorm as `1 + weight`, while the
converted checkpoint stored direct gamma. Eager off/on remained token- and
logprob-identical, yet both arms produced empty or incoherent short and
multi-turn answers because both shared the same bad effective gain. A bounded
repair used the complete attention hyper-connection norm population to admit a
checkpoint-wide convention, converted direct gamma only when the FP32
`1 + residual` round trip was bit-exact, and made the MTP sidecar inherit the
backbone decision. The gated GDN norm is a separate direct-gamma class and
must not be recentered.

The paired oMLX 10-by-10 check illustrates the complementary answer-quality
gate. All 100 optimized/control pairs matched exactly and every candidate
engaged the intended MTP, RMS and grouped paths, but independent review scored
84 factual and 81 all-criteria passes in each arm. Shared overstatements remain
model-quality limitations even when a kernel introduces zero regressions. Keep
raw answers, domain-level judgments and prompt ambiguities beside the numerical
receipt; do not turn a bespoke regression corpus into a standardized benchmark.

The publication multi-turn check then exposed a serving integration boundary:
oMLX reconstructs cached recurrent state inside a sized wrapper. The RMS path
now admits only that exact wrapper around an exact qualified inner cache,
while preserving the outer object's commits, advancement and token count.
All eight off/on conversation pairs matched full logits, logprobs, hidden
and recurrent state, snapshots, output tokens and acceptance; 159 actual
candidate rollbacks matched. Six candidate follow-up turns restored 4096
tokens and still engaged RMS fusion. This is a stronger serving check than
repeating unrelated cold prompts. The separate answer review passed six of
eight turns in each arm; the same two omissions remain model-quality limits.

Host lifecycle checks also belong in an offload qualification. Fork a reader
while a parent thread holds its lock, then require the child to recover or
refuse before touching inherited synchronization. Unified preserves its
existing reopen contract; Rapid and oMLX refuse inherited use. None of these
tests establish general MLX model execution after fork. On load failure,
close only resources newly created by that load even when a traceback retains
the partial model; preserve resources belonging to successful or concurrent
loads. Drain submitted page reads before releasing descriptor ownership when
one worker fails. Real descriptors, bounded child deadlines, retained
tracebacks and injected read errors reveal bugs that output parity misses.

The sampled unified 10×10 rerun now carries request-owned engagement receipts
for all 160 requests. Same-rubric qualitative review scored 94/100 versus 93/100
on the earlier set, with 50/50 tools in both. These are bounded sampled checks,
not causal quality gains. Peer publication 10×10 checks are separate work;
the unified result must not be attributed to Rapid or oMLX.

These results are recorded in the lab's September 12 exclusive and remaining
composition experiments, with source/native hashes and compressed raw
receipts. Peer feature inventory distinguishes imported Anton Bobrik #3553
mechanisms from the lab's RMS adaptation and Rapid ports. Proposed PRs and
comments remain drafts until operator review.

---

## Megakernels

A megakernel computes an entire token — all layers plus the output projection —
in a **single persistent GPU dispatch**, instead of the ~hundreds-to-thousands
of dependent kernel launches a normal forward pass issues. On a launch-bound
decode path this deletes the per-dispatch floor. It is a real lever: on a large
MoE model, plain decode measured **~1.68–1.81x** over stock across a context
ladder (e.g. 1.74x at 32K rising to 1.77x at 128K, still 1.68x near 256K), with
identical greedy digests. A recurrent/GDN family reached similar
**1.72x/1.77x** at 1K/16K on plain decode.

> **Prior art.** Kernel fusion / persistent (megakernel) approaches in GPU inference generally.
> **How we differ.** We built a whole-token persistent megakernel as a decode lane on Apple Metal and measured its costs.
> **Our finding.** *Extends prior art* — ~1.68–1.81x plain decode, with two documented constraints: a per-dispatch cost (~29us) that makes naive op-by-op splitting bit-identical yet ~0.36x, and grid-size>residency deadlocking the grid barrier.

But it comes with hard constraints that a normal kernel does not have.

### The dispatch cost is per-binary, and it is large

A tempting middle ground is to split the token into a few persistent "glue"
segments and let ordinary matmul kernels do the rest. It was built and it is
**bit-identical** to the single megakernel — and **0.36x stock** (77 ms vs 28 ms
at 1K). The reason:

```text
# Measured per-dispatch cost of the SAME persistent binary:
#   one megakernel dispatch  ~29 us   (many bindings, persistent grid)
#   one ordinary MLX kernel   ~2 us
#
# Splitting a token into N dispatches of the persistent binary pays ~29 us x N.
# At 242 dispatches/token:  242 * 29 us = ~7 ms of pure launch overhead,
# a >31.7 ms lower bound  vs  ~28 ms for the whole stock token.
```

The launch floor is a property of the *binary you dispatch*, not a universal
"~2 us per kernel." Measure the floor of the kernel you will actually launch
before designing around it. (A host-side variant that walks phases in Python is
worse still: the graph *build* alone cost more than a whole stock token, and
async evaluation cannot hide it — the walk scales with phases, not dispatches.)

### Occupancy is a correctness parameter

For a persistent kernel the grid must be fully resident, because its threadgroups
spin on a device-scope barrier. If the launch grid exceeds resident capacity
(e.g. a threads-per-core × groups product tuned for one geometry, reused at a
wider one), the grid barrier **deadlocks** and wedges the GPU — recoverable only
by reboot. The same applies to killing a run: never `SIGKILL` a process with a
persistent dispatch in flight; its threadgroups keep spinning on the device with
no owning process. Let the launch finish or use the kernel's abort flag.

!!! warning "Treat every persistent launch as a device-acknowledged transaction"
    A host return from the launch is *not* evidence the grid completed. The
    device must publish an abort count and final phase, and the host must
    synchronize and validate both before exposing any output or committing state.
    State that cannot alias its input — recurrent and convolution state, and any
    other non-aliasable side state — is written to a separate buffer and swapped
    only on commit; a second launch
    is illegal until the first is committed or rolled back.

### Fidelity: near-fp32, not bit-exact

Folding a whole token into one kernel changes reduction order, so a megakernel is
typically **not token-exact** with stock — it lands in a "closer-to-fp32 than
stock" tolerance class. That has to be gated on *behavior over natural text*, not
on a distance: one build passed eight boundary tables, a bit-identity proof of
its attention phase, a 46-test suite and a 128-token greedy gate **while it could
not attend its own three most recent tokens**. Two documents of teacher-forced
perplexity found the defect in ~100 seconds (+0.033 nats). A component's
bit-identity proof does not cover the component's *caller*, and a random-token
fixture suppresses exactly the local-context term a masking defect removes. Gate
on mean NLL over real text with a document-level confidence interval; keep the
distance metrics for attribution, not acceptance.

### Where the megakernel actually pays back

Fusing dispatches only helps if dispatch overhead was on the critical path.
On a recurrent/GDN family the fused-forward passes are **GPU-bound** — the
host-exposed fraction is only ~4.1–4.5% at width 1 and ~0% at the width-3 verify
slab. So the megakernel's dispatch-fusion optimizes a cost that is already
nearly gone *inside* the forward, which is why its curve flattens there.

The large multi-turn win from a megakernel came from a different place:
**publishing its state back into the prefix cache**. With APC publish-back, a
multi-turn conversation dropped from **~33.5s to ~14.0s** (~2.4x) with decode
speed unchanged and the published state CPU-byte-lossless. The win is
orchestration *between* forwards (skipped prefill on the next turn), not anything
inside the GPU-bound forward.

**How to apply.** Prove launch-bound behavior with an encoder-interval trace,
prototype one complete token path, and gate device completion before reading
output. Measure the actual persistent binary's empty or minimal-work dispatch
floor. Qualify both arithmetic fidelity and full-model behavior on natural
text, then test cancellation and rollback as first-class paths.

**When not to.** Do not start with a megakernel when encoder gaps are already a
small share of the step, when the model changes frequently, or when the team
cannot maintain a device-side scheduler and its correctness suite. It is a
specialized runtime lane, not a general flag.

> **Prior art.** None directly.
> **How we differ.** We publish prefix state back from the megakernel lane into the prefix cache.
> **Our finding.** *Extends prior art* — cut a measured multi-turn time ~33.5s→14.0s with decode unchanged; the win is orchestration between forwards.

### Megakernel and self-speculation can be mutually exclusive

On a recurrent/GDN family the megakernel and self-MTP speculation **do not
compose** — they are substitutes, not complements. Two independent reasons:

1. The MTP draft head reads a *pre-mixer* hidden tensor that the megakernel's
   fused schedule computes internally and does not surface in a form the draft
   path can consume. (This tap is cheap to add, but adding it does not change
   the verdict.)
2. Priced head to head, the wide verify slab inside the megakernel runs at
   **~0.71–0.74x** of the stock fused verify, and the combined draft+verify
   round comes out to **~0.92–1.03x** of the stock speculative round — a wash.
   Speculation already earns its keep by making a wide verify nearly free in a
   launch-bound engine; the megakernel has *already spent* that same launch
   budget, so there is nothing left to compose.

The general principle: **compose only levers that share state and attack
different costs.** Two levers that both delete the launch floor overlap rather
than stack.

> **Prior art.** None.
> **How we differ.** We tried composing the megakernel with self-MTP speculation on a recurrent/GDN family.
> **Our finding.** *Extends prior art* — the two cannot compose (the MTP head drafts from a pre-mixer tensor the megakernel doesn't emit); compose only levers that share state.

---

## Batching vs single-user changes the calculus

The right serving lever depends heavily on whether you are optimizing one
interactive stream or many concurrent ones.

**Single user.** Decode is latency-bound and the queue is width 1. Here
self-speculation (a self-MTP head verifying its own drafts) is usually the
winner — on the models above it beat plain decode by ~1.9–2.3x across contexts
and beat plain megakernel decode at every context measured. Compiled replay adds
a further small win at short context on top of plain decode. The megakernel's own
prize is **plain** single-user decode and the long-context regime where the
baseline collapses — precisely where speculation's acceptance falls off.

| Lever | Single user | Many concurrent users |
|---|---|---|
| Self-MTP speculation | Strong (fills width for one user) | Weak — adds width the batch already has |
| Plain batching | N/A | Strong (fills width across users) |
| Megakernel (plain) | Good, best at long context | Real width-1 steps exist here; niche |
| Compiled replay (M=1) | Good at short context | Superseded by batch scheduling |

**Many users.** A full batch already supplies the arithmetic width that
speculation manufactures for a single stream, so self-MTP added on top of an
already-wide batch *loses*. The lever that supplies width across users is
ordinary continuous batching. A cost model calibrated at one batch width does not
transfer: an admission controller with only a memory ceiling admitted 2.5x past
the throughput knee on a small model, because free RAM scales with model size
while GPU compute saturation is fixed — cap by **both**.

A practical admission controller therefore needs two ceilings:

```text
admit only if
    projected_resident_bytes <= memory_budget
and projected_batch_width     <= measured_throughput_knee
```

Derive the width ceiling from a context-matched sweep, not from free memory.
Track queue delay separately from model time: batching can improve aggregate
throughput while making one interactive user's latency worse.

The single-user floor, once speculation and an async-dispatch overlap are in
place, is the GPU roofline itself: on the recurrent family the remaining
per-round host idle was ~3ms (~8%), mostly already hidden, and idle CPU there is
the *signature* of a GPU-bound roofline, not spare capacity to reclaim. Know
which regime you are in before you reach for a lever built for the other.

!!! note "Transfer to llama.cpp and vLLM-on-Metal"
    APC keys, cache ownership, publish-back, and composition gates are portable
    service concerns. vLLM's paged blocks make sharing and eviction explicit;
    `llama.cpp` may expose a different session/cache interface, but a warm hit
    still needs compatibility metadata and token-level validation. Compiled
    replay maps to graph capture where the backend supports it. Megakernels map
    only when the Metal backend can own a whole persistent schedule; ordinary
    operator fusion is not automatically equivalent.

---

## Sources

Public, upstream concepts this chapter builds on:

- **`mx.compile`** — MLX's graph tracing/replay, the mechanism behind compiled
  decode replay.
- **MLX ring / rotating KV caches** — the shape-stable cache class compiled
  replay requires; see the MLX-LM cache implementations and `--max-kv-size`.
- **Prefix caching** — general automatic-prefix-cache / content-addressed KV
  reuse as described in the vLLM and SGLang literature, and the KV-leakage
  analyses that exempt content-addressed caches.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving
  with PagedAttention* — SOSP 2023, arXiv:2309.06180.
- vLLM documentation — automatic prefix caching and paged KV management:
  <https://docs.vllm.ai/>
- **Persistent / megakernels** — the "one kernel per token" pattern from the
  persistent-kernel literature (e.g. Mirage-style persistent kernels and
  monokernel decode work on other accelerators), here retargeted to Metal via
  custom-kernel compilation.

All quantitative figures are from lab measurements on Apple M5 Max hardware and
should be treated as directional for that hardware class, not as portable
constants. Re-measure on your own device and model before adopting any lever.
