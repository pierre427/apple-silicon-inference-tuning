# Heterogeneous GPU + Apple Neural Engine inference

The Apple Neural Engine (ANE) is a genuinely separate execution lane from the
GPU. That makes an appealing proposition: keep an already-fast Metal target
model on the GPU, move speculative or auxiliary work to ANE, and overlap the
two. We tested that proposition against a persistent Qwen4-class Flash-Next
megakernel whose GPU path had little spare compute or bandwidth.

The result is useful but narrower than the slogan:

- independent ANE and GPU work overlapped at **94% efficiency**;
- same-request ANE self-MTP still lost, because the target verifier could not
  use a proposal before it existed and duplicated target work consumed the
  apparent headroom;
- a component-level cross-request pipeline gained **1.053x at eight**, **1.077x
  at sixteen**, and **1.109x at sixty-four** ready drafts, but did not improve
  batch-one latency; and
- the gain disappeared in the live continuous-batching path because splitting
  or serializing work destroyed the incumbent GPU batch's sharing.

The practical lesson is that ANE is not “free compute.” It is a
**ready-work and handoff resource**. Use it only for a coarse, independently
ready stage whose result is compact and whose consumer can proceed without
breaking a stronger GPU batch or fused kernel.

!!! warning "Do not confuse ANE with the M5 GPU Neural Accelerators"
    This chapter concerns the separate **Apple Neural Engine**, reached through
    Core ML or a direct e5rt execution path. The M5 **Neural Accelerators (NA)**
    are matrix units inside the GPU cores. NA/NAX work competes with a Metal
    megakernel for GPU resources; ANE work can overlap it. See [The hardware
    model](hardware.md#the-m5-neural-accelerators-a-prefill-lever-mostly).

---

## The question we actually tested

Native self-MTP can approach a 2x gain when it exploits compute or bandwidth
left idle by an inefficient token-at-a-time target. A megakernel removes much
of that inefficiency by coalescing operations and sharing weight traffic across
rows. The MTP work then has no free GPU slot to occupy.

We therefore asked a stricter question:

> Can ANE supply enough independent proposal work beside an already-saturated
> GPU target to reach 1.5x over the megakernel, while the target remains the
> sole authority for committed tokens and state?

The model under test was a 4-bit Qwen3.8-Flash-Next checkpoint exposed through
the `qwen4_exp` architecture: 48 recurrent/sparse-MoE layers, a trained MTP
sidecar, and a persistent Metal verifier. Measurements were taken on an M5 Max
with a 40-core GPU and 128 GB unified memory. These numbers describe that
operating point, not every model or Apple SoC.

The correctness contract never changed:

1. ANE may propose approximate tokens or hidden states.
2. The GPU target verifies every proposal.
3. Only the accepted prefix mutates durable KV, recurrent, convolution, PLE,
   or MTP state.
4. Every asynchronous result carries a request generation and is discarded if
   it becomes stale, late, cancelled, or mismatched.

This contract lets aggressive proposer experiments fail harmlessly. A proposer
mismatch is allowed; a committed target or cache mismatch is not.

## Physical concurrency is necessary, not sufficient

The first experiment paired the real width-one, 48-layer Metal megakernel with
an ANE-resident surrogate shaped like the trained MTP fusion prefix. At 16K the
two isolated tasks were almost equal in duration:

| Auxiliary lane | Target isolated | Auxiliary isolated | Joint | Serial / joint | Target slowdown |
|---|---:|---:|---:|---:|---:|
| Apple Neural Engine | 15.740 ms | 15.336 ms | 16.667 ms | **1.865x** | 5.9% |
| GPU NA/NAX-shaped control | 15.594 ms | 15.590 ms | 27.710 ms | **1.125x** | 10.6% |

All non-constant operations in the ANE graph preferred ANE in the compiled
compute plan. The NA/NAX control's own latency grew 1.78x under concurrency,
which is what shared GPU contention looks like. The ANE workload stayed nearly
flat.

That proves the hardware can run both engines together. It does **not** prove a
model speedup, because one request's self-MTP round is dependency-ordered:

```text
MTP draft 1 -> MTP draft 2 -> target width-3 verify -> accept and commit
```

The verifier cannot start until the proposed token IDs exist. Merely moving the
draft boxes to ANE leaves the GPU idle unless the GPU performs other work. If
the GPU runs a width-one target step while waiting, that work duplicates the
first row of the later verifier. Physical overlap can therefore look excellent
while accepted-token throughput gets worse.

### Always draw the dependency graph first

Before exporting anything to ANE, label every node with:

- the data it consumes and produces;
- the earliest time it becomes ready;
- which engine can run it;
- its isolated and concurrent latency;
- bytes and dtype crossing the device boundary; and
- whether a late result can be discarded without restoring state.

If the proposed ANE node is not ready while useful GPU work exists, there is no
pipeline—only a slower serial device transfer.

## Real MTP: where the same-request route stopped

The real trained MTP fusion prefix executed on ANE, but it was too small a cut:
the MLX GPU path took 0.200 ms and the complete ANE round trip took 0.414 ms.
Moving a wider attention/projection prefix also worked and kept its real
weights resident, but same-sequence composition remained negative.

| Schedule | Result versus megakernel | Acceptance | Target-only perfect-acceptance ceiling |
|---|---:|---:|---:|
| dependency-correct `k=2`, predictable filler | **0.809x** | 100% | 1.022x |
| dependency-correct `k=2`, natural text | **0.721x** | 74.5% | 0.933x |
| `k=1` control | **0.854x** | 100% | 1.021x |

The important column is the last one. It removes *all* ANE, drafting, handoff,
catch-up, and host overhead. If even that counterfactual cannot reach the goal,
no scheduler tuning can rescue the design.

Use the same pre-gate for any heterogeneous speculative path. If a plain target
step costs `T1`, a width-`w` authoritative verify costs `Tw`, and perfect
acceptance commits at most `w` tokens, then the absolute target-side ceiling is:

```text
perfect_acceptance_ceiling = w * T1 / Tw
```

Apply a handoff and scheduler reserve after this calculation. Stop before
training or server integration when the result is already below the target.

## Improving the verifier helped, but did not change the verdict

The target verifier was already a physical multi-row kernel: packed weight
blocks were loaded outside the row loop, attention reused K/V, recurrent state
stayed resident, and MoE read the union of selected experts once. Two further
cuts were worthwhile:

| Verifier change | Width 2 | Width 3 | Disposition |
|---|---:|---:|---|
| compile separate exact-width bodies | **1.128–1.130x** | **1.065–1.084x** | retain, default off |
| fuse projection families across scratch planes | **1.026–1.031x** | **1.089–1.092x** | retain, default off |

The optimized transaction raised the predictable-filler target-only ceiling to
**1.172x**. It was real progress, but still far below 1.5x. Removing more phase
boundaries without sharing a large weight traversal was flat or slightly
negative.

A compact routing layout then made verifier widths 5–8 physically legal under
the threadgroup-memory cap. That exposed a register/occupancy cliff:

| Verify width | Best target time | Perfect-acceptance ceiling |
|---:|---:|---:|
| 5 | 60.304 ms | **1.233x** |
| 6 | 142.360 ms | **0.627x** |
| 8 | 327.547 ms | **0.363x** |

Making a kernel launch is not the same as making it useful. Width 8 fit, ran,
and was dramatically worse. The monolithic wider-verifier route is closed until
a different allocation domain or multi-kernel design beats the measured cost
budgets.

!!! warning "Correction: \"peak register usage sets occupancy\" is obsolete on M3 and later"
    We described this cliff, and two similar results, with the model that a
    single dispatch gets one register allocation sized by its widest phase, so
    every other phase pays that maximum. **On Apple family 9 GPUs and later that
    model is no longer correct.** Apple states that before family 9 the
    allocation "would be equal to the maximum register usage at any point in the
    program", but that with the family 9 dynamic shader core memory feature "the
    maximum register usage no longer dictates how many SIMDgroups can be run",
    because register memory is allocated and released over the lifetime of the
    shader according to what each part actually uses. Family 9 is M3; M5 is
    family 10.

    Consistent with that, pipeline reflection on our own kernels reports
    `maxTotalThreadsPerThreadgroup` of 1024 for bodies of quite different width.
    So the cliff above is real and measured, but the *mechanism* we attributed it
    to probably is not. The likelier mechanism is **live ranges**: a persistent
    kernel holds per-query state live across phase boundaries, which is exactly
    what defeats lifetime-based deallocation, and the pressure then shows up as
    graded L1 residency with hardware occupancy throttling rather than a hard
    allocation cliff.

    This changes what to try. Shrinking a phase's peak usage is the old fix.
    The new one is **shortening live ranges** so the hardware can release
    between phases, reloading or recomputing at phase entry instead of carrying
    values through. On family 9 and later the instruments are the Occupancy
    Manager Target and L1 Residency counters, not a register count — and there is
    no Metal API that reports registers directly. The one officially sanctioned
    observable is `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`.

    We have not yet re-measured the widths above against that model. Read the
    table as a measured cost curve, not as an explanation.

!!! tip "Where the speculative round's decode budget actually goes"
    A timed-vs-exposed decode-wall attribution (greedy, `num_draft=2`, fused
    GDN verify, **persistent megakernel off** — this is the ordinary eager
    forward, not the megakernel lane) pins the blame on the **verify step**,
    not on host orchestration. At 4K and 16K context the per-round wall is 38.0
    and 39.9 ms, and the span bracketing the verify boundary is 28.8 and 29.9
    ms, about **73%** of the wall.

    Be careful reading that 28.8 ms: splitting the span shows it is almost
    entirely the **GPU drain of the verify forward**, waited on at the first
    synchronization after the drafts were dispatched. The host-side accept work
    itself — comparing tokens, scanning the accepted prefix, trimming caches —
    is about **0.2 ms**. Accept is therefore not a fusion candidate, and a
    bucket named for a host step can be almost entirely device time. Recoverable
    host orchestration across the whole round is around 3 ms, the largest piece
    being draft-chain control.

    So **the lever is the verifier cost itself**, which is what the width-3
    ceiling work above is trying to shrink. Any accelerator-side proposer that
    hopes to help must amortize or offload that verify step; moving the draft
    work alone leaves it on the critical path. The ~3 ms of real host
    orchestration is better attacked by overlapping it — see
    [per-layer submission pipelining](serving-techniques.md#submit-each-layer-as-you-build-it).

## Compression and handoff details were first-order

The initial ANE package duplicated about 111 MB of FP16 weights. Core ML
palettization produced a useful result: LUT8 halved package storage and made the
ANE stage **1.54x faster** while remaining ANE-native. Per-block 4-bit
quantization did not: the compiled plan assigned all 46 operations to CPU.

The result is a warning against reasoning from file format alone. A compressed
package is useful only when the compiled execution plan keeps the compressed
operators on the intended device and warmed latency improves.

The consumer dtype mattered just as much. In the cross-request prototype, an
FP16 ANE packet changed the downstream GPU dispatch regime and made the path
**0.914x** of the fused GPU control. Casting the returned packet to the GPU's
native BF16 before the remainder recovered **0.984–0.986x**; moving submission
onto a persistent host thread reached parity at one draft.

For every handoff, record:

```text
producer dtype and shape
bytes returned per request and per step
conversion location and latency
consumer kernel selected after conversion
submission and synchronization time
```

A numerically harmless dtype conversion can be a systems-level performance
change if it selects a different consumer kernel.

## Cross-request scheduling: the one positive throughput result

Independent requests create naturally ready work. While the GPU performs
target work for request A, ANE can prepare an MTP stage for request B without
duplicating A's target computation:

```text
GPU:  target A ---- target C ---- consume B proposal ---- verify B
ANE:       prepare B ---- prepare D ---------------------------->
```

With direct resident e5rt execution, LUT8 weights, BF16 handoff, and threaded
submission, the component pipeline measured:

| Independent drafts per target interval | Aggregate speedup | Request-A target latency |
|---:|---:|---:|
| 1 | **1.001x** | +1.0% |
| 4 | **1.035x** | +1.6% |
| 8 | **1.053x** | +2.1% |
| 16 | **1.077x** | +3.9% |
| 32 | **1.099x** | +7.8% |
| 64 | **1.109x** | +7.8% |

This is not faster local batch-one inference. It is a serving-throughput curve,
and the useful resource is queue depth. Around 8–16 ready drafts gave the best
latency/throughput tradeoff; deeper queues mostly bought another few percent at
a larger latency cost.

### Why the live batch still lost

The live continuous-batching incumbent already shares MTP and target-verifier
weights across all active lanes. The ANE cut returned a 26,756-element BF16
activation packet—**53,512 bytes per lane per step**—and required serial B=1 ANE
jobs before one batched GPU remainder.

The exact live pipeline measured **0.979x at four lanes, 0.974x at eight, and
0.916x at sixteen**. A two-cohort schedule was worse because splitting the GPU
batch discarded more shared work than overlap hid. The implementation proved
transaction, cancellation, and stale-result safety, but the performance route
is closed.

This is the central composition rule:

!!! danger "Do not fracture the winning batch to manufacture ready work"
    Compare against the strongest incumbent schedule, not against a serial
    decomposition of that schedule. Heterogeneous overlap must repay any lost
    weight sharing, megakernel fusion, and batch width before it counts as a
    gain.

## Cache branching can hide setup work, not target verification

A follow-up campaign on the same Qwen4-class serving stack examined the
roughly 0.6-second preparation interval for two-row self-MTP. Most of that
interval was target/MTP catch-up plus construction of the physical B=2 cache,
not Python object cloning: cloning the target cache cost about 40--42 ms, while
generator preparation cost 535--597 ms.

The first pass tested ways to avoid, defer, or share that cache work. The
thermally interleaved 64-token A/B results were mostly negative:

| Preparation candidate | Prep ratio | Decode tok/s ratio | Total-wall ratio | Decision |
|---|---:|---:|---:|---|
| consume an already-prepared GDN live tip | **0.983x** | **1.026x** | **1.010x** | hold default-off; one bracket is not promotion evidence |
| extend the QSA tail horizon | **0.899x** | **1.027x** | **0.984x** | reject |
| physically materialize immutable B=2 fan-out | **0.940x** | **1.007x** | **0.982x** | reject |
| immutable fan-out plus QSA horizon | **0.943x** | **1.044x** | **1.006x** | reject as a prep lever; apparent total win is too small for one bracket |
| consume live tip plus QSA horizon | **0.927x** | **1.005x** | **0.977x** | reject |

Ratios above 1 are better. The live-tip path is the only candidate worth
keeping behind a switch: it improved decode by about 2.6%, but preparation got
about 1.7% slower and total wall improved only about 1.0%. It needs repeated
thermal brackets and composition with the production levers before it should
become a default.

We also corrected a tempting but invalid coalescing experiment. Coalescing the
target's canonical `(T-1)+1` tail into one `T`-token call changed the recurrent
and attention kernel geometry, so it was not a valid performance comparison.
Restricting coalescing to the MTP teacher-forcing tail restored exactness, but
still did not pay:

| Corrected MTP-only arm | Prep ratio | Decode tok/s ratio | Total-wall ratio |
|---|---:|---:|---:|
| coalesced tail alone | 0.979x | 1.010x | 0.999x |
| live-tip consume plus coalesced tail | 0.952x | 1.012x | 0.990x |

The experimental runtime hooks were removed. This is a useful general rule for
hybrid recurrent models: two call schedules that consume the same token count
are not necessarily equivalent. Preserve the target's established state-update
geometry unless continuation-state equivalence has been proved.

### A cheap branch descriptor is not a cheap cache consumer

At a 16,376-token prefix, 26 BF16 state tensors occupied 536,477,760 bytes for
B=1. Creating a descriptor or zero-stride B=2 view was nearly free, but the
first operation that needed private or contiguous storage paid the bill:

| Operation | Median | Active allocation |
|---|---:|---:|
| stop-gradient B=1 descriptor | 0.066 ms | 0 |
| zero-stride B=2 broadcast | 0.059 ms | 0 |
| physical B=1 `mx.array` copy | 9.039 ms | 536,739,840 bytes |
| physical B=2 concatenate | 16.658 ms | 1,073,053,696 bytes |
| first append through B=1 descriptor | 8.797 ms | 536,739,840 bytes |
| first append through B=2 broadcast | 16.367 ms | 1,073,479,680 bytes |
| private eight-token B=2 delta only | 0.276 ms | 262,080 bytes |
| ready B=2 capsule build | 16.191 ms | 1,090,093,056 bytes |
| first patch into an already-ready capsule | 0.298 ms | 0 |

The segment-aware representation remains attractive--a shared immutable
prefix plus a tiny private delta--but only if downstream kernels consume it
directly. Stock fused SDPA did not. On a 16K attention probe, physical B=2 KV
ran in 0.464 ms, while the exact zero-stride broadcast took 0.801 ms and added
33,828,864 transient bytes. The shared view achieved only **0.579x** of the
physical path, so broadcast shared-prefix SDPA is rejected for this consumer.

### CPU is an acceptable fallback for cache copies; ANE overlaps better

Direct e5rt exposes no supported "force this ANE graph onto CPU" switch, so a
resident NumPy copy was used as the deliberate CPU control. It was not
crippling in isolation, but its overlap with a concurrent Metal concatenate
degraded as the cache grew:

| B=1 input / B=2 output | CPU only | ANE only | GPU only | CPU + GPU | ANE + GPU |
|---|---:|---:|---:|---:|---:|
| 4 / 8 MiB | 0.108 ms | 0.184 ms | 0.210 ms | 0.184 ms | 0.189 ms |
| 16 / 32 MiB | 0.463 ms | 0.499 ms | 0.271 ms | 0.560 ms | 0.514 ms |
| 32 / 64 MiB | 0.944 ms | 0.942 ms | 0.427 ms | 1.122 ms | 0.973 ms |

At the largest case, ANE retained 92.9% overlap efficiency versus 58.5% for
the CPU control; the CPU concurrent wall was about 15.3% slower. CPU is still
a sensible opportunistic lane for cache packing, metadata, hashing, eviction,
tokenization, scheduler bookkeeping, lookup/decompression, serialization, and
precomputed masks--provided none of those paths introduces a GPU readback or a
new synchronization point.

### The ANE cache capsule transport boundary is viable

The positive systems result is a bit-preserving cache transport capsule. An
e5rt B=1-to-B=2 concatenate was consumed by MLX through
`mx.from_dlpack(..., copy=False)` and then by a real Metal equality kernel while
the e5rt program and buffer owner remained pinned. FP16 input staging, adoption,
alias coherence, and the Metal consumer were exact at 4, 16, and 32 MiB:

| B=1 input | Resident ANE | MLX input -> e5rt + execute | MLX no-copy adoption | Metal consumer |
|---:|---:|---:|---:|---:|
| 4 MiB | 0.220 ms | 0.265 ms | 0.027 ms | 0.347 ms |
| 16 MiB | 0.511 ms | 0.764 ms | 0.027 ms | 0.741 ms |
| 32 MiB | 0.946 ms | 1.508 ms | 0.034 ms | 0.922 ms |

Real Qwen cache state is BF16, so a second probe carried BF16 payloads as
opaque `uint16` bits through the same FP16-shaped e5rt graph. It was bit-exact
at all three sizes, retained alias coherence, used `copy=False`, and kept the
owner alive through Metal consumption. No-copy adoption was 0.027, 0.028, and
0.033 ms; MLX-input staging plus execution was 0.266, 0.783, and 1.486 ms.
This qualifies **transport and materialization**, not native BF16 ANE arithmetic
and not use-after-owner-release. A production lease must pin the e5rt program,
output view, and adopted MLX alias until the GPU consumer has synchronized.

The implication for speculative validation is deliberately narrow:

!!! note "This does not buy a second validator"
    Cache transport can reduce setup bubbles and keep another speculative row
    admission-ready for a wider GPU batch. It does not make target-model
    verification run on CPU or ANE: authoritative target verification still
    traverses the GPU model. The measured savings are sub-millisecond cache
    movement beside a tens-of-milliseconds target verify, so describe this as
    scheduler readiness, not another validation lane.

Exact source artifacts for this campaign:

```text
mlx-uag/results/qwen4-gdn-prep-matrix-64tok-20260910.json
mlx-uag/results/qwen4-gdn-prep-mtp-coalesced-tail-64tok-20260910.json
mlx-uag/results/cache-branch-state-microbench-20260910.json
mlx-uag/results/cache-shared-prefix-attention-20260910.json
mlx-uag/results/ane-gpu-cache-overlap-cpu-control-20260910.json
mlx-uag/results/ane-cache-capsule-ingress-lifetime-20260910.json
mlx-uag/results/ane-cache-capsule-bf16-bits-20260910.json
```

## Candidate ledger

The campaign deliberately screened adjacent ideas rather than repeatedly
tuning one failed schedule.

| Candidate | Measured result | Decision |
|---|---|---|
| ANE/GPU physical overlap | 1.865x serial/joint; 94% efficiency | mechanism proven |
| exact-width and projection-family verifier work | up to 1.13x body-level | retain experimental |
| LUT8 ANE weights | ~half package; 1.54x ANE-stage speed | retain experimental |
| cross-request MTP component pipeline | 1.053–1.109x at 8–64 ready drafts | retain default-off prototype |
| live continuous-batch ANE MTP | 0.979x / 0.974x / 0.916x at B=4/8/16 | close current wide-packet cut |
| compact 98,304-row ANE vocabulary head | 0.941x sync; 0.812x split verify | retain substrate, stop scheduling tweaks |
| one-shot two-token ANE sidecar | 3.905 ms B=1; verifier ceiling 1.226x | topology passes; do not train yet |
| complete GDN block on ANE | 1.041 ms ANE vs 0.310 ms Q4 GPU; poor state fidelity | stop at one layer |
| PLE device-half offload | at most 1.004x end to end | do not build |
| PLD lookup offload | already tiny CPU retrieval work | keep on CPU |
| projection-only prefill offload | Q4 GPU 1.9–5.0x faster at wide rows | no batch-one build; QoS only |
| resident expert-major repack | layout and union reads already present | no second resident copy |
| macro/phrase proposal sidecar | held-out prose/code fail; verifier ceiling too low | close general route |
| whole-cache KV8 | halves 64K incremental cache cost but fails quality gate | capacity fact, no admission |
| selective-layer KV8 | passes 8K/16K, fails 64K | do not promote |

Negative results are scoped. For example, ANE can still be a good proposer or
ranker even though FP16 GDN state is a poor offload boundary. KV8 may still be a
capacity emergency policy under a relaxed quality contract even though it is
not a lossless default. State the contract when carrying a result forward.

## What a promising ANE stage looks like

A candidate is worth building when it satisfies most of these conditions:

- **coarse:** enough work per dispatch to amortize Core ML/e5rt submission;
- **independently ready:** it can run while useful GPU work is already ready;
- **compact output:** token IDs, top-k candidates, or small metadata rather than
  a full hidden activation per lane;
- **resident:** weights and preferably draft state remain on ANE across steps;
- **transactional:** late work can be discarded and rejection can teacher-force
  or restore the proposer without corrupting target state;
- **consumer-native:** the returned dtype and shape preserve the winning GPU
  kernel; and
- **better than the incumbent batch:** it does not obtain overlap by breaking a
  wider fused schedule into smaller, slower pieces.

The most credible reopened design is therefore not “offload one more operator.”
It is an ANE-resident multi-token proposal chain that returns only token-sized
candidates, while the GPU performs one full-width authoritative verification.
Persistent proposal state needs generation-stamped commit/rollback, and a
rejection must repair that state without round-tripping a full activation
history.

The current two-token package proves that a shared LUT8 vocabulary head plus a
low-rank second head can stay on ANE and return only 192 bytes at B=1. Training
remains intentionally held because the target verifier caps perfect acceptance
at **1.226x**. Reopen it only if width-3 verification falls from 36.393 ms to
**29.748 ms or less**—an 18.3% reduction before reserve—or a new wider verifier
demonstrates a better measured ceiling.

!!! warning "A token-exact proposer can still corrupt downstream state"
    Before that reopen, one more gate is now known. A width-3 commit oracle that
    compared the unfused commit against the sequential reference matched the
    top-1 token across the whole run — yet diverged from the first non-exact
    phase onward: the attention reduction carried a max-abs of ~1.2, and the
    recurrent state had ~25.4M of 25.7M elements mismatched. **The next token
    was right; the next-step input was not.** So a proposer that passes the
    token-acceptance ceiling is *not* automatically safe to hand a multi-kernel
    state to — the handoff consumes the (divergent) state, not the token. Treat
    "verifier ceiling satisfied" and "state faithful" as two separate sign-offs;
    fix attention-reduction faithfulness before any phase-local ANE allocation.

## Follow-through: transport is not product overlap

The next production-shaped gate corrected an over-broad reading of the cache
capsule microbench.  At a 16K QSA-plane shape (H2, D128, B2; 16 MiB input and
32 MiB capsule), e5rt did overlap useful work: 26.247 ms overlapped versus
37.455 ms serial.  But the ordinary GPU consumer finished in **11.628 ms**.
The e5rt product path was therefore only **0.443x** the incumbent, despite
hiding about 11.2 ms of its own serial time.  A matched asynchronous CPU arm
was 16.811 ms and a GPU-built capsule was 14.607 ms.

This distinction is essential on unified memory:

- **Alias and ownership proof:** an accelerator result can be adopted safely;
- **overlap proof:** independent engines can execute concurrently; and
- **product proof:** the complete producer, readiness wait and consumer beat
  the existing path.

Passing the first two does not imply the third.  Keep the generation-stamped,
exactly-once release/abort machinery—it prevents real use-after-release and
leak bugs—but leave production cache construction on the GPU for this shape.
Also do not enter an MLX GPU stream from an arbitrary host worker: stage raw
bits on the owning thread, perform genuinely host-only work in the worker, and
adopt the result back into MLX on the owning thread.

## Follow-through: shared-prefix consumers must share the expensive read

An immutable-base/private-delta cache can remove B-times-prefix construction
and still lose at decode.  In accepted, swap-flat, thermally interleaved
full-model tests, a Qwen4 private-delta consumer reached only **0.8907x** the
physical-B2 decode rate at 16K/256 output tokens and **0.9100x** at 32K/256.
It avoided hundreds of gigabytes of cumulative duplicate-base formation over
thousands of layer calls, but each row still reread the same immutable pages.

Real B2 topology receipts showed the lever the representation alone had
missed: all 65 simultaneous query cohorts selected exactly the same base-page
set for both rows (Jaccard 1.0), giving a topology-only ceiling of 50% fewer
base reads.  The resulting exact-set fold:

1. proves ordered selected-page equality with a device-resident predicate;
2. reads each shared base page once while maintaining row-private softmax
   accumulators;
3. keeps suffix work private; and
4. executes the exact row-local path inside the kernel on every proof miss.

There is no `.item()` or host decision in the hot path.  At 16K B2, a
101-repetition direct Metal gate was raw-bit exact for M1--M4.  The fold was
**1.078x** faster than the private-delta path and **1.072x** faster than
physical B2 at M3; at M1 it was 1.040x faster than private delta but 0.987x
physical.  The durable rule is to share the controlling bandwidth work, not
merely the object that names it, and to gate by query shape because M1 and M3
can have different crossovers.

The runtime result is smaller but reproducible. Three accepted 16K/256-token
brackets improved the already gated private-delta lane by 2.38%, 0.49% and
0.42% decode, with a **0.49% median** and zero proof/dispatch fallbacks across
8,814 folded calls. That justifies selecting the fold inside admitted B2
private-delta traffic; it does not promote private delta over physical B2,
which remained about 9--11% faster in separate accepted 16K/32K tests.

Host work has a different useful boundary. A process-local cache of exact
request rendering and tokenization reduced a 5,374-token real-tokenizer
component from 4.832 ms to 0.091 ms (53.1x). A live server then proved repeat
hits, response-level cached prompt tokens, soft-reload invalidation and
post-reload repopulation. Keep this opt-in for known-pure tokenizers: it is a
high-churn request-preparation lever, not a decode throughput claim.

## Follow-through: screen confidence routers by perfect-oracle budget

Confidence is useful for allocating verification width only when the width
increment is large enough to pay for extracting and routing on the signal.  A
current K=2 trace with 196 proposals, 157 accepted tokens and 98 cycles bounds
even a perfect oracle to **0.535--0.816%** of verifier projection time—only
4.44--6.77 microseconds of total decision budget per cycle.  A synthetic
5-microsecond router made the best arm 0.9940x.

Do not add a synchronized confidence trace just to discover this afterward.
Compute the perfect-oracle ceiling first.  Reopen the router when telemetry is
already available without synchronization, draft depth reaches at least four,
or measured full-model verification width is materially steeper than its
projection-only model.

## Qualification checklist

Use this order; each step has an early stop.

1. **Ceiling:** measure width-one and intended verification widths on the real
   target. Stop if perfect acceptance cannot clear the goal with reserve.
2. **Placement:** inspect every non-constant operation in the compiled compute
   plan. CPU/GPU fallback is a failed ANE arm.
3. **Isolated component:** measure warm latency, package size, resident memory,
   transfer bytes, and output/state fidelity.
4. **Physical overlap:** pair the component with the real target and report both
   engines' slowdown, start skew, serial/joint ratio, and overlap efficiency.
5. **Dependency-correct composition:** include drafting, transfer, conversion,
   synchronization, verification, rejection, rollback, and catch-up.
6. **Exact lifecycle:** cover accept, early reject, EOS, cancellation, request
   detachment, membership changes, deadlines, and stale results.
7. **Separate claims:** report B=1 latency, aggregate throughput, and per-request
   p50/p95 independently. A queue-width win is not a local decode win.
8. **Composition:** rerun against the strongest batching, megakernel, cache, and
   speculative configuration—not a convenient decomposed baseline.

Keep every heterogeneous path default-off until it passes all eight gates on
the target model and workload.

> **Prior art.** Core ML exposes Neural Engine execution and supports weight
> palettization/quantization; speculative decoding supplies the exact
> target-verification contract. These are adjacent building blocks, not an
> end-to-end performance claim for an ANE proposer beside a persistent Metal
> target. **How we differ.** We measured device overlap, real trained MTP cuts,
> handoff formats, verifier ceilings, cross-request scheduling, and the live
> continuous-batch incumbent on one released recurrent MoE. **Our finding.**
> *Extends adjacent prior art*—the engines overlap, but dependency-ready work,
> transfer shape, consumer dtype, and preservation of GPU batch sharing decide
> whether that overlap becomes useful throughput.

---

## Sources

- Apple Core ML Tools, [Palettization](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html).
- Apple Core ML Tools, [Weight quantization](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html).
- Leviathan et al., *Fast Inference from Transformers via Speculative
  Decoding* — ICML 2023, arXiv:2211.17192.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative
  Sampling* — arXiv:2302.01318.
