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
