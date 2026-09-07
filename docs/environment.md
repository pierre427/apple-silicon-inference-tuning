# Environment & build

This is the highest-leverage chapter in the guide and the one most people skip.
On M5 hardware, the difference between a runtime that reaches the Neural
Accelerators and one that doesn't is **~3.3–4× on prefill** — on the *same
machine, same model, same code*. And it is easy to lose without any error
message: a stale interpreter, a wheel built without the right flag, or an
absent-minded reinstall will silently drop you back to a "fast M4" and nothing
will tell you except your TTFT.

Get this chapter right before you tune anything else. A 5% win from some clever
serving lever is noise if you're leaving a 4× win on the table here.

## The two gates for the M5 matrix path

The Neural-Accelerator kernels are gated on **both** of:

1. **MLX ≥ 0.30.0** — earlier MLX has no accelerator path and will never use
   the matrix units regardless of the OS.
2. **macOS ≥ 26.2** — the Metal 4 tensor-op support the kernels compile
   against.

Miss either one and you get correct results at M4-class prefill speed. There is
no warning; the kernels simply aren't in the binary or aren't dispatched.

!!! danger "Never serve from the system Python"
    The stock `/usr/bin/python3` on macOS resolves to an old PyPI MLX with no
    accelerator path. It runs your model correctly and quietly gives up the
    entire prefill win. Serve from a dedicated virtualenv whose MLX satisfies
    both gates above — treat "which interpreter launched this server" as a
    first-class part of your deploy, not an afterthought.

## Prove the accelerators are live

Do not assume — measure. A pure-GEMM microbenchmark with the GPU otherwise
idle takes seconds and is the contention-resistant way to know whether you're
on the fast path. The signature of a working accelerator runtime is roughly
**3.5–4× the fp16/bf16 GEMM throughput** of a non-accelerator runtime at large
`n`:

```python
# na_check.py — is this interpreter reaching the M5 matrix units?
import mlx.core as mx, time

def gemm_tflops(n=4096, dtype=mx.float16, iters=50):
    a = mx.random.normal((n, n)).astype(dtype)
    b = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, b)
    # warmup (kernel specialization, lazy alloc)
    for _ in range(3):
        mx.eval(a @ b)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    mx.eval(c)
    dt = (time.perf_counter() - t0) / iters
    flop = 2 * n**3           # one matmul
    return flop / dt / 1e12

for n in (2048, 4096, 8192):
    print(f"n={n:<5} {gemm_tflops(n):.0f} TFLOP/s")
```

On an M5 Max with the accelerator kernels live this prints on the order of
~1,600–1,700 TFLOP/s at n=4096; without them, ~450. If you see the low number,
one of the two gates isn't satisfied — fix that before anything else. (Print
`mx.__version__` alongside it; you want ≥ 0.30, and in practice a recent
0.31/0.32 build.)

!!! note "Microbench, not model-level, for this check"
    Use the GEMM microbench, not a full model run, to decide whether the
    accelerators are live. Model-level numbers are contaminated by background
    machine contention (see [Measuring it right](measurement.md)); a
    single-kernel GEMM on an idle GPU is the clean signal.

## The source-build trap: losing NAX on a rebuild

If you build MLX from source (common when you need a version newer than the
current PyPI wheel), there is a **silent** failure mode worth burning into
memory:

!!! danger "A wrong deployment target compiles the accelerators out"
    Building the MLX source **without** setting the macOS deployment target to
    a value ≥ 26.2 compiles in the `MLX_METAL_NO_NAX` path and *drops every
    accelerator kernel* — no error, no warning, just a ~3–4× prefill
    regression the next time you serve. This has bitten real deployments.
    Always pin the deployment target for a source build:

    ```bash
    # building MLX from source: pin the deployment target so the
    # Neural-Accelerator kernels are actually compiled in
    MACOSX_DEPLOYMENT_TARGET=26.2 pip install .
    ```

    Then immediately run `na_check.py` above as a smoke test. The build
    succeeding tells you nothing; the GEMM throughput tells you everything.

> **Prior art.** MLX's build system and release notes document the version and
> OS gates for the M5 tensor kernels; the deployment-target flag is standard
> macOS build hygiene.
> **How we differ.** We hit the *silent* regression in production and turned
> it into a fixed recipe plus a boot-time GEMM smoke test.
> **Our finding.** *Extends prior art* — the gate is documented, but "a bad
> deployment target silently compiles the accelerators out, with no error"
> is an operational failure mode we learned the hard way and now guard against.

## Wheel vs source build

You do not always need a source build. Once the PyPI wheel catches up to a
version ≥ 0.30 built for the right target, it can be **as fast as or faster
than** a local source build and it is **not** subject to the deployment-target
trap above — a measured wheel hit ~23.6 TFLOP/s fp16 GEMM where a
contemporaneous local source build sat at ~19.7. Prefer the wheel when it's new
enough; reach for a source build only when you need something the wheel doesn't
have yet (a specific fix, a bleeding-edge feature), and when you do, pin the
deployment target and smoke-test.

> **Prior art.** Conventional wisdom is that a from-source build is faster than
> a pip wheel.
> **How we differ.** We microbenchmarked both on the same machine.
> **Our finding.** *Diverges from prior art* — once the published wheel is
> new enough, it matched or beat a local source build (~23.6 vs ~19.7 TFLOP/s
> fp16 GEMM) *and* sidesteps the deployment-target trap. Prefer the wheel when
> it's current; source-build only for something the wheel lacks.

Decision order:

1. Is the current **PyPI wheel** ≥ 0.30 and are you on macOS ≥ 26.2? → use the
   wheel, run `na_check.py`, done.
2. Need something newer than the wheel? → **source build** with
   `MACOSX_DEPLOYMENT_TARGET=26.2`, then `na_check.py`.
3. Either way, the smoke test is not optional.

## Version-drift discipline

Runtimes drift. A machine accumulates several virtualenvs at different MLX
versions — one pinned for a benchmark, one tracking latest, one holding a
specific patched wheel — and it is genuinely easy to serve from the wrong one
after an upgrade. Two rules keep you honest:

- **Resolve the interpreter from today's serving profile, not yesterday's
  upgrade.** When you `pip install -U` one environment, that says nothing about
  which environment your service actually launches from. Pin the launch path
  explicitly (an absolute interpreter path in your service definition) and
  verify it, rather than trusting `python` on `PATH`.
- **Keep the accelerator smoke test in your startup or health check.** A
  one-line GEMM assertion at boot (throughput above a floor) turns a silent
  3–4× regression into a loud, immediate failure. This is cheap insurance
  against the entire failure class this chapter is about.

!!! note "Correctness bugs hide until real Metal"
    A related discipline point: MLX numerical/correctness issues frequently
    surface **only on real Metal hardware** — a CPU-only or no-GPU environment
    cannot validate them. So both your speed smoke test and your correctness
    checks have to run on the actual device you serve from, not in CI on a
    headless box. More in [Measuring it right](measurement.md).

## Checklist

- [ ] Serving interpreter is a dedicated venv, **not** system Python.
- [ ] `mx.__version__` ≥ 0.30, macOS ≥ 26.2.
- [ ] `na_check.py` shows ~3.5–4× the non-accelerator GEMM throughput.
- [ ] If source-built: `MACOSX_DEPLOYMENT_TARGET=26.2` was set.
- [ ] The launch path pins the interpreter absolutely; a boot-time GEMM floor
      check guards against drift.

## Sources

- Apple ML Research, *Exploring LLMs with MLX and the Neural Accelerators in
  the M5 GPU* — <https://machinelearning.apple.com/research/exploring-llms-mlx-m5>
- MLX (`ml-explore/mlx`) release notes for the Metal tensor-op / M5 kernel
  support.
- mlx-lm README — memory-limit and serving notes.
