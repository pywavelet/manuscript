"""
Colab benchmark script for wdm_transform paper.

Run on Google Colab with a GPU runtime (T4 or A100 recommended).
Produces two JSON files:

  benchmark_results_colab.json  -- runtime + error table (replaces benchmark_results.json)
  vmap_results_colab.json       -- vectorization (vmap) speedup sweep

===========================================================================
COLAB SETUP — run this cell first, then run this script
===========================================================================

  !pip install -q "wdm-transform[jax]"

  # If installing from a local wheel uploaded to /content/:
  # !pip install -q /content/wdm_transform-*.whl

  import subprocess, json
  exec(open("colab_benchmarks.py").read())

  # Download results when done:
  from google.colab import files
  files.download("benchmark_results_colab.json")
  files.download("vmap_results_colab.json")
===========================================================================
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 0.  Hardware fingerprint
# ---------------------------------------------------------------------------

def _gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip().splitlines()[0]
    except Exception:
        return "CPU-only"

def _cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"

hardware: dict[str, str] = {
    "gpu": _gpu_name(),
    "cpu": _cpu_name(),
    "python": platform.python_version(),
    "platform": platform.platform(),
}
print("Hardware:")
for k, v in hardware.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 1.  Imports
# ---------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from wdm_transform import TimeSeries, WDM

print(f"\nJAX devices : {jax.devices()}")
print(f"JAX backend : {jax.default_backend()}")

# ---------------------------------------------------------------------------
# 2.  Shared helpers
# ---------------------------------------------------------------------------

NUM_RUNS = 7
A_PARAM  = 1.0 / 3.0
D_PARAM  = 1.0
DT       = 1.0


def _nt(n: int) -> int:
    """Choose nt as the largest even divisor of n close to sqrt(n)."""
    nt = int(n ** 0.5)
    nt = nt if nt % 2 == 0 else nt - 1
    while n % nt != 0:
        nt -= 2
        if nt < 2:
            raise ValueError(f"Cannot find valid nt for n={n}")
    return nt


def _bench_numpy(n: int, rng: np.random.Generator) -> dict[str, Any]:
    nt = _nt(n)
    x  = rng.standard_normal(n)
    ts = TimeSeries(x, dt=DT, backend="numpy")

    # Warmup
    wdm = WDM.from_time_series(ts, nt=nt, a=A_PARAM, d=D_PARAM)
    ts_rec = wdm.to_time_series()

    fwd_times, inv_times = [], []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        wdm = WDM.from_time_series(ts, nt=nt, a=A_PARAM, d=D_PARAM)
        fwd_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ts_rec = wdm.to_time_series()
        inv_times.append(time.perf_counter() - t0)

    x_rec = np.asarray(ts_rec.data).squeeze()
    err   = float(np.max(np.abs(x - x_rec)))
    rel   = float(np.linalg.norm(x - x_rec) / np.linalg.norm(x))
    return {
        "nt": nt, "nf": n // nt,
        "forward_mean_s":    float(np.mean(fwd_times)),
        "forward_std_s":     float(np.std(fwd_times)),
        "inverse_mean_s":    float(np.mean(inv_times)),
        "inverse_std_s":     float(np.std(inv_times)),
        "max_abs_error":     err,
        "relative_l2_error": rel,
    }


def _bench_jax(n: int, rng: np.random.Generator) -> dict[str, Any]:
    nt   = _nt(n)
    x_np = rng.standard_normal(n)
    ts   = TimeSeries(x_np, dt=DT, backend="jax")

    # Warmup — triggers JIT compilation
    wdm    = WDM.from_time_series(ts, nt=nt, a=A_PARAM, d=D_PARAM, backend="jax")
    wdm.coeffs.block_until_ready()
    ts_rec = wdm.to_time_series()
    ts_rec.data.block_until_ready()

    fwd_times, inv_times = [], []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        wdm = WDM.from_time_series(ts, nt=nt, a=A_PARAM, d=D_PARAM, backend="jax")
        wdm.coeffs.block_until_ready()
        fwd_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ts_rec = wdm.to_time_series()
        ts_rec.data.block_until_ready()
        inv_times.append(time.perf_counter() - t0)

    x_rec = np.asarray(ts_rec.data).squeeze()
    err   = float(np.max(np.abs(x_np - x_rec)))
    rel   = float(np.linalg.norm(x_np - x_rec) / np.linalg.norm(x_np))
    return {
        "nt": nt, "nf": n // nt,
        "forward_mean_s":    float(np.mean(fwd_times)),
        "forward_std_s":     float(np.std(fwd_times)),
        "inverse_mean_s":    float(np.mean(inv_times)),
        "inverse_std_s":     float(np.std(inv_times)),
        "max_abs_error":     err,
        "relative_l2_error": rel,
    }


# ---------------------------------------------------------------------------
# 3.  Runtime sweep: N = 2^11 … 2^22
# ---------------------------------------------------------------------------
# Upper limit 2^22 = 4 194 304 covers LISA 1-year at f_s = 0.05 Hz.
# Each signal is ~32 MB as float64; Colab T4 has ~15 GB host RAM, fine.

N_VALUES = [2**k for k in range(11, 23)]

rng = np.random.default_rng(42)

results: dict[str, Any] = {
    "hardware":   hardware,
    "num_runs":   NUM_RUNS,
    "parameters": {"a": A_PARAM, "d": D_PARAM, "dt": DT},
    "n_values":   N_VALUES,
    "numpy":      {},
    "jax":        {},
}

print("\n=== NumPy (CPU) ===")
for n in N_VALUES:
    print(f"  N={n:>10,} ...", end=" ", flush=True)
    r = _bench_numpy(n, rng)
    results["numpy"][str(n)] = r
    fwd_ms = r["forward_mean_s"] * 1e3
    inv_ms = r["inverse_mean_s"] * 1e3
    print(f"fwd {fwd_ms:7.2f} ms  inv {inv_ms:7.2f} ms  err {r['max_abs_error']:.2e}")

print("\n=== JAX (GPU/CPU) ===")
for n in N_VALUES:
    print(f"  N={n:>10,} ...", end=" ", flush=True)
    r = _bench_jax(n, rng)
    results["jax"][str(n)] = r
    fwd_ms = r["forward_mean_s"] * 1e3
    inv_ms = r["inverse_mean_s"] * 1e3
    print(f"fwd {fwd_ms:7.2f} ms  inv {inv_ms:7.2f} ms  err {r['max_abs_error']:.2e}")

out = Path("benchmark_results_colab.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {out.resolve()}")

# Quick speedup summary
try:
    n_ref = str(2**20)
    sp_fwd = results["numpy"][n_ref]["forward_mean_s"] / results["jax"][n_ref]["forward_mean_s"]
    sp_inv = results["numpy"][n_ref]["inverse_mean_s"] / results["jax"][n_ref]["inverse_mean_s"]
    print(f"\nAt N=2^20: JAX speedup  fwd={sp_fwd:.1f}×  inv={sp_inv:.1f}×")
except KeyError:
    pass


# ---------------------------------------------------------------------------
# 4.  vmap vectorization sweep
# ---------------------------------------------------------------------------
# Fixed signal size N_VMAP. Sweep batch size B = 1 … B_MAX.
# Three strategies:
#   jax_vmap  — jax.vmap over B independent signals (true SIMD)
#   jax_loop  — Python for-loop over B sequential JAX calls
#   numpy_loop — Python for-loop over B NumPy calls

N_VMAP   = 2**18          # ~256 k samples; fits easily in VRAM for large batches
B_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
V_RUNS   = 5

print(f"\n=== vmap sweep (N={N_VMAP:,} per signal) ===")

nt_v = _nt(N_VMAP)
ts_template_jax = TimeSeries(
    np.zeros(N_VMAP), dt=DT, backend="jax"
)  # shape only; replaced per batch

# Build the vmapped forward function once
def _fwd_jax_single(x: jnp.ndarray) -> jnp.ndarray:
    """Forward WDM for a single 1-D signal (shape [N])."""
    ts = TimeSeries(x, dt=DT, backend="jax")
    wdm = WDM.from_time_series(ts, nt=nt_v, a=A_PARAM, d=D_PARAM, backend="jax")
    return wdm.coeffs.squeeze(0)          # remove the singleton batch axis


try:
    _fwd_jax_batch = jax.vmap(_fwd_jax_single)
    # Warmup at max batch size to compile all shapes we'll need
    _x_warm = jnp.ones((B_VALUES[-1], N_VMAP))
    _fwd_jax_batch(_x_warm).block_until_ready()
    HAS_VMAP = True
    print("  jax.vmap compiled OK")
except Exception as exc:
    HAS_VMAP = False
    print(f"  jax.vmap unavailable: {exc}")

vmap_results: dict[str, Any] = {
    "hardware":      hardware,
    "n_per_signal":  N_VMAP,
    "nt":            nt_v,
    "nf":            N_VMAP // nt_v,
    "num_runs":      V_RUNS,
    "batch_sizes":   B_VALUES,
    "jax_vmap":      {},
    "jax_loop":      {},
    "numpy_loop":    {},
}

rng2 = np.random.default_rng(7)

for B in B_VALUES:
    xs_np  = rng2.standard_normal((B, N_VMAP))
    xs_jax = jnp.array(xs_np)

    # ---- JAX vmap ----
    if HAS_VMAP:
        _fwd_jax_batch(xs_jax).block_until_ready()   # warmup for this B
        times = []
        for _ in range(V_RUNS):
            t0 = time.perf_counter()
            out = _fwd_jax_batch(xs_jax)
            out.block_until_ready()
            times.append(time.perf_counter() - t0)
        vmap_results["jax_vmap"][str(B)] = {
            "total_mean_s":        float(np.mean(times)),
            "total_std_s":         float(np.std(times)),
            "per_signal_mean_ms":  float(np.mean(times)) / B * 1e3,
        }

    # ---- JAX loop ----
    # Warmup
    for i in range(B):
        ts_i = TimeSeries(xs_jax[i], dt=DT, backend="jax")
        WDM.from_time_series(ts_i, nt=nt_v, a=A_PARAM, d=D_PARAM,
                             backend="jax").coeffs.block_until_ready()
    times = []
    for _ in range(V_RUNS):
        t0 = time.perf_counter()
        for i in range(B):
            ts_i = TimeSeries(xs_jax[i], dt=DT, backend="jax")
            WDM.from_time_series(ts_i, nt=nt_v, a=A_PARAM, d=D_PARAM,
                                 backend="jax").coeffs.block_until_ready()
        times.append(time.perf_counter() - t0)
    vmap_results["jax_loop"][str(B)] = {
        "total_mean_s":        float(np.mean(times)),
        "total_std_s":         float(np.std(times)),
        "per_signal_mean_ms":  float(np.mean(times)) / B * 1e3,
    }

    # ---- NumPy loop ----
    times = []
    for _ in range(V_RUNS):
        t0 = time.perf_counter()
        for i in range(B):
            ts_i = TimeSeries(xs_np[i], dt=DT, backend="numpy")
            WDM.from_time_series(ts_i, nt=nt_v, a=A_PARAM, d=D_PARAM)
        times.append(time.perf_counter() - t0)
    vmap_results["numpy_loop"][str(B)] = {
        "total_mean_s":        float(np.mean(times)),
        "total_std_s":         float(np.std(times)),
        "per_signal_mean_ms":  float(np.mean(times)) / B * 1e3,
    }

    vmap_str = (
        f"vmap {vmap_results['jax_vmap'][str(B)]['per_signal_mean_ms']:6.2f} ms/sig  "
        if HAS_VMAP else ""
    )
    print(
        f"  B={B:>4}  "
        f"{vmap_str}"
        f"jax-loop {vmap_results['jax_loop'][str(B)]['per_signal_mean_ms']:6.2f} ms/sig  "
        f"numpy    {vmap_results['numpy_loop'][str(B)]['per_signal_mean_ms']:6.2f} ms/sig"
    )

vmap_out = Path("vmap_results_colab.json")
vmap_out.write_text(json.dumps(vmap_results, indent=2))
print(f"\nSaved → {vmap_out.resolve()}")

print("\n=== Done — download files ===")
print("  from google.colab import files")
print("  files.download('benchmark_results_colab.json')")
print("  files.download('vmap_results_colab.json')")
