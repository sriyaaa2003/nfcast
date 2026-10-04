"""Cloud-native 5G network-function (NF) telemetry simulator with known dynamics.

Why a simulator: lab and testbed NF telemetry is not available in the open, and a simulator lets every claim be checked
against known structure. The price is that results describe this generator, not a live network (stated in the README).

Each NF instance sees its own regional traffic (diurnal and weekly pattern, heavy-tailed ON/OFF bursts, scripted demand
surges) and turns it into resource demand with an NF-specific model:

* upf   user plane: CPU proportional to throughput, memory follows load plus a slow leak that ends in a pod restart
* amf   signalling: request rate grows sub-linearly with traffic, bursty, with registration storms at surges
* smf   sessions: lagged, session-driven memory, a smaller leak
* cuup  RAN user plane moved to the cloud: CPU follows traffic plus a periodic housekeeping task

A reactive horizontal autoscaler (HPA semantics: scale up at once, scale down after a stabilisation window) sets the
replicas, and a power model turns replicas and utilisation into energy, so the energy series carries the reactive
over-provisioning that proactive control is meant to remove.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, NFProfile, TARGETS

REF_REQUEST_RATE = 50.0       # requests per second at the reference traffic level (amf, smf)


def pareto_mean(alpha: float, minimum: float) -> float:
    return alpha * minimum / (alpha - 1.0)


def onoff_state(rng: np.random.Generator, n: int, alpha_on: float, alpha_off: float, min_on: float, min_off: float) -> np.ndarray:
    """Boolean ON/OFF series of length n with Pareto sojourn times (1 < alpha < 2 gives long-range dependence)."""
    m_on, m_off = pareto_mean(alpha_on, min_on), pareto_mean(alpha_off, min_off)
    duty = m_on / (m_on + m_off)
    pairs = int(1.5 * n / (m_on + m_off)) + 16
    while True:
        on = min_on * (1.0 + rng.pareto(alpha_on, pairs))
        off = min_off * (1.0 + rng.pareto(alpha_off, pairs))
        start_on = rng.random() < duty
        seg = np.empty(2 * pairs)
        seg[0::2], seg[1::2] = (on, off) if start_on else (off, on)
        bounds = np.cumsum(seg)
        if bounds[-1] >= n:
            break
        pairs *= 2
    offset = rng.uniform(0.0, min(bounds[-1] - n, 10.0 * (m_on + m_off)))
    idx = np.searchsorted(bounds, np.arange(n) + offset, side="right")
    return ((idx + (0 if start_on else 1)) % 2) == 0


def ema(x: np.ndarray, minutes: float) -> np.ndarray:
    """Exponential moving average with time constant `minutes` (one-minute steps)."""
    a = 1.0 - np.exp(-1.0 / max(minutes, 1e-6))
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc += a * (v - acc)
        out[i] = acc
    return out


def sawtooth_leak(n: int, rate: float, limit: float, phase: float) -> np.ndarray:
    """Extra memory that grows by `rate` per minute and drops to zero when it reaches `limit` (a pod restart)."""
    if rate <= 0:
        return np.zeros(n)
    period = limit / rate
    return rate * np.mod(np.arange(n) + phase * period, period)


def traffic_and_events(cfg: Config, rng: np.random.Generator, n: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    t = cfg.sim.traffic
    scale = float(rng.lognormal(0.0, t.instance_scale_sigma))
    minute = np.arange(n)
    day = minute / 1440.0
    phase = rng.uniform(0.0, 0.12)
    daily = 1.0 + t.daily_amp1 * np.sin(2 * np.pi * (day - 0.25 - phase)) + t.daily_amp2 * np.sin(4 * np.pi * (day - 0.1 - phase))
    weekend = np.where((minute // 1440) % 7 >= 5, t.weekend_factor, 1.0)
    rates = rng.lognormal(np.log(t.rate_mbps_mean) - 0.5 * t.rate_sigma ** 2, t.rate_sigma, t.n_sources)
    burst = np.zeros(n)
    for r in rates:
        burst += r * onoff_state(rng, n, t.on_alpha, t.off_alpha, t.on_min_minutes, t.off_min_minutes)
    traffic = (t.base_mbps * daily * weekend + burst) * scale
    surge = np.ones(n)
    events: list[tuple[int, int]] = []
    ramp = 10
    for _ in range(t.events_per_instance):
        dur = int(rng.integers(t.event_minutes_min, t.event_minutes_max + 1))
        start = int(rng.integers(1440, n - dur - ramp))
        mult = float(rng.uniform(t.event_multiplier_min, t.event_multiplier_max))
        up = np.clip((minute - start) / ramp, 0, 1)
        down = np.clip((start + dur - minute) / ramp, 0, 1)
        surge *= 1.0 + (mult - 1.0) * np.minimum(up, down)
        events.append((start, dur))
    traffic = traffic * surge * rng.lognormal(0.0, t.noise_cv, n)
    return traffic, events


def hpa_replicas(cpu_cores: np.ndarray, pod_cores: float, cfg: Config) -> np.ndarray:
    """Reactive horizontal autoscaling: desired replicas from the last observation; scale up at once, scale down only
    once the desired count has stayed lower for the whole stabilisation window (max over the window)."""
    h = cfg.sim.hpa
    desired = np.ceil(cpu_cores / (pod_cores * h.target_util))
    s = pd.Series(desired)
    window = max(h.down_stabilization_minutes, 1)
    rep = s.rolling(window, min_periods=1).max().shift(1).fillna(desired[0]).to_numpy()
    return np.clip(rep, h.min_replicas, h.max_replicas)


def nf_demand(profile: NFProfile, cfg: Config, traffic: np.ndarray, events: list[tuple[int, int]],
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """CPU demand (cores) and memory demand (GiB) of one NF instance."""
    n = len(traffic)
    ref = float(np.median(traffic))
    if profile.name in ("upf", "cuup"):
        load = traffic
    else:
        shape = 1.0 / max(profile.burst_cv ** 2, 1e-6)
        noise = rng.gamma(shape, 1.0 / shape, n) if profile.burst_cv > 0 else np.ones(n)
        load = REF_REQUEST_RATE * (traffic / ref) ** profile.signalling_exponent * noise
        if profile.name == "amf":                            # registration storms right after a surge starts
            minute = np.arange(n)
            for start, _ in events:
                k = minute - start
                load = load * (1.0 + 2.0 * np.where(k >= 0, np.exp(-np.clip(k, 0, None) / 6.0), 0.0))
    cpu = profile.cpu_base_cores + profile.cpu_per_unit * load
    if profile.name == "cuup" and profile.periodic_minutes > 0:
        phase = int(rng.integers(0, profile.periodic_minutes))
        active = ((np.arange(n) + phase) % profile.periodic_minutes) < profile.periodic_duration_minutes
        cpu = cpu + profile.periodic_cores * active
    cpu = cpu * rng.lognormal(0.0, profile.cpu_noise_cv, n)
    leak = sawtooth_leak(n, profile.leak_gib_per_min, 0.35 * profile.pod_mem_gib, float(rng.uniform(0, 1)))
    mem = profile.mem_base_gib + profile.mem_per_unit_gib * ema(load, profile.mem_ema_minutes) + leak
    return np.maximum(cpu, 0.05), mem


def power_watts(util: np.ndarray, replicas: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    p = cfg.sim.power
    u = np.clip(util, 0.0, 1.0)
    per_pod = p.idle_w + (p.max_w - p.idle_w) * u ** p.alpha
    return replicas * per_pod * rng.lognormal(0.0, p.noise_cv, len(util))


def simulate(cfg: Config) -> pd.DataFrame:
    """Wide frame: one row per (nf, instance, minute) with the five forecast targets plus `replicas`."""
    n = cfg.sim.days * 1440
    frames = []
    for k, profile in enumerate(cfg.sim.nf_profiles):
        for inst in range(profile.instances):
            rng = np.random.default_rng([cfg.seed, k, inst])
            traffic, events = traffic_and_events(cfg, rng, n)
            cpu, mem = nf_demand(profile, cfg, traffic, events, rng)
            replicas = hpa_replicas(cpu, profile.pod_cores, cfg)
            util = cpu / (replicas * profile.pod_cores)
            energy = power_watts(util, replicas, cfg, rng)
            frames.append(pd.DataFrame({
                "nf": profile.name, "instance": inst, "minute": np.arange(n, dtype=np.int32),
                "traffic_mbps": traffic.astype(np.float32), "cpu_cores": cpu.astype(np.float32),
                "cpu_util_pod": np.minimum(util, 1.5).astype(np.float32), "mem_gib": mem.astype(np.float32),
                "energy_w": energy.astype(np.float32), "replicas": replicas.astype(np.float32)}))
    return pd.concat(frames, ignore_index=True)


def split_bounds(cfg: Config) -> dict[str, tuple[int, int]]:
    tr = cfg.split.train_days * 1440
    ca = tr + cfg.split.calib_days * 1440
    return {"train": (0, tr), "calib": (tr, ca), "test": (ca, cfg.sim.days * 1440)}


def series_dict(df: pd.DataFrame, cfg: Config) -> dict[tuple[str, int, str], np.ndarray]:
    """(nf, instance, target) -> float32 array of length days*1440."""
    out = {}
    for (nf, inst), g in df.groupby(["nf", "instance"], sort=True):
        g = g.sort_values("minute")
        for target in TARGETS:
            out[(nf, int(inst), target)] = g[target].to_numpy(dtype=np.float32)
    return out
