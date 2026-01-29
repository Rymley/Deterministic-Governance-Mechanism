#!/usr/bin/env python3
"""
Benchmark harness.

Reports p50/p95/p99 latency (ms) for:
  - Engine loop: (candidates, steps, active_substrate_size)
  - Shard retrieval: (total_vectors, shard_size, top_k)
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import hashlib

from material_field_engine import MaterialFieldEngine, VerifiedSubstrate, Vector2D
from substrate_sharding import CompactVector, ShardedSubstrate, SubstrateShard
from deterministic_rng import normal, uniform, uint31


def _pctl(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    idx = int(round(p * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def _fmt_ms(ns: float) -> float:
    return float(ns) / 1e6


@dataclass(frozen=True)
class Summary:
    runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float


def _summarize_ns(samples_ns: Sequence[int]) -> Summary:
    s = sorted(float(x) for x in samples_ns)
    return Summary(
        runs=len(samples_ns),
        p50_ms=_fmt_ms(_pctl(s, 0.50)),
        p95_ms=_fmt_ms(_pctl(s, 0.95)),
        p99_ms=_fmt_ms(_pctl(s, 0.99)),
    )


def _print_engine_header() -> None:
    print(
        "benchmark,candidates,steps,active_substrate_states,warmup,runs,"
        "p50_init_ms,p95_init_ms,p99_init_ms,mean_init_ms,"
        "p50_run_ms,p95_run_ms,p99_run_ms,mean_run_ms,"
        "p50_total_ms,p95_total_ms,p99_total_ms,mean_total_ms"
    )


def _print_retrieval_header() -> None:
    print(
        "benchmark,total_vectors,shard_size,top_k,warmup,runs,"
        "p50_ms,p95_ms,p99_ms,mean_ms"
    )


def bench_engine(
    *,
    candidates: int,
    steps: int,
    active_substrate_states: int,
    warmup: int,
    runs: int,
    seed: int,
) -> None:
    seed_bytes = hashlib.blake2b(f"bench_engine|{seed}".encode("utf-8"), digest_size=16).digest()

    # Deterministic substrate states.
    substrate_points: list[tuple[float, float]] = []
    for i in range(active_substrate_states):
        x = uniform(seed_bytes, i * 2, -1.0, 1.0)
        y = uniform(seed_bytes, i * 2 + 1, -1.0, 1.0)
        substrate_points.append((x, y))

    # Deterministic candidates.
    #
    # Use substrate-aligned candidates with a tiny perturbation so:
    # - no exclusions occur (fixed work per step)
    # - the loop runs all `steps` iterations (no early termination)
    if active_substrate_states == 0:
        candidate_tuples = [
            (uniform(seed_bytes, 1_000_000 + i * 2, -1.0, 1.0), uniform(seed_bytes, 1_000_000 + i * 2 + 1, -1.0, 1.0))
            for i in range(candidates)
        ]
    else:
        noise_seed = hashlib.blake2b(f"bench_engine_noise|{seed}".encode("utf-8"), digest_size=16).digest()
        candidate_tuples = []
        for i in range(candidates):
            bx, by = substrate_points[i % active_substrate_states]
            nx = normal(noise_seed, i * 4, mean=0.0, std=1e-6)
            ny = normal(noise_seed, i * 4 + 2, mean=0.0, std=1e-6)
            candidate_tuples.append((bx + nx, by + ny))

    substrate = VerifiedSubstrate(elastic_modulus_mode="multiplicative", elastic_modulus_sigma=0.40)
    for x, y in substrate_points:
        substrate.add_verified_state(Vector2D(x=float(x), y=float(y), properties=None))

    engine = MaterialFieldEngine(substrate, lambda_min=0.35, lambda_max=1.20, inference_steps=steps)
    engine.phase_controller.nucleation_threshold = 0.375
    engine.phase_controller.quenching_threshold = 0.875

    # Warmup.
    for _ in range(warmup):
        engine.initialize_candidates(candidate_tuples)
        engine.run_inference()

    init_ns: list[int] = []
    run_ns: list[int] = []
    total_ns: list[int] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        engine.initialize_candidates(candidate_tuples)
        t1 = time.perf_counter_ns()
        engine.run_inference()
        t2 = time.perf_counter_ns()
        init_ns.append(t1 - t0)
        run_ns.append(t2 - t1)
        total_ns.append(t2 - t0)

    init_s = _summarize_ns(init_ns)
    run_s = _summarize_ns(run_ns)
    total_s = _summarize_ns(total_ns)

    init_mean = _fmt_ms(statistics.mean(init_ns))
    run_mean = _fmt_ms(statistics.mean(run_ns))
    total_mean = _fmt_ms(statistics.mean(total_ns))

    print(
        f"engine,{candidates},{steps},{active_substrate_states},{warmup},{runs},"
        f"{init_s.p50_ms:.6f},{init_s.p95_ms:.6f},{init_s.p99_ms:.6f},{init_mean:.6f},"
        f"{run_s.p50_ms:.6f},{run_s.p95_ms:.6f},{run_s.p99_ms:.6f},{run_mean:.6f},"
        f"{total_s.p50_ms:.6f},{total_s.p95_ms:.6f},{total_s.p99_ms:.6f},{total_mean:.6f}"
    )


def _build_sharded_substrate(
    *,
    total_vectors: int,
    shard_size: int,
    seed: int,
) -> ShardedSubstrate:
    seed_bytes = hashlib.blake2b(f"bench_retrieval|{seed}".encode("utf-8"), digest_size=16).digest()
    substrate = ShardedSubstrate(shard_size=shard_size)

    domains = ("biology", "geography", "physics")
    shard_count = int(math.ceil(total_vectors / shard_size))
    next_vector_idx = 0

    for shard_id in range(shard_count):
        domain = domains[shard_id % len(domains)]
        vectors: list[CompactVector] = []
        for _ in range(shard_size):
            if next_vector_idx >= total_vectors:
                break
            x = uniform(seed_bytes, next_vector_idx * 4, -1.0, 1.0)
            y = uniform(seed_bytes, next_vector_idx * 4 + 1, -1.0, 1.0)
            domain_hash = uint31(seed_bytes, next_vector_idx * 4 + 2)
            vectors.append(CompactVector(x=float(x), y=float(y), shard_id=shard_id, domain_hash=domain_hash))
            next_vector_idx += 1

        if not vectors:
            break

        centroid = vectors[0]
        shard = SubstrateShard(shard_id=shard_id, domain=domain, centroid=centroid, vectors=vectors)
        substrate.add_shard(shard)

    return substrate


def bench_retrieval(
    *,
    total_vectors: int,
    shard_size: int,
    top_k: int,
    warmup: int,
    runs: int,
    seed: int,
) -> None:
    substrate = _build_sharded_substrate(total_vectors=total_vectors, shard_size=shard_size, seed=seed)
    q_seed = hashlib.blake2b(f"bench_retrieval_query|{seed}".encode("utf-8"), digest_size=16).digest()

    # Fixed query; force domain selection deterministically via domain_hash % 3.
    qx = uniform(q_seed, 0, -1.0, 1.0)
    qy = uniform(q_seed, 1, -1.0, 1.0)
    query = CompactVector(x=float(qx), y=float(qy), shard_id=-1, domain_hash=0)

    for _ in range(warmup):
        substrate.retrieve_relevant_shards(query_vector=query, top_k=top_k)

    samples_ns: list[int] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        substrate.retrieve_relevant_shards(query_vector=query, top_k=top_k)
        t1 = time.perf_counter_ns()
        samples_ns.append(t1 - t0)

    summary = _summarize_ns(samples_ns)
    mean_ms = _fmt_ms(statistics.mean(samples_ns))
    print(
        f"shard_retrieve,{total_vectors},{shard_size},{top_k},{warmup},{runs},"
        f"{summary.p50_ms:.6f},{summary.p95_ms:.6f},{summary.p99_ms:.6f},{mean_ms:.6f}"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_engine = sub.add_parser("engine")
    p_engine.add_argument("--candidates", type=int, nargs="+", default=[4, 16, 64])
    p_engine.add_argument("--steps", type=int, nargs="+", default=[8, 16])
    p_engine.add_argument("--active-substrate-states", type=int, nargs="+", default=[3, 64, 256])
    p_engine.add_argument("--warmup", type=int, default=50)
    p_engine.add_argument("--runs", type=int, default=500)
    p_engine.add_argument("--seed", type=int, default=0)

    p_ret = sub.add_parser("retrieval")
    p_ret.add_argument("--total-vectors", type=int, nargs="+", default=[4096, 65536, 262144])
    p_ret.add_argument("--shard-size", type=int, default=64)
    p_ret.add_argument("--top-k", type=int, default=8)
    p_ret.add_argument("--warmup", type=int, default=200)
    p_ret.add_argument("--runs", type=int, default=2000)
    p_ret.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv[1:])

    if args.cmd == "engine":
        _print_engine_header()
        for cand in args.candidates:
            for steps in args.steps:
                for sub_states in args.active_substrate_states:
                    bench_engine(
                        candidates=cand,
                        steps=steps,
                        active_substrate_states=sub_states,
                        warmup=args.warmup,
                        runs=args.runs,
                        seed=args.seed,
                    )
        return 0

    if args.cmd == "retrieval":
        _print_retrieval_header()
        for total in args.total_vectors:
            bench_retrieval(
                total_vectors=total,
                shard_size=args.shard_size,
                top_k=args.top_k,
                warmup=args.warmup,
                runs=args.runs,
                seed=args.seed,
            )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(argv=list(__import__("sys").argv)))
