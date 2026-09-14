"""Validate complete AB/BA records before computing a speedup."""

from __future__ import annotations

import itertools
import math
import statistics


def validate_pairs(records: list[dict], variants: tuple[str, str]) -> list[dict[str, float]]:
    if len(records) != 12:
        raise ValueError("campaign requires exactly twelve records in six process pairs")
    pairs = [{} for _ in range(6)]
    positions = [set() for _ in range(6)]
    for record in records:
        process, position = record["process"], record["position"]
        variant = record["variant"]
        if type(process) is not int or process not in range(6) or type(position) is not int or position not in (0, 1):
            raise ValueError("invalid process or position")
        order = variants if process % 2 == 0 else variants[::-1]
        if variant != order[position] or variant in pairs[process] or position in positions[process]:
            raise ValueError("duplicate record or incorrect AB/BA order")
        value = float(record["milliseconds"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("latency must be finite and positive")
        pairs[process][variant] = value
        positions[process].add(position)
    if any(set(pair) != set(variants) for pair in pairs):
        raise ValueError("incomplete process pair")
    return pairs


def summarize_pairs(pairs: list[dict], numerator: str, denominator: str) -> dict:
    ratios = [pair[numerator] / pair[denominator] for pair in pairs]
    logs = list(map(math.log, ratios))
    # Enumerate the six-pair process bootstrap exactly, matching the frozen estimator.
    bootstrap = sorted(math.exp(sum(logs[i] for i in indices) / 6)
                       for indices in itertools.product(range(6), repeat=6))

    def percentile(q):
        position = (len(bootstrap) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        return bootstrap[lower] + (position - lower) * (bootstrap[upper] - bootstrap[lower])

    return {
        "process_ratios": ratios,
        "process_wins": sum(value > 1 for value in ratios),
        "processes": 6,
        "paired_geomean_speedup": math.exp(statistics.mean(logs)),
        "paired_bootstrap_95ci": [percentile(0.025), percentile(0.975)],
    }
