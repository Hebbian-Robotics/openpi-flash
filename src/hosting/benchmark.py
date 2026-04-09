"""Shared benchmark utilities for inference test scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class InferablePolicy(Protocol):
    """Protocol for policies that support infer() and get_server_metadata()."""

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...
    def get_server_metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class InferenceTimingSample:
    """Timing data from a single inference call."""

    client_round_trip_ms: float
    server_infer_ms: float
    policy_forward_ms: float

    @property
    def network_overhead_ms(self) -> float:
        return self.client_round_trip_ms - self.server_infer_ms

    @property
    def server_overhead_ms(self) -> float:
        return self.server_infer_ms - self.policy_forward_ms


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results from multiple inference calls."""

    samples: list[InferenceTimingSample] = field(default_factory=list)
    action_shape: tuple[int, ...] | None = None

    @property
    def num_iterations(self) -> int:
        return len(self.samples)

    def _stat_line(self, label: str, values: list[float]) -> str:
        mean = sum(values) / len(values)
        return f"  {label:<20s} mean={mean:.0f}ms  min={min(values):.0f}ms  max={max(values):.0f}ms"

    def print_summary(self) -> None:
        if not self.samples:
            print("No samples collected.")
            return

        print(f"\nAction shape: {self.action_shape}")
        print(f"\nBenchmark summary ({self.num_iterations} iterations):")
        print(self._stat_line("Client round trip:", [s.client_round_trip_ms for s in self.samples]))
        print(self._stat_line("Server infer:", [s.server_infer_ms for s in self.samples]))
        print(self._stat_line("Policy forward:", [s.policy_forward_ms for s in self.samples]))

        mean_network = sum(s.network_overhead_ms for s in self.samples) / self.num_iterations
        mean_server = sum(s.server_overhead_ms for s in self.samples) / self.num_iterations
        print(f"  {'Network overhead:':<20s} mean={mean_network:.0f}ms")
        print(f"  {'Server overhead:':<20s} mean={mean_server:.0f}ms")


def run_benchmark(
    policy: InferablePolicy,
    make_observation: Any,
    *,
    num_warmup: int = 1,
    num_iterations: int = 5,
) -> BenchmarkResult:
    """Run warmup + timed inference iterations and return structured results."""
    # Warmup.
    print(f"\nWarmup ({num_warmup} inference) ...")
    warmup_start = time.monotonic()
    for _ in range(num_warmup):
        policy.infer(make_observation())
    print(f"Warmup done in {1000 * (time.monotonic() - warmup_start):.0f}ms")

    # Benchmark.
    print(f"\nRunning {num_iterations} inferences ...")
    result = BenchmarkResult()
    action: dict[str, Any] = {}

    for i in range(num_iterations):
        start = time.monotonic()
        action = policy.infer(make_observation())
        client_ms = 1000 * (time.monotonic() - start)

        sample = InferenceTimingSample(
            client_round_trip_ms=client_ms,
            server_infer_ms=action.get("server_timing", {}).get("infer_ms", 0),
            policy_forward_ms=action.get("policy_timing", {}).get("infer_ms", 0),
        )
        result.samples.append(sample)

        print(
            f"  [{i + 1}/{num_iterations}]"
            f" client={sample.client_round_trip_ms:.0f}ms"
            f"  server={sample.server_infer_ms:.0f}ms"
            f"  policy={sample.policy_forward_ms:.0f}ms"
        )

    if action:
        result.action_shape = tuple(action["actions"].shape)
    return result
