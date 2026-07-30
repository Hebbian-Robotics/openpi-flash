"""Same-host transport benchmark for flash-transport vs. openpi WebSocket.

Runs serial inference calls at a target pacing against a running server and
records per-call latency. Designed to be invoked from inside a Docker
container that already has the openpi-flash wheel + its dependencies, so the
client dependency on ``openpi_client`` and ``hosting.flash_transport_policy``
is satisfied without a separate install step.

See ``run_matrix.sh`` for the wrapper that runs the full profile matrix with
bidirectional netem shaping (ifb-mirrored ingress). For a one-shot invocation
the shape is:

    python benchmark.py --transport quic --host 127.0.0.1 --port 5555 \\
        --target-rate-hz 20 --min-samples 200 --min-duration-s 30 \\
        --max-duration-s 600 --output /tmp/quic-clean.json

The server is assumed to be running locally (action slot on 8000/TCP + 5555/UDP).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

# Reach into the openpi-flash install tree. These import paths assume we are
# running inside the openpi-flash container image (or a local dev checkout
# with ``uv sync``).
from hosting.warmup import make_droid_observation  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402

Transport = Literal["ws", "quic"]


class InferablePolicy(Protocol):
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]: ...
    def get_server_metadata(self) -> dict[str, Any]: ...


@dataclass
class CallSample:
    """Per-call timing row."""

    iteration_index: int
    send_unix_s: float
    client_round_trip_ms: float
    server_infer_ms: float
    policy_forward_ms: float
    failed: bool = False
    error_message: str | None = None

    @property
    def network_overhead_ms(self) -> float:
        if self.failed:
            return 0.0
        return max(0.0, self.client_round_trip_ms - self.server_infer_ms)


@dataclass
class RunSummary:
    transport: Transport
    host: str
    port: int
    target_rate_hz: float
    min_samples: int
    min_duration_s: float
    max_duration_s: float
    warmup_iterations: int
    total_iterations: int
    successful_iterations: int
    failure_count: int
    wall_clock_duration_s: float
    stop_reason: str
    effective_rate_hz: float
    client_round_trip_ms_p50: float
    client_round_trip_ms_p95: float
    client_round_trip_ms_p99: float
    client_round_trip_ms_mean: float
    client_round_trip_ms_stddev: float
    client_round_trip_ms_max: float
    server_infer_ms_mean: float
    network_overhead_ms_mean: float
    network_overhead_ms_p95: float
    network_overhead_ms_p99: float
    samples: list[dict[str, Any]] = field(default_factory=list)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fractional = rank - lower_index
    return sorted_values[lower_index] + fractional * (
        sorted_values[upper_index] - sorted_values[lower_index]
    )


def build_policy(transport: Transport, host: str, port: int) -> InferablePolicy:
    if transport == "ws":
        return _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    if transport == "quic":
        from hosting.flash_transport_policy import FlashTransportPolicy

        return FlashTransportPolicy(host=host, port=port)
    raise ValueError(f"Unknown transport: {transport}")


def run_benchmark(
    *,
    transport: Transport,
    host: str,
    port: int,
    target_rate_hz: float,
    min_samples: int,
    min_duration_s: float,
    max_duration_s: float,
    warmup_iterations: int,
    prompt: str,
    seed: int,
) -> RunSummary:
    rng = np.random.default_rng(seed)

    def make_observation() -> dict[str, Any]:
        # Fresh random frames per call so server-side preprocessing can't
        # short-circuit.
        observation = make_droid_observation(prompt=prompt)
        observation["observation/joint_position"] = rng.random(7)
        return observation

    print(f"Building {transport} policy → {host}:{port}", flush=True)
    policy = build_policy(transport, host, port)
    try:
        metadata = policy.get_server_metadata()
        print(f"Server metadata: {metadata}", flush=True)

        print(f"Warmup: {warmup_iterations} iteration(s) ...", flush=True)
        warmup_start = time.monotonic()
        for _ in range(warmup_iterations):
            policy.infer(make_observation())
        print(
            f"Warmup done in {1000 * (time.monotonic() - warmup_start):.0f}ms",
            flush=True,
        )

        inter_call_interval_s = 1.0 / target_rate_hz if target_rate_hz > 0 else 0.0
        wall_clock_start = time.monotonic()
        min_deadline = wall_clock_start + min_duration_s
        max_deadline = wall_clock_start + max_duration_s

        samples: list[CallSample] = []
        iteration_index = 0
        next_send_monotonic = wall_clock_start
        stop_reason = "max_duration_s_reached"

        while True:
            now_for_stop_check = time.monotonic()
            successful_so_far = sum(1 for sample in samples if not sample.failed)
            # Stop when we have enough samples AND the minimum runtime has
            # elapsed (so target-rate pacing gets enough wall-clock time to
            # stabilize). Hard cap on max_duration_s regardless.
            if now_for_stop_check >= max_deadline:
                stop_reason = "max_duration_s_reached"
                break
            if successful_so_far >= min_samples and now_for_stop_check >= min_deadline:
                stop_reason = "min_samples_reached"
                break
            # Sleep until next scheduled send (serial, paced).
            now_monotonic = time.monotonic()
            if now_monotonic < next_send_monotonic:
                time.sleep(next_send_monotonic - now_monotonic)

            send_unix = time.time()
            call_start_monotonic = time.monotonic()
            try:
                action = policy.infer(make_observation())
                client_round_trip_ms = 1000 * (time.monotonic() - call_start_monotonic)
                server_infer_ms = float(
                    action.get("server_timing", {}).get("infer_ms", 0.0)
                )
                policy_forward_ms = float(
                    action.get("policy_timing", {}).get("infer_ms", 0.0)
                )
                samples.append(
                    CallSample(
                        iteration_index=iteration_index,
                        send_unix_s=send_unix,
                        client_round_trip_ms=client_round_trip_ms,
                        server_infer_ms=server_infer_ms,
                        policy_forward_ms=policy_forward_ms,
                    )
                )
            except Exception as error:  # noqa: BLE001
                client_round_trip_ms = 1000 * (time.monotonic() - call_start_monotonic)
                samples.append(
                    CallSample(
                        iteration_index=iteration_index,
                        send_unix_s=send_unix,
                        client_round_trip_ms=client_round_trip_ms,
                        server_infer_ms=0.0,
                        policy_forward_ms=0.0,
                        failed=True,
                        error_message=f"{type(error).__name__}: {error}",
                    )
                )

            iteration_index += 1
            next_send_monotonic += inter_call_interval_s
            # If the call itself ran over the interval, skip ahead so we
            # don't queue up a burst of back-to-back sends.
            if time.monotonic() > next_send_monotonic:
                next_send_monotonic = time.monotonic()

        wall_clock_duration_s = time.monotonic() - wall_clock_start
    finally:
        close_fn = getattr(policy, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # noqa: BLE001
                pass

    successful_samples = [s for s in samples if not s.failed]
    failure_count = len(samples) - len(successful_samples)
    client_round_trip_values = [s.client_round_trip_ms for s in successful_samples]
    server_infer_values = [s.server_infer_ms for s in successful_samples]
    network_overhead_values = [s.network_overhead_ms for s in successful_samples]

    summary = RunSummary(
        transport=transport,
        host=host,
        port=port,
        target_rate_hz=target_rate_hz,
        min_samples=min_samples,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        warmup_iterations=warmup_iterations,
        total_iterations=len(samples),
        successful_iterations=len(successful_samples),
        failure_count=failure_count,
        wall_clock_duration_s=wall_clock_duration_s,
        stop_reason=stop_reason,
        effective_rate_hz=(
            len(successful_samples) / wall_clock_duration_s if wall_clock_duration_s else 0.0
        ),
        client_round_trip_ms_p50=_percentile(client_round_trip_values, 50),
        client_round_trip_ms_p95=_percentile(client_round_trip_values, 95),
        client_round_trip_ms_p99=_percentile(client_round_trip_values, 99),
        client_round_trip_ms_mean=(
            statistics.fmean(client_round_trip_values) if client_round_trip_values else 0.0
        ),
        client_round_trip_ms_stddev=(
            statistics.pstdev(client_round_trip_values)
            if len(client_round_trip_values) > 1
            else 0.0
        ),
        client_round_trip_ms_max=max(client_round_trip_values) if client_round_trip_values else 0.0,
        server_infer_ms_mean=(
            statistics.fmean(server_infer_values) if server_infer_values else 0.0
        ),
        network_overhead_ms_mean=(
            statistics.fmean(network_overhead_values) if network_overhead_values else 0.0
        ),
        network_overhead_ms_p95=_percentile(network_overhead_values, 95),
        network_overhead_ms_p99=_percentile(network_overhead_values, 99),
        samples=[asdict(sample) for sample in samples],
    )

    return summary


def print_summary_table(summary: RunSummary) -> None:
    print("\n--- Benchmark summary ---")
    print(
        f"transport={summary.transport} host={summary.host}:{summary.port} "
        f"target_rate={summary.target_rate_hz:.1f}Hz "
        f"min_samples={summary.min_samples} "
        f"min_duration={summary.min_duration_s:.0f}s "
        f"max_duration={summary.max_duration_s:.0f}s "
        f"stop={summary.stop_reason} "
        f"wall_clock={summary.wall_clock_duration_s:.1f}s"
    )
    print(
        f"iterations: total={summary.total_iterations} "
        f"ok={summary.successful_iterations} fail={summary.failure_count} "
        f"effective_rate={summary.effective_rate_hz:.2f}Hz"
    )
    print("Client round-trip ms:")
    print(f"  mean={summary.client_round_trip_ms_mean:.1f}  stddev={summary.client_round_trip_ms_stddev:.1f}")
    print(
        f"  p50={summary.client_round_trip_ms_p50:.1f} "
        f"p95={summary.client_round_trip_ms_p95:.1f} "
        f"p99={summary.client_round_trip_ms_p99:.1f} "
        f"max={summary.client_round_trip_ms_max:.1f}"
    )
    print(f"Server infer ms mean: {summary.server_infer_ms_mean:.1f}")
    print(
        f"Network overhead ms: mean={summary.network_overhead_ms_mean:.1f} "
        f"p95={summary.network_overhead_ms_p95:.1f} "
        f"p99={summary.network_overhead_ms_p99:.1f}"
    )


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transport benchmark harness")
    parser.add_argument("--transport", choices=["ws", "quic"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Stop once this many successful calls have been collected (and --min-duration-s has elapsed).",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=30.0,
        help="Minimum wall-clock run time before early termination is allowed.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=600.0,
        help="Hard wall-clock cap. Run terminates at this deadline even if min_samples is not reached.",
    )
    parser.add_argument("--target-rate-hz", type=float, default=20.0)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--prompt", default="pick up the red cup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write raw samples + summary JSON to this path.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional free-form label written to the output JSON (e.g. 'quic-50ms-1pct').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    summary = run_benchmark(
        transport=args.transport,
        host=args.host,
        port=args.port,
        target_rate_hz=args.target_rate_hz,
        min_samples=args.min_samples,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        warmup_iterations=args.warmup_iterations,
        prompt=args.prompt,
        seed=args.seed,
    )
    print_summary_table(summary)
    if args.output is not None:
        payload = asdict(summary)
        if args.tag is not None:
            payload["tag"] = args.tag
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
