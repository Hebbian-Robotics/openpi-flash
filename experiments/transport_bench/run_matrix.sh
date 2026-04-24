#!/usr/bin/env bash
# Run the transport benchmark across a matrix of netem profiles × transports,
# with **bidirectional** shaping (egress + ifb-redirected ingress) and a
# **sample-target** stop condition so heavy-loss cells still gather enough
# samples for stable tail percentiles.
#
# Requires:
#   - openpi-flash action slot reachable on the docker host, with ports
#     published to 0.0.0.0 ({8000,5555}) — the client container reaches the
#     server via the docker bridge gateway ("host.docker.internal" on Linux
#     with --add-host=host-gateway).
#   - benchmark.py mounted into the client container
#   - iproute2 available in the client container (installed on demand below)
#
# IMPORTANT: the client container uses bridge networking (NOT host networking)
# so that `tc qdisc add dev eth0 ...` inside the container shapes only the
# container's veth, not the host's real network interface. Using host
# networking here would disrupt SSH.
#
# Why bidi shaping: shaping only the container's egress leaves the return
# path (server→client) unshaped, which is asymmetric and biases the
# comparison for response-heavy workloads. We fold container ingress into an
# IFB device and apply the same netem profile there so both directions see
# the same delay/loss.
#
# Why no jitter: the first-pass matrix (2026-04-21) used `delay 25ms 2ms
# distribution normal`, which applies per-packet jitter that can reorder
# packets on the wire. QUIC's reorder detection triggered spurious
# retransmits on ~300-packet DROID observation messages, producing a
# misleading 940ms p50 at only +25ms delay while TCP absorbed it cleanly.
# Zero-jitter profiles isolate the transport comparison from this artifact.
# A "jitter study" profile set can be added separately if we want to
# characterize reorder sensitivity explicitly.
#
# Usage (on the EC2 host):
#   export IMAGE=438136598620.dkr.ecr.us-west-2.amazonaws.com/openpi-flash:latest
#   bash run_matrix.sh /tmp/bench/results
set -euo pipefail

OUTPUT_DIR="${1:-/tmp/bench/results}"
IMAGE="${IMAGE:?IMAGE env var must be set to the client container image}"
# Reach the server on the docker bridge gateway. host-gateway resolves to the
# host's IP on the docker bridge (typically 172.17.0.1).
HOST="${HOST:-host.docker.internal}"
TARGET_RATE_HZ="${TARGET_RATE_HZ:-20}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-3}"
# Sample-target termination — run each cell until MIN_SAMPLES successful
# calls have been collected AND MIN_DURATION_S has elapsed, capped at
# MAX_DURATION_S. See benchmark.py for details.
MIN_SAMPLES="${MIN_SAMPLES:-200}"
MIN_DURATION_S="${MIN_DURATION_S:-30}"
MAX_DURATION_S="${MAX_DURATION_S:-600}"
WS_PORT="${WS_PORT:-8000}"
QUIC_PORT="${QUIC_PORT:-5555}"

mkdir -p "$OUTPUT_DIR"

# Each entry is: profile_name|tc-netem-args (empty args => no shaping).
# Args are passed to `netem` on BOTH the container's egress (eth0) and the
# ifb device carrying its mirrored ingress, so both request and response
# experience the profile. Zero jitter — see header comment.
PROFILES=(
  "clean|"
  "delay25ms|delay 25ms"
  "delay75ms_loss0_1pct|delay 75ms loss 0.1%"
  "delay150ms_loss0_5pct|delay 150ms loss 0.5%"
)

TRANSPORTS=(
  "ws ${WS_PORT}"
  "quic ${QUIC_PORT}"
)

run_one() {
  local profile_name="$1"
  local tc_args="$2"
  local transport="$3"
  local port="$4"
  local output_path="${OUTPUT_DIR}/${transport}_${profile_name}.json"

  echo "=== ${transport} @ ${profile_name} → ${output_path} ==="

  # Build the tc setup inside-container. For non-clean profiles we:
  #   1. Apply netem on eth0 root → shapes container egress (client → server)
  #   2. Create ifb0 in the container's netns
  #   3. Add an ingress qdisc on eth0 and mirror ingress packets to ifb0
  #   4. Apply the same netem on ifb0 root → shapes container ingress (server → client)
  # Net result: both directions see the same delay/loss profile.
  local tc_setup=""
  if [[ -n "${tc_args}" ]]; then
    tc_setup="tc qdisc add dev eth0 root netem ${tc_args};
      ip link add ifb0 type ifb;
      ip link set ifb0 up;
      tc qdisc add dev eth0 handle ffff: ingress;
      tc filter add dev eth0 parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev ifb0;
      tc qdisc add dev ifb0 root netem ${tc_args};
      echo [tc] eth0:; tc qdisc show dev eth0;
      echo [tc] ifb0:; tc qdisc show dev ifb0;"
  fi

  docker run --rm --cap-add=NET_ADMIN \
    --add-host=host.docker.internal:host-gateway \
    -v "$(pwd)/benchmark.py:/tmp/benchmark.py:ro" \
    -v "${OUTPUT_DIR}:/out" \
    "${IMAGE}" \
    bash -lc "set -euo pipefail;
      if ! command -v tc >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y --no-install-recommends iproute2 >/dev/null
      fi;
      ${tc_setup}
      python /tmp/benchmark.py \
        --transport ${transport} \
        --host ${HOST} \
        --port ${port} \
        --target-rate-hz ${TARGET_RATE_HZ} \
        --min-samples ${MIN_SAMPLES} \
        --min-duration-s ${MIN_DURATION_S} \
        --max-duration-s ${MAX_DURATION_S} \
        --warmup-iterations ${WARMUP_ITERATIONS} \
        --tag ${transport}_${profile_name} \
        --output /out/${transport}_${profile_name}.json"
}

for profile_entry in "${PROFILES[@]}"; do
  profile_name="${profile_entry%%|*}"
  tc_args="${profile_entry#*|}"
  for transport_entry in "${TRANSPORTS[@]}"; do
    read -r transport port <<<"${transport_entry}"
    run_one "${profile_name}" "${tc_args}" "${transport}" "${port}"
  done
done

echo "All runs complete. Results in ${OUTPUT_DIR}:"
ls -l "${OUTPUT_DIR}"
