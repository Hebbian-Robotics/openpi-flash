#!/usr/bin/env bash
# Run every check listed in CONTRIBUTING.md §Code Quality.
# Continues past failures; prints a summary and exits non-zero if any failed.
# Optional tools (lychee, hadolint, tflint, tofu) are skipped with a warning
# when not installed.

set -u -o pipefail

cd "$(dirname "$0")/.."

PASSED=()
FAILED=()
SKIPPED=()

run_step() {
  local name="$1"; shift
  local cwd="$1"; shift
  echo ""
  echo "==> [$name] (in $cwd) $*"
  if ( cd "$cwd" && "$@" ); then
    PASSED+=("$name")
  else
    FAILED+=("$name")
  fi
}

run_optional() {
  local name="$1"; shift
  local cwd="$1"; shift
  local bin="$1"; shift
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo ""
    echo "==> [$name] SKIP: $bin not installed"
    SKIPPED+=("$name ($bin missing)")
    return
  fi
  run_step "$name" "$cwd" "$@"
}

run_step     "ruff-check"    "."             uv run ruff check --fix
run_step     "ruff-format"   "."             uv run ruff format
run_step     "ty-check"      "."             uv run ty check
run_step     "cargo-fmt"     "flash-transport"  cargo fmt
run_step     "cargo-clippy"  "flash-transport"  cargo clippy --all-targets --all-features
run_optional "lychee"        "."             lychee   lychee -v .
run_optional "hadolint"      "."             hadolint hadolint Dockerfile
run_optional "tofu-fmt"      "infra"         tofu     tofu fmt
run_optional "tofu-validate" "infra"         tofu     tofu validate
run_optional "tflint"        "infra"         tflint   tflint

echo ""
echo "=== Summary ==="
printf 'PASSED  (%d): %s\n' "${#PASSED[@]}"  "${PASSED[*]:-}"
printf 'SKIPPED (%d): %s\n' "${#SKIPPED[@]}" "${SKIPPED[*]:-}"
printf 'FAILED  (%d): %s\n' "${#FAILED[@]}"  "${FAILED[*]:-}"

[[ ${#FAILED[@]} -eq 0 ]]
