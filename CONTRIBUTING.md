# Contributing

## Project Overview

openpi-hosting is a hosted inference service that wraps OpenPI policy models in a WebSocket server. It supports deployment on AWS EC2 (Docker Compose) and Modal (serverless).

## Running Locally

```bash
# Local development
INFERENCE_CONFIG_PATH=config.json uv run python -m hosting.serve

# Docker (requires config.json)
docker compose up --build

# Modal dev (hot-reload, temporary URL)
uv run modal serve modal_app.py

# Modal production (persistent URL, auto-scales)
uv run modal deploy modal_app.py

# Modal tunnel (direct TCP, lower latency)
uv run modal run modal_tunnel_app.py

# Modal QUIC portal (experimental, lowest latency via UDP/QUIC + NAT traversal)
uv run modal run modal_quic_app.py
```

## Testing

```bash
# Smoke test against a running instance
uv run python test_modal.py wss://your-modal-url

# Smoke test tunnel variant
uv run python test_modal_tunnel.py

# Smoke test QUIC portal variant (no URL needed — discovery via Modal Dict)
uv run python test_modal_quic.py
```

## Code Quality

```bash
uv run ruff check --fix  # Linting with auto-fix
uv run ruff format       # Code formatting
uv run ty check          # Type checking
```

## Code Style & Philosophy

### Typing & Pattern Matching

- Prefer **explicit types** over raw dicts -- make invalid states unrepresentable where practical
- Prefer **typed variants over string literals** when the set of valid values is known
- Use **exhaustive pattern matching** (`match`) so the type checker can verify all cases are handled
- Structure types to enable exhaustive matching when handling variants
- Prefer **shared internal functions over factory patterns** when extracting common logic -- keep each export explicitly defined for better IDE navigation and readability

### Logging

- Use `print()` instead of `logging.getLogger()` for code that runs inside Modal containers — Modal does not reliably capture Python logger output.

### Self-Documenting Code

- **Verbose naming**: Variable and function naming should read like documentation
- **Strategic comments**: Only for non-obvious logic or architectural decisions; avoid restating what code shows
