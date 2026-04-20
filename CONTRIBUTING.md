# Contributing

## Project Overview

openpi-flash is a real-time inference engine that serves OpenPI models over QUIC and WebSocket transports with torch.compile acceleration. It has two optional component slots — an **action** slot (PyTorch or JAX policy) and a **planner** slot (JAX subtask generator for pi0.5 two-phase inference) — and picks one of three modes at startup based on which slots are configured: `action_only`, `planner_only`, or `combined`. See the [README](README.md#subtask-generation-planner) for the planner's config / endpoints / modes. It supports deployment on AWS EC2 (Docker Compose) and Modal (serverless).

For terraform use `tofu` not `terraform` CLI for consistency.

## Running Locally

```bash
# Local development
uv run python main.py serve --config config.json

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
# Action endpoint (default slot — ports 8000/5555)
uv run python main.py test ws ws://localhost:8000
uv run python main.py test quic localhost

# Modal variants
uv run python main.py test modal-tunnel
uv run python main.py test modal-quic                 # discovery via Modal Dict
```

If the planner slot is loaded, the planner endpoint is on its own transport triple (WebSocket 8002 / QUIC 5556). It doesn't have a `main.py test` helper yet; smoke-test it with a direct Python client:

```bash
uv run python -c "
from openpi_client import websocket_client_policy as wcp
import numpy as np
client = wcp.WebsocketClientPolicy(host='localhost', port=8002)
print(client.infer({'prompt': 'pick up the red cup',
                    'images': {'cam_high': np.zeros((224,224,3), dtype=np.uint8)}}))"

# Admin endpoint (only started with the planner slot; port 8001, bind to 127.0.0.1)
curl http://localhost:8001/health
curl http://localhost:8001/config
```

## Code Quality

Run everything below in one shot with `./scripts/check.sh` (optional tools are skipped with a warning if missing).

```bash
uv run ruff check --fix  
uv run ruff format       
uv run ty check          

# Rust transport layer
cd flash-transport && cargo fmt
cd flash-transport && cargo clippy --all-targets --all-features

lychee -v .              # Markdown link checking
hadolint Dockerfile      # Docker linting (brew install hadolint)

# OpenTofu (infra/)
cd infra && tofu fmt 
cd infra && tofu validate 
cd infra && tflint
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
