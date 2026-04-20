#!/usr/bin/env python3
"""Test running JAX subtask generation + PyTorch action generation simultaneously.

Both models loaded on the same L40S GPU at the same time.
Requires: XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 (or similar) to prevent JAX from
grabbing all VRAM.

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python experiments/subtask_probe/dual_runtime_test.py
"""

from __future__ import annotations

import string
import sys
import time
from pathlib import Path

import numpy as np

OPENPI_SRC = Path(__file__).resolve().parents[2] / "src"
HOSTING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HOSTING_ROOT))
sys.path.insert(0, str(OPENPI_SRC))


def check_jax_memory_fraction() -> None:
    """Verify JAX memory fraction is set before importing JAX."""
    import os

    fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if fraction is None and preallocate != "false":
        print(
            "ERROR: Set XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 (or XLA_PYTHON_CLIENT_PREALLOCATE=false)"
        )
        print("       Without this, JAX will grab all GPU memory and PyTorch won't load.")
        sys.exit(1)
    print(f"JAX memory config: MEM_FRACTION={fraction}, PREALLOCATE={preallocate}")


check_jax_memory_fraction()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi.shared.download as download  # noqa: E402
import safetensors.torch  # noqa: E402
import sentencepiece  # noqa: E402
import torch  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import Pi0, make_attn_mask  # noqa: E402
from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402


def load_tokenizer() -> sentencepiece.SentencePieceProcessor:
    path = download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
    )
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())  # ty: ignore[unknown-argument]


def report_gpu_memory(label: str) -> None:
    """Print GPU memory usage from both JAX and PyTorch perspectives."""
    # PyTorch view
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        print(
            f"  [{label}] PyTorch: {allocated_mb:.0f} MB allocated, {reserved_mb:.0f} MB reserved"
        )

    # JAX view
    try:
        for device in jax.devices():
            stats = device.memory_stats()
            if stats:
                used_mb = stats.get("bytes_in_use", 0) / 1024 / 1024
                limit_mb = stats.get("bytes_limit", 0) / 1024 / 1024
                print(
                    f"  [{label}] JAX ({device}): {used_mb:.0f} MB used / {limit_mb:.0f} MB limit"
                )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# JAX subtask generation
# ---------------------------------------------------------------------------


def generate_subtask_jax(
    jax_model: Pi0,
    tokenizer: sentencepiece.SentencePieceProcessor,
    task_prompt: str,
    max_tokens: int = 20,
) -> tuple[str, float]:
    """Generate subtask text using JAX. Returns (subtask_text, elapsed_seconds)."""
    cleaned = task_prompt.lower().strip().replace("_", " ").replace("\n", " ")
    if cleaned and cleaned[-1] in string.punctuation:
        cleaned = cleaned[:-1]
    prefix_str = f"Task: {cleaned}. Subtask: "
    tokens = tokenizer.encode(prefix_str, add_bos=True)  # ty: ignore[unresolved-attribute]
    num_real = len(tokens)
    max_len = 200
    if num_real < max_len:
        mask = [True] * num_real + [False] * (max_len - num_real)
        tokens = tokens + [0] * (max_len - num_real)
    else:
        tokens = tokens[:max_len]
        mask = [True] * max_len
        num_real = max_len

    tokens_np = np.asarray(tokens, dtype=np.int32)
    mask_np = np.asarray(mask, dtype=np.bool_)

    action_dim = 32
    zero_img = np.zeros((224, 224, 3), dtype=np.float32)
    obs = _model.Observation(
        images={
            "base_0_rgb": jnp.array(zero_img[None]),
            "left_wrist_0_rgb": jnp.array(zero_img[None]),
            "right_wrist_0_rgb": jnp.array(zero_img[None]),
        },
        image_masks={
            "base_0_rgb": jnp.array([True]),
            "left_wrist_0_rgb": jnp.array([True]),
            "right_wrist_0_rgb": jnp.array([True]),
        },
        state=jnp.array(np.zeros(action_dim, dtype=np.float32)[None]),
        tokenized_prompt=jnp.array(tokens_np[None]),
        tokenized_prompt_mask=jnp.array(mask_np[None]),
    )

    start = time.monotonic()

    obs = _model.preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = jax_model.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1  # ty: ignore[invalid-argument-type]

    (prefix_out, _), kv_cache = jax_model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
        adarms_cond=[None, None],
    )

    B, prefix_S = prefix_tokens.shape[:2]
    seq_indices = jnp.arange(prefix_S)[None, :]
    last_pos = jnp.max(jnp.where(prefix_mask, seq_indices, -1), axis=1).astype(jnp.int32)  # ty: ignore[no-matching-overload]
    last_hidden = prefix_out[jnp.arange(B), last_pos, :]

    embed_table = jax_model.PaliGemma.llm.embedder["input_embedding"].value  # ty: ignore[unresolved-attribute]
    logits = jnp.dot(last_hidden, embed_table.T)

    generated_ids = []
    next_pos = jnp.array([num_real], dtype=jnp.int32)

    for step in range(max_tokens):
        token_id = int(jnp.argmax(logits[0]))
        generated_ids.append(token_id)
        if token_id in (0, 1):
            break

        token_emb = jax_model.PaliGemma.llm(jnp.array([[token_id]]), method="embed")
        gen_count = step + 1
        gen_mask = jnp.ones((1, gen_count), dtype=jnp.bool_)
        full_mask = jnp.concatenate([prefix_mask, gen_mask], axis=1)  # ty: ignore[invalid-argument-type]
        attn_mask = full_mask[:, None, :]

        (new_out, _), kv_cache = jax_model.PaliGemma.llm(
            [token_emb, None],
            mask=attn_mask,
            positions=next_pos[:, None],
            kv_cache=kv_cache,
            adarms_cond=[None, None],
        )
        logits = jnp.dot(new_out[:, -1, :], embed_table.T)
        next_pos = next_pos + 1

    elapsed = time.monotonic() - start

    if 1 in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(1)]
    return tokenizer.decode(generated_ids), elapsed  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# PyTorch action generation
# ---------------------------------------------------------------------------


class FakeObservation:
    def __init__(
        self,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
        device: str = "cuda",
        action_dim: int = 32,
    ) -> None:
        self.images = {
            "base_0_rgb": torch.zeros(1, 3, 224, 224, device=device),
            "left_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device),
            "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224, device=device),
        }
        self.image_masks = {k: torch.ones(1, dtype=torch.bool, device=device) for k in self.images}
        self.state = torch.zeros(1, action_dim, device=device)
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = None
        self.token_loss_mask = None


def make_action_observation(
    task_prompt: str,
    tokenizer: sentencepiece.SentencePieceProcessor,
    device: str = "cuda",
    action_dim: int = 32,
) -> FakeObservation:
    """Build observation with standard pi0.5 action prompt format."""
    cleaned = task_prompt.strip().replace("_", " ").replace("\n", " ")
    state = np.zeros(action_dim, dtype=np.float32)
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "

    tokens = tokenizer.encode(full_prompt, add_bos=True)  # ty: ignore[unresolved-attribute]
    num_real = len(tokens)
    max_len = 200
    if num_real < max_len:
        mask = [True] * num_real + [False] * (max_len - num_real)
        tokens = tokens + [0] * (max_len - num_real)
    else:
        tokens = tokens[:max_len]
        mask = [True] * max_len

    tok_t = torch.tensor([tokens], dtype=torch.long, device=device)
    mask_t = torch.tensor([mask], dtype=torch.bool, device=device)
    return FakeObservation(tok_t, mask_t, device, action_dim)


def generate_actions_pytorch(
    pt_model: PI0Pytorch,
    obs: FakeObservation,
    device: str = "cuda",
) -> tuple[np.ndarray, float]:
    """Run action inference. Returns (actions, elapsed_seconds)."""
    start = time.monotonic()
    with torch.no_grad():
        actions = pt_model.sample_actions(device, obs, num_steps=10)  # ty: ignore[missing-argument, invalid-argument-type]
    elapsed = time.monotonic() - start
    return actions[0].detach().cpu().numpy(), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dual runtime test: JAX + PyTorch on same GPU")
    parser.add_argument(
        "--jax_checkpoint",
        type=str,
        default=str(Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base"),
    )
    parser.add_argument(
        "--pytorch_checkpoint",
        type=str,
        default=str(Path.home() / "checkpoints/pi05_base_pytorch/model.safetensors"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    report_gpu_memory("before loading")

    # =============================================
    # Load BOTH models simultaneously
    # =============================================
    print("\n=== Loading JAX model ===")
    jax_config = Pi0Config(pi05=True)
    jax_model = jax_config.create(jax.random.key(0))
    params = _model.restore_params(f"{args.jax_checkpoint}/params", dtype=jnp.bfloat16)
    import flax.nnx as nnx

    nnx.update(jax_model, nnx.State(params))
    jax_model.eval()
    print("JAX model loaded.")
    report_gpu_memory("after JAX load")

    print("\n=== Loading PyTorch model ===")
    pt_config = Pi0Config(pi05=True, pytorch_compile_mode=None)
    pt_model = PI0Pytorch(pt_config)
    safetensors.torch.load_model(pt_model, args.pytorch_checkpoint)
    pt_model = pt_model.to(args.device).eval()
    pt_model.requires_grad_(False)
    print("PyTorch model loaded.")
    report_gpu_memory("after both loaded")

    # =============================================
    # Two-phase inference loop
    # =============================================
    test_cases = [
        "pick up the red cup and place it on the shelf",
        "fold the towel neatly",
        "open the drawer and put the block inside",
        "wipe the table with the sponge",
    ]

    print(f"\n{'=' * 70}")
    print("  TWO-PHASE INFERENCE: JAX subtask → PyTorch actions")
    print(f"{'=' * 70}")

    for task_prompt in test_cases:
        print(f'\n  Task: "{task_prompt}"')

        # Phase 1: JAX subtask generation
        subtask_text, subtask_time = generate_subtask_jax(jax_model, tokenizer, task_prompt)
        print(f'  Phase 1 (JAX subtask):  "{subtask_text}"  [{subtask_time * 1000:.0f}ms]')

        # Phase 2: PyTorch action generation (with subtask in prompt)
        hybrid_prompt = f"{task_prompt}. Subtask: {subtask_text}"
        obs = make_action_observation(hybrid_prompt, tokenizer, args.device)
        actions, action_time = generate_actions_pytorch(pt_model, obs, args.device)
        action_norm = np.linalg.norm(actions)
        print(f"  Phase 2 (PT actions):   norm={action_norm:.4f}  [{action_time * 1000:.0f}ms]")
        print(f"  Total latency:          {(subtask_time + action_time) * 1000:.0f}ms")

    # =============================================
    # Latency benchmark (5 rounds)
    # =============================================
    print(f"\n{'=' * 70}")
    print("  LATENCY BENCHMARK (5 rounds, same prompt)")
    print(f"{'=' * 70}")

    benchmark_prompt = "pick up the red cup and place it on the shelf"
    subtask_times = []
    action_times = []

    for i in range(5):
        subtask_text, st = generate_subtask_jax(jax_model, tokenizer, benchmark_prompt)
        hybrid_prompt = f"{benchmark_prompt}. Subtask: {subtask_text}"
        obs = make_action_observation(hybrid_prompt, tokenizer, args.device)
        _, at = generate_actions_pytorch(pt_model, obs, args.device)
        subtask_times.append(st)
        action_times.append(at)
        print(
            f"  Round {i + 1}: subtask={st * 1000:.0f}ms  action={at * 1000:.0f}ms  total={(st + at) * 1000:.0f}ms"
        )

    avg_subtask = np.mean(subtask_times) * 1000
    avg_action = np.mean(action_times) * 1000
    print(
        f"\n  Average: subtask={avg_subtask:.0f}ms  action={avg_action:.0f}ms  total={avg_subtask + avg_action:.0f}ms"
    )

    report_gpu_memory("end of benchmark")

    print(f"\n{'=' * 70}")
    print("  DONE — Both models coexisted on the same GPU successfully.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
