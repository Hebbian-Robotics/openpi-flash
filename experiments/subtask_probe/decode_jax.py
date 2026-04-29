#!/usr/bin/env python3
"""JAX-based subtask generation probe for pi0.5.

Uses the proven JAX code path (adapted from LisavilaLee/openpi_with_subtask
and BrunoFANG1's implementation referenced in Physical-Intelligence/openpi#701).

The approach:
  1. Load the pi0.5 JAX model from the Orbax checkpoint
  2. Build a proper observation with images + prompt
  3. embed_prefix() -> forward through PaliGemma -> KV cache
  4. decode_to_logits (Embedder.decode = dot(h, embed_table.T))
  5. Autoregressive token generation from last prefix position
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

OPENPI_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(OPENPI_SRC))

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import Pi0, make_attn_mask  # noqa: E402
from openpi.models.pi0_config import Pi0Config  # noqa: E402
from openpi.models.tokenizer import PaligemmaTokenizer  # noqa: E402


def _add_subtask_tokenizer_methods(tok: PaligemmaTokenizer) -> None:
    """Monkey-patch the subtask tokenization methods onto the stock tokenizer."""
    import string

    def tokenize_high_level_prefix(
        self: PaligemmaTokenizer, high_prompt: str
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned = high_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        if cleaned and cleaned[-1] in string.punctuation:
            cleaned = cleaned[:-1]
        prefix_str = f"Task: {cleaned}. Subtask: "
        tokens = self._tokenizer.encode(prefix_str, add_bos=True)  # ty: ignore[unresolved-attribute]
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            pad_len = self._max_len - tokens_len
            mask = [True] * tokens_len + [False] * pad_len
            tokens = tokens + [0] * pad_len
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=np.bool_)

    def detokenize(self: PaligemmaTokenizer, tokens: np.ndarray) -> str:
        valid = [int(t) for t in tokens if t != 0 and t != 1]
        return self._tokenizer.decode(valid)  # ty: ignore[unresolved-attribute]

    # Bind methods
    import types

    tok.tokenize_high_level_prefix = types.MethodType(tokenize_high_level_prefix, tok)  # ty: ignore[unresolved-attribute]
    tok.detokenize = types.MethodType(detokenize, tok)  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_pi05_jax(checkpoint_dir: str) -> Pi0:
    """Load the pi0.5 JAX model from an Orbax checkpoint."""
    config = Pi0Config(pi05=True)
    rng = jax.random.key(0)
    model = config.create(rng)

    print(f"[load] Restoring params from {checkpoint_dir}/params ...")
    params = _model.restore_params(f"{checkpoint_dir}/params", dtype=jnp.bfloat16)
    import flax.nnx as nnx

    nnx.update(model, nnx.State(params))
    model.eval()
    print("[load] Model loaded and in eval mode.")
    return model


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------


def make_observation(
    prompt: str,
    tokenizer: PaligemmaTokenizer,
    action_dim: int = 32,
    use_random_images: bool = True,
) -> tuple[_model.Observation, int]:
    """Build an observation for the probe.

    Uses the SUBTASK prompt format: "Task: {task}. Subtask: " (no state, no Action:).
    This matches what the model expects for subtask generation mode.

    Returns (observation, num_real_tokens).
    """
    # Tokenize with subtask prefix format (NOT the action format)
    # "Task: {task}. Subtask: " -- no state, no Action:
    tokens, mask = tokenizer.tokenize_high_level_prefix(prompt)  # ty: ignore[unresolved-attribute]
    num_real_tokens = int(mask.sum())
    state = np.zeros(action_dim, dtype=np.float32)

    # Images: random noise or zeros
    def make_img() -> np.ndarray:
        if use_random_images:
            return np.random.default_rng(42).random((224, 224, 3)).astype(np.float32) * 2 - 1
        return np.zeros((224, 224, 3), dtype=np.float32)

    # Build observation (batch dim added, convert to JAX arrays)
    obs = _model.Observation(
        images={
            "base_0_rgb": jnp.array(make_img()[None]),
            "left_wrist_0_rgb": jnp.array(make_img()[None]),
            "right_wrist_0_rgb": jnp.array(make_img()[None]),
        },
        image_masks={
            "base_0_rgb": jnp.array([True]),
            "left_wrist_0_rgb": jnp.array([True]),
            "right_wrist_0_rgb": jnp.array([True]),
        },
        state=jnp.array(state[None]),
        tokenized_prompt=jnp.array(tokens[None]),
        tokenized_prompt_mask=jnp.array(mask[None]),
    )

    return obs, num_real_tokens


# ---------------------------------------------------------------------------
# Subtask generation (JAX, adapted from LisavilaLee's generate_subtask)
# ---------------------------------------------------------------------------


def generate_subtask_text(
    model: Pi0,
    observation: _model.Observation,
    max_decoding_steps: int = 50,
    temperature: float = 0.0,
) -> dict:
    """Generate subtask text autoregressively from the pi0.5 model.

    This follows the proven JAX approach:
      1. embed_prefix -> PaliGemma forward -> KV cache
      2. Embedder.decode (dot with embedding table transpose) -> logits
      3. Greedy/sampled token generation in a loop
    """
    observation = _model.preprocess_observation(None, observation, train=False)

    # Step 1: Embed prefix (images + language tokens)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    B, prefix_S, _ = prefix_tokens.shape

    # Step 2: Forward through PaliGemma to get KV cache + prefix output
    # No mask padding -- KV cache size matches prefix length.
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1  # ty: ignore[invalid-argument-type]

    (prefix_out, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions,
        adarms_cond=[None, None],
    )

    # Step 3: Find the last VALID token position (LisavilaLee's fix)
    seq_indices = jnp.arange(prefix_S)[None, :]  # [1, S]
    last_pos = jnp.max(jnp.where(prefix_mask, seq_indices, -1), axis=1).astype(jnp.int32)  # ty: ignore[no-matching-overload]

    last_hidden = prefix_out[jnp.arange(B), last_pos, :]  # [B, D]

    # Step 4: Project to vocab logits via Embedder.decode
    # Embedder.decode = dot(x, embedding_table.T)
    embed_table = model.PaliGemma.llm.embedder["input_embedding"].value  # ty: ignore[unresolved-attribute]
    logits = jnp.dot(last_hidden, embed_table.T)  # [B, vocab_size]

    # Collect initial top predictions
    probs = jax.nn.softmax(logits, axis=-1)
    top_k_probs, top_k_indices = jax.lax.top_k(probs[0], 10)
    initial_predictions = list(zip(top_k_indices.tolist(), top_k_probs.tolist(), strict=True))

    # Step 5: Autoregressive generation loop (eager, matches LisavilaLee's generate_subtask)
    EOS_TOKEN = 1
    num_real = int(jnp.sum(prefix_mask, axis=-1)[0])  # ty: ignore[invalid-argument-type]
    next_pos = jnp.array([num_real], dtype=jnp.int32)  # [B]
    generated_token_ids = []

    current_logits = logits[
        None, :, :
    ]  # reshape to [B, 1, V] to match their pattern... actually [B, V]
    current_cache = kv_cache

    for step_idx in range(max_decoding_steps):
        # Greedy decode
        if temperature > 0:
            token_id = int(
                jax.random.categorical(jax.random.key(step_idx), current_logits[0] / temperature)
            )
        else:
            token_id = int(jnp.argmax(current_logits[0]))

        generated_token_ids.append(token_id)
        if token_id == EOS_TOKEN:
            break

        # Embed the token
        token_jax = jnp.array([[token_id]], dtype=jnp.int32)
        token_embedding = model.PaliGemma.llm(token_jax, method="embed")  # [B, 1, D]

        # Attention mask: [B, 1, prefix_S + gen_count]
        # New token attends to all prefix tokens + all previously generated tokens + itself.
        gen_count = step_idx + 1
        gen_mask = jnp.ones((B, gen_count), dtype=jnp.bool_)
        full_mask = jnp.concatenate([prefix_mask, gen_mask], axis=1)  # ty: ignore[invalid-argument-type]
        attn_mask = full_mask[:, None, :]  # [B, 1, prefix_S + gen_count]

        new_positions = next_pos[:, None]  # [B, 1]

        # Forward with KV cache
        (new_out, _), current_cache = model.PaliGemma.llm(
            [token_embedding, None],
            mask=attn_mask,
            positions=new_positions,
            kv_cache=current_cache,
        )

        # Project to logits via embedding table transpose
        new_hidden = new_out[:, -1, :]  # [B, D]
        current_logits = jnp.dot(new_hidden, embed_table.T)  # [B, vocab_size]
        next_pos = next_pos + 1

    return {
        "output_tokens": generated_token_ids,
        "num_steps": len(generated_token_ids),
        "initial_predictions": initial_predictions,
        "last_valid_position": int(last_pos[0]),
        "prefix_length": prefix_S,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="JAX subtask generation probe for pi0.5")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_base"),
    )
    parser.add_argument("--random_images", action="store_true", default=True)
    parser.add_argument("--zero_images", action="store_true")
    args = parser.parse_args()

    use_random = not args.zero_images

    tokenizer = PaligemmaTokenizer(max_len=200)
    _add_subtask_tokenizer_methods(tokenizer)
    print("Tokenizer loaded (with subtask methods)")

    model = load_pi05_jax(args.checkpoint_dir)

    test_prompts = [
        "pick up the red cup and place it on the shelf",
        "fold the towel neatly",
        "open the drawer and put the block inside",
        "stack the blue block on top of the red block",
        "wipe the table with the sponge",
    ]

    for prompt in test_prompts:
        print(f"\n{'=' * 80}")
        print(f'  Prompt: "{prompt}"')
        print(f"{'=' * 80}")

        obs, num_real = make_observation(prompt, tokenizer, use_random_images=use_random)
        print(f"  Real tokens: {num_real}/200, images: {'random' if use_random else 'zeros'}")

        result = generate_subtask_text(model, obs, max_decoding_steps=50, temperature=0.0)

        # Decode tokens
        token_ids = result["output_tokens"]
        # Remove EOS if present
        if 1 in token_ids:
            token_ids = token_ids[: token_ids.index(1)]

        generated_text = tokenizer._tokenizer.decode(token_ids)  # ty: ignore[unresolved-attribute]

        print(
            f"  Prefix length: {result['prefix_length']}, last valid pos: {result['last_valid_position']}"
        )
        print(f"  Steps generated: {result['num_steps']}")

        # Initial predictions
        print("  Top-10 next-token predictions from last prefix position:")
        for token_id, prob in result["initial_predictions"]:
            token_str = tokenizer._tokenizer.decode([token_id])  # ty: ignore[unresolved-attribute]
            print(f'    [{token_id:6d}] "{token_str}" (prob={prob:.4f})')

        print(f'  Generated text: "{generated_text}"')
        print(f"  Raw tokens: {token_ids[:30]}")

    print(f"\n{'=' * 80}")
    print("  DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
