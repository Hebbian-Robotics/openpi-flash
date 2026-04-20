"""JIT-compiled JAX subtask generation for pi0.5 two-phase inference.

Generates subtask text autoregressively using the PaliGemma backbone, following
the same JIT compilation pattern as openpi's action generation (module_jit).

Two decode paths are available:
  - Eager (use_jit_decode=False): prefix is JIT-compiled, AR decode loop runs
    eagerly with growing KV cache shapes. ~3.9s per generation.
  - JIT-unrolled (use_jit_decode=True, default): prefix + AR decode are compiled
    into a single XLA graph. The Python for loop is unrolled by JAX's tracer,
    so each iteration has concrete shapes. ~1.1s per generation.

Architecture reference:
    pi0.5 paper Section V.E, Figure 7 — subtask generation uses the prompt
    format "Task: {task}. Subtask: " and decodes via weight-tied lm_head
    (dot(hidden_state, embed_tokens.T) -> vocab logits).
"""

from __future__ import annotations

import functools
import string
import threading
import time
from typing import Any, Protocol

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece
from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer

from hosting.config import DEFAULT_GENERATION_PROMPT_FORMAT

EOS_TOKEN_ID = 1


class _RuntimeConfigLike(Protocol):
    """Structural protocol for the one admin-mutable setting we consume.

    Kept structural (not a hard import of ``admin_server.RuntimeConfig``)
    to avoid coupling this module to the admin endpoint — any object with
    a mutable ``generation_prompt_format: str`` satisfies it, which keeps
    tests and alternate admin implementations simple to swap in.
    """

    generation_prompt_format: str


# ---------------------------------------------------------------------------
# Vocabulary masking — restrict generation to ASCII-only tokens
# ---------------------------------------------------------------------------


def _build_ascii_vocab_mask(tokenizer: SubtaskTokenizer) -> np.ndarray:
    """Build a boolean mask over the vocabulary allowing only printable ASCII tokens.

    The base pi0.5 checkpoint's LM head was degraded by post-training. It
    frequently assigns higher probability to non-English tokens (CJK, Korean,
    Cyrillic, math symbols, emoji) than to correct English tokens. This mask
    is applied before argmax to force English-only generation.

    We require both ``isascii()`` AND ``isprintable()`` — the ASCII range
    includes control characters (0x00-0x1F, 0x7F) that leaked through the
    earlier isascii()-only check and manifested as ``\\x16``/``\\x19``/``\\x1d``
    garbage in the DROID eval sweep (2026-04-17). ``str.isprintable()``
    excludes those control chars but keeps spaces, letters, digits, and
    punctuation — exactly what subtask text needs.

    Industry context: this is the same technique as vLLM's ``allowed_token_ids``,
    HuggingFace's ``LogitsProcessor``, and OpenAI's ``logit_bias`` API — all
    constrain the output vocabulary at decode time. More powerful alternatives
    include constrained decoding libraries (``outlines``, ``guidance``,
    ``lm-format-enforcer``) that enforce a regex/grammar, but simple vocabulary
    masking is sufficient here since we only need language filtering.

    Uses ``id_to_piece`` (raw SentencePiece token) rather than ``decode``
    to avoid merging artifacts. The SentencePiece word-boundary marker
    (U+2581, ``▁``) is treated as a space for the ASCII check.
    """
    sp_processor = tokenizer.sentencepiece_processor
    vocab_size = sp_processor.vocab_size()
    mask = np.zeros(vocab_size, dtype=bool)
    mask[0] = True  # PAD
    mask[EOS_TOKEN_ID] = True  # EOS — must remain to stop generation

    for token_id in range(2, vocab_size):
        piece = sp_processor.id_to_piece(token_id)  # ty: ignore[unresolved-attribute]
        # Replace SentencePiece word-boundary marker with ASCII space
        cleaned_piece = piece.replace("\u2581", " ")
        if cleaned_piece and cleaned_piece.isascii() and cleaned_piece.isprintable():
            mask[token_id] = True

    allowed_count = int(mask.sum())
    print(
        f"[subtask] Vocabulary mask: {allowed_count}/{vocab_size} tokens allowed "
        f"({allowed_count / vocab_size * 100:.1f}% ASCII)",
        flush=True,
    )
    return mask


class SubtaskTokenizer:
    """Wraps PaligemmaTokenizer with subtask-specific tokenization methods.

    Handles subtask prompt tokenization. The prompt format is configurable
    at runtime via the admin endpoint (default: ``"Task: {task}. Subtask: "``).
    """

    def __init__(
        self,
        max_len: int = 200,
        prompt_format: str = DEFAULT_GENERATION_PROMPT_FORMAT,
    ) -> None:
        self._inner = PaligemmaTokenizer(max_len=max_len)
        self._sp = self._inner._tokenizer
        self._max_len = max_len
        self.prompt_format: str = prompt_format

    @property
    def sentencepiece_processor(self) -> sentencepiece.SentencePieceProcessor:
        """Access the underlying SentencePiece processor (for vocab mask building)."""
        return self._sp

    def tokenize_prefix(self, task_prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize a task prompt into the subtask generation format.

        Returns (token_ids [max_len], mask [max_len]) with right-padding.
        The prompt format is read from self.prompt_format, which can be
        updated at runtime via the admin endpoint.
        """
        cleaned = task_prompt.lower().strip().replace("_", " ").replace("\n", " ")
        if cleaned and cleaned[-1] in string.punctuation:
            cleaned = cleaned[:-1]
        prefix_str = self.prompt_format.format(task=cleaned)

        tokens = self._sp.encode(prefix_str, add_bos=True)  # ty: ignore[unresolved-attribute]
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            pad_len = self._max_len - tokens_len
            mask = [True] * tokens_len + [False] * pad_len
            tokens = tokens + [0] * pad_len
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=np.bool_)

    def detokenize(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs to text, skipping PAD (0) and EOS (1)."""
        valid_ids = [t for t in token_ids if t not in (0, EOS_TOKEN_ID)]
        return self._sp.decode(valid_ids)  # ty: ignore[unresolved-attribute]


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to float32 [-1, 1] for the JAX model.

    Handles:
      - uint8 [0, 255] → float32 [-1, 1]
      - float in [0, 1] range → rescale to [-1, 1]
      - float already in [-1, 1] → pass through
      - CHW layout → transpose to HWC
    """
    # Transpose CHW → HWC if needed (C dimension is 3, H/W are larger)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[1] > 3:
        image = np.transpose(image, (1, 2, 0))

    # Convert uint8 to float32 [-1, 1]
    if image.dtype == np.uint8:
        return (image.astype(np.float32) / 127.5) - 1.0

    # Float images: check range and rescale if in [0, 1]
    image = image.astype(np.float32)
    if image.min() >= -0.01 and image.max() <= 1.01:
        return image * 2.0 - 1.0

    return image


def _build_subtask_observation(
    task_prompt: str,
    images: dict[str, np.ndarray],
    tokenizer: SubtaskTokenizer,
    action_dim: int = 32,
) -> _model.Observation:
    """Build a JAX Observation for subtask generation.

    Args:
        task_prompt: The high-level task description (e.g., "pick up the red cup").
        images: Camera images as numpy arrays. Accepts any format — uint8 or float,
            HWC or CHW. Keys should use the JAX model's naming convention
            (base_0_rgb, etc.) after camera name mapping.
        tokenizer: SubtaskTokenizer instance.
        action_dim: Action dimension for the dummy state vector.
    """
    tokens, mask = tokenizer.tokenize_prefix(task_prompt)
    state = np.zeros(action_dim, dtype=np.float32)

    expected_image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    jax_images: dict[str, jnp.ndarray] = {}
    jax_image_masks: dict[str, jnp.ndarray] = {}
    for key in expected_image_keys:
        if key in images:
            image_array = _normalize_image(images[key])
            jax_images[key] = jnp.array(image_array[None])  # add batch dim
            jax_image_masks[key] = jnp.array([True])
        else:
            # Zero image fallback — model still runs but subtask quality degrades
            jax_images[key] = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
            jax_image_masks[key] = jnp.array([True])

    return _model.Observation(
        images=jax_images,
        image_masks=jax_image_masks,
        state=jnp.array(state[None]),
        tokenized_prompt=jnp.array(tokens[None]),
        tokenized_prompt_mask=jnp.array(mask[None]),
    )


# ---------------------------------------------------------------------------
# JIT-compiled prefix forward
# ---------------------------------------------------------------------------


def _prefix_forward_impl(
    model: Pi0,
    observation: _model.Observation,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
    """Prefix forward: preprocess + embed + transformer layers + initial logits.

    This function is JIT-compiled via the module_jit pattern (split model state,
    jax.jit the pure function, merge state inside). It handles the expensive
    prefix encoding (~12s eager → <1s JIT).

    Returns:
        initial_logits: [B, vocab_size]
        embed_table: [vocab_size, D]
        prefix_mask: [B, S]
        last_position: [B]
        kv_cache: tuple of (keys, values) arrays
    """
    observation = _model.preprocess_observation(None, observation, train=False)

    # Embed prefix (SigLIP image encoding + language token embedding)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    batch_size = prefix_tokens.shape[0]
    prefix_sequence_length = prefix_tokens.shape[1]

    # Forward through PaliGemma transformer → KV cache
    prefix_attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask.astype(jnp.int32), axis=1) - 1  # ty: ignore[unresolved-attribute]

    (prefix_output, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attention_mask,
        positions=positions,
        adarms_cond=[None, None],
    )

    # Find last valid token position (LisavilaLee's fix from openpi#701)
    sequence_indices = jnp.arange(prefix_sequence_length)[None, :]
    last_position = jnp.max(
        jnp.where(prefix_mask, sequence_indices, jnp.array(-1)),  # ty: ignore[no-matching-overload]
        axis=1,
    ).astype(jnp.int32)

    # Project last hidden state to vocab logits via weight-tied lm_head
    last_hidden_state = prefix_output[jnp.arange(batch_size), last_position, :]
    embed_table = model.PaliGemma.llm.embedder["input_embedding"].value  # ty: ignore[unresolved-attribute]
    initial_logits = jnp.dot(last_hidden_state, embed_table.T)

    return initial_logits, embed_table, prefix_mask, last_position, kv_cache  # ty: ignore[invalid-return-type]


def _make_jit_prefix_forward(
    model: Pi0,
) -> Any:
    """Create a JIT-compiled prefix forward function using the module_jit pattern.

    Follows the same approach as openpi's nnx_utils.module_jit: split the model
    into (graphdef, state), JIT a pure function that merges them, freeze state.
    """
    graphdef, state = nnx.split(model)

    def pure_prefix_forward(
        frozen_state: nnx.State, observation: _model.Observation
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        reconstructed_model = nnx.merge(graphdef, frozen_state)
        return _prefix_forward_impl(reconstructed_model, observation)

    jitted_forward = jax.jit(pure_prefix_forward)

    @functools.wraps(_prefix_forward_impl)
    def wrapper(
        observation: _model.Observation,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any]:
        return jitted_forward(state, observation)

    return wrapper


# ---------------------------------------------------------------------------
# JIT-compiled full generate (prefix + unrolled AR decode)
# ---------------------------------------------------------------------------


def _full_generate_impl(
    model: Pi0,
    observation: _model.Observation,
    max_tokens: int,
    valid_vocab_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Combined prefix forward + AR decode, designed to be JIT-compiled as one graph.

    The Python for loop is unrolled by JAX's tracer — each iteration becomes a
    separate set of XLA ops with concrete shapes. The concat-based KV cache is
    fine because there's no loop at the XLA level; it's max_tokens straight-line
    subgraphs.

    Returns:
        token_buffer: [max_tokens] int32 array of generated token IDs (0 after EOS)
        done: scalar bool, True if EOS was generated
    """
    # Prefix forward (reuse existing implementation)
    initial_logits, embed_table, prefix_mask, last_position, kv_cache = _prefix_forward_impl(
        model, observation
    )

    # Unrolled decode loop — traced by JAX, compiled to XLA
    token_buffer = jnp.zeros(max_tokens, dtype=jnp.int32)
    current_logits = initial_logits
    current_cache = kv_cache
    next_position = last_position + 1
    done = jnp.array(False)
    batch_size = initial_logits.shape[0]

    for step in range(max_tokens):
        # Mask non-ASCII tokens before argmax to prevent Unicode garbage
        masked_logits = jnp.where(valid_vocab_mask, current_logits, jnp.float32(-1e9))
        token_id = jnp.argmax(masked_logits[0], axis=-1).astype(jnp.int32)
        token_buffer = token_buffer.at[step].set(jnp.where(done, jnp.int32(0), token_id))
        done = done | (token_id == EOS_TOKEN_ID)

        # Embed the generated token (existing model method)
        token_emb = model.PaliGemma.llm(
            jnp.array([[token_id]], dtype=jnp.int32), method="embed"
        )  # [B, 1, D]

        # Attention mask: new token attends to all prefix + generated tokens
        generated_mask = jnp.ones((batch_size, step + 1), dtype=jnp.bool_)
        full_mask = jnp.concatenate([prefix_mask, generated_mask], axis=1)
        attention_mask = full_mask[:, None, :]  # [B, 1, total_seq_len]

        # Forward with KV cache (existing model call)
        (new_output, _), current_cache = model.PaliGemma.llm(
            [token_emb, None],
            mask=attention_mask,
            positions=next_position[:, None],
            kv_cache=current_cache,
        )

        # Project to vocab logits
        current_logits = jnp.dot(new_output[:, -1, :], embed_table.T)
        next_position = next_position + 1

    return token_buffer, done


def _make_jit_full_generate(
    model: Pi0,
    max_tokens: int,
    valid_vocab_mask: jnp.ndarray,
) -> Any:
    """JIT-compile the full generate path (prefix + AR decode) using module_jit pattern.

    The entire generation (SigLIP encoding, 18 transformer layers for prefix,
    then max_tokens unrolled decode steps) is compiled into a single XLA graph.
    The vocab mask is captured in the closure and embedded as a constant.
    """
    graphdef, state = nnx.split(model)

    def pure_full_generate(
        frozen_state: nnx.State, observation: _model.Observation
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        reconstructed_model = nnx.merge(graphdef, frozen_state)
        return _full_generate_impl(reconstructed_model, observation, max_tokens, valid_vocab_mask)

    jitted_generate = jax.jit(pure_full_generate)

    @functools.wraps(_full_generate_impl)
    def wrapper(
        observation: _model.Observation,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jitted_generate(state, observation)

    return wrapper


# ---------------------------------------------------------------------------
# Eager AR decode loop (kept for A/B comparison with use_jit_decode=False)
# ---------------------------------------------------------------------------


def _autoregressive_decode(
    model: Pi0,
    initial_logits: jnp.ndarray,
    embed_table: jnp.ndarray,
    prefix_mask: jnp.ndarray,
    last_position: jnp.ndarray,
    kv_cache: Any,
    max_tokens: int,
    valid_vocab_mask: jnp.ndarray | None = None,
) -> list[int]:
    """Run the AR decode loop (eager mode, 3-5 tokens typically).

    Each step: argmax → embed → forward with KV cache → project to logits.
    Runs eagerly because the KV cache grows each step (variable shapes
    incompatible with jax.lax.while_loop).
    """
    batch_size = initial_logits.shape[0]
    current_logits = initial_logits
    current_cache = kv_cache
    next_position = last_position + 1
    generated_token_ids: list[int] = []

    for step_index in range(max_tokens):
        # Mask non-ASCII tokens before argmax to prevent Unicode garbage
        decode_logits = current_logits
        if valid_vocab_mask is not None:
            decode_logits = jnp.where(valid_vocab_mask, decode_logits, jnp.float32(-1e9))
        token_id = int(jnp.argmax(decode_logits[0]))
        generated_token_ids.append(token_id)

        if token_id == EOS_TOKEN_ID:
            break

        # Embed the generated token
        token_array = jnp.array([[token_id]], dtype=jnp.int32)
        token_embedding = model.PaliGemma.llm(token_array, method="embed")  # [B, 1, D]

        # Attention mask: new token attends to all prefix + all generated tokens
        generated_count = step_index + 1
        generated_mask = jnp.ones((batch_size, generated_count), dtype=jnp.bool_)
        full_mask = jnp.concatenate(
            [prefix_mask, generated_mask],
            axis=1,
        )
        attention_mask = full_mask[:, None, :]  # [B, 1, total_seq_len]

        new_positions = next_position[:, None]  # [B, 1]

        # Forward with KV cache
        (new_output, _), current_cache = model.PaliGemma.llm(
            [token_embedding, None],
            mask=attention_mask,
            positions=new_positions,
            kv_cache=current_cache,
        )

        # Project to vocab logits
        new_hidden_state = new_output[:, -1, :]
        current_logits = jnp.dot(new_hidden_state, embed_table.T)
        next_position = next_position + 1

    return generated_token_ids


# ---------------------------------------------------------------------------
# Camera name mapping per embodiment
# ---------------------------------------------------------------------------

# Maps client-side camera names to JAX model's expected names.
# The JAX model always expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
CAMERA_NAME_MAPPINGS: dict[str, dict[str, str]] = {
    "aloha": {
        "cam_high": "base_0_rgb",
        "cam_left_wrist": "left_wrist_0_rgb",
        "cam_right_wrist": "right_wrist_0_rgb",
    },
    "droid": {
        "observation/exterior_image_1_left": "base_0_rgb",
        "observation/wrist_image_left": "left_wrist_0_rgb",
    },
    "galaxea_r1": {
        "head_camera": "base_0_rgb",
        "left_wrist_camera": "left_wrist_0_rgb",
        "right_wrist_camera": "right_wrist_0_rgb",
    },
}


def _detect_embodiment(client_image_keys: set[str]) -> str | None:
    """Auto-detect embodiment from the client's camera key names."""
    for embodiment_name, mapping in CAMERA_NAME_MAPPINGS.items():
        if client_image_keys & set(mapping.keys()):
            return embodiment_name
    return None


def _map_camera_names(
    client_images: dict[str, np.ndarray],
    embodiment_name: str,
) -> dict[str, np.ndarray]:
    """Map client camera names to JAX model camera names.

    If embodiment_name is empty, auto-detects from the image keys. If the
    images already use JAX model names (base_0_rgb, etc.), passes through.
    """
    if not embodiment_name:
        embodiment_name = _detect_embodiment(set(client_images.keys())) or ""

    mapping = CAMERA_NAME_MAPPINGS.get(embodiment_name)
    if mapping is None:
        return client_images

    mapped_images: dict[str, np.ndarray] = {}
    for client_name, jax_name in mapping.items():
        if client_name in client_images:
            mapped_images[jax_name] = client_images[client_name]
    return mapped_images


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SubtaskGenerator:
    """JIT-compiled JAX subtask text generator for pi0.5.

    Loads a JAX model from an Orbax checkpoint and generates subtask text
    autoregressively. The prefix forward is JIT-compiled for performance.

    Usage:
        generator = SubtaskGenerator("gs://openpi-assets/checkpoints/pi05_base")
        generator.load()
        subtask = generator.generate("pick up the red cup", images)
        # subtask == "pick up cup"
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_tokens: int = 20,
        embodiment_name: str = "",
        use_jit_decode: bool = True,
        runtime_config: _RuntimeConfigLike | None = None,
        generation_prompt_format: str = DEFAULT_GENERATION_PROMPT_FORMAT,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._max_tokens = max_tokens
        self._embodiment_name = embodiment_name
        self._use_jit_decode = use_jit_decode
        self._runtime_config = runtime_config
        self._initial_prompt_format = generation_prompt_format
        self._model: Pi0 | None = None
        self._tokenizer: SubtaskTokenizer | None = None
        self._valid_vocab_mask: jnp.ndarray | None = None
        self._jit_prefix_forward: Any = None
        self._jit_full_generate: Any = None
        # Serializes generate() across the action and planner endpoints in
        # combined mode. The two transport threads share this one instance.
        self._generate_lock = threading.Lock()

    def load(self) -> None:
        """Load JAX model, patch tokenizer, and JIT-compile the prefix forward."""
        import pathlib
        import shutil

        from openpi.shared import download as _download

        resolved_checkpoint_dir = str(_download.maybe_download(self._checkpoint_dir))

        # Guard against partial GCS downloads cached by maybe_download. If the
        # directory exists but params/_METADATA is missing, the previous download
        # was incomplete — remove the stale cache and force a fresh download.
        params_metadata_path = pathlib.Path(resolved_checkpoint_dir) / "params" / "_METADATA"
        if not params_metadata_path.exists():
            print(
                f"[subtask] Incomplete checkpoint at {resolved_checkpoint_dir} "
                f"(missing {params_metadata_path}), re-downloading",
                flush=True,
            )
            shutil.rmtree(resolved_checkpoint_dir, ignore_errors=True)
            resolved_checkpoint_dir = str(
                _download.maybe_download(self._checkpoint_dir, force_download=True)
            )
            if not (pathlib.Path(resolved_checkpoint_dir) / "params" / "_METADATA").exists():
                raise FileNotFoundError(
                    f"JAX checkpoint download failed: params/_METADATA still missing at "
                    f"{resolved_checkpoint_dir} after forced re-download"
                )

        print(f"[subtask] Loading JAX model from {resolved_checkpoint_dir}", flush=True)
        load_start = time.monotonic()

        config = Pi0Config(pi05=True)
        rng = jax.random.key(0)
        model = config.create(rng)

        params = _model.restore_params(f"{resolved_checkpoint_dir}/params", dtype=jnp.bfloat16)
        nnx.update(model, nnx.State(params))
        model.eval()
        self._model = model

        load_elapsed = time.monotonic() - load_start
        print(f"[subtask] JAX model loaded in {load_elapsed:.1f}s", flush=True)

        # Build tokenizer and vocabulary mask
        self._tokenizer = SubtaskTokenizer(max_len=200, prompt_format=self._initial_prompt_format)
        self._valid_vocab_mask = jnp.array(_build_ascii_vocab_mask(self._tokenizer))

        # JIT-compile the generation path
        if self._use_jit_decode:
            print(
                "[subtask] Preparing JIT-compiled full generate "
                f"(prefix + {self._max_tokens}-step unrolled decode)...",
                flush=True,
            )
            self._jit_full_generate = _make_jit_full_generate(
                model, self._max_tokens, self._valid_vocab_mask
            )
            print("[subtask] Full generate JIT ready", flush=True)
        else:
            print("[subtask] Preparing JIT-compiled prefix forward (eager decode)...", flush=True)
            self._jit_prefix_forward = _make_jit_prefix_forward(model)
            print("[subtask] Prefix forward JIT ready", flush=True)

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, task_prompt: str, client_images: dict[str, np.ndarray] | None = None) -> str:
        """Generate subtask text for a given task prompt and images.

        Safe to call concurrently from multiple transports — an internal lock
        serializes access to the JAX runtime.

        Args:
            task_prompt: High-level task description (e.g., "pick up the red cup").
            client_images: Camera images from the client, keyed by client camera names.
                Images should be numpy arrays in [H, W, C] format, float32 [-1, 1].
                If None, zero images are used (subtask quality will degrade).

        Returns:
            Generated subtask text (e.g., "pick up cup").
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("SubtaskGenerator not loaded. Call load() first.")

        with self._generate_lock:
            # Sync prompt format from runtime config (updated via admin endpoint)
            if self._runtime_config is not None:
                self._tokenizer.prompt_format = self._runtime_config.generation_prompt_format

            # Map client camera names to JAX model names
            jax_images: dict[str, np.ndarray] = {}
            if client_images:
                jax_images = _map_camera_names(client_images, self._embodiment_name)

            observation = _build_subtask_observation(task_prompt, jax_images, self._tokenizer)

            if self._use_jit_decode and self._jit_full_generate is not None:
                # JIT path: single compiled graph for prefix + decode
                token_buffer, _done = self._jit_full_generate(observation)
                generated_token_ids = [
                    int(t) for t in token_buffer if int(t) not in (0, EOS_TOKEN_ID)
                ]
            else:
                # Eager path: JIT prefix + eager decode loop
                initial_logits, embed_table, prefix_mask, last_position, kv_cache = (
                    self._jit_prefix_forward(observation)
                )
                generated_token_ids = _autoregressive_decode(
                    self._model,
                    initial_logits,
                    embed_table,
                    prefix_mask,
                    last_position,
                    kv_cache,
                    self._max_tokens,
                    valid_vocab_mask=self._valid_vocab_mask,
                )
                # Strip EOS token if present
                if EOS_TOKEN_ID in generated_token_ids:
                    generated_token_ids = generated_token_ids[
                        : generated_token_ids.index(EOS_TOKEN_ID)
                    ]

            return self._tokenizer.detokenize(generated_token_ids)

    def warmup(self) -> None:
        """Run one generation to trigger JIT compilation.

        With use_jit_decode=True, this compiles the full prefix + decode graph
        (may take 60-90s due to unrolled decode loop). With use_jit_decode=False,
        this only compiles the prefix forward (~33s).
        """
        if not self.is_loaded():
            raise RuntimeError("SubtaskGenerator not loaded. Call load() first.")

        mode_label = "full generate (prefix + decode)" if self._use_jit_decode else "prefix only"
        print(f"[subtask] Running warmup ({mode_label})...", flush=True)
        warmup_start = time.monotonic()
        result = self.generate("warmup test prompt")
        warmup_elapsed = time.monotonic() - warmup_start
        print(
            f'[subtask] Warmup complete in {warmup_elapsed:.1f}s (generated: "{result}")',
            flush=True,
        )
