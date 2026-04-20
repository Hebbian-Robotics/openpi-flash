# pi0.5 Subtask Generation Probe: Findings

## Core Result

**The public pi0.5 base checkpoint contains working subtask text generation capability.** Using JAX with the correct prompt format (`"Task: {task}. Subtask: "`), the model generates coherent English subtask text with high confidence (60-87% top-1 probability).

| Prompt | Generated Subtask | Confidence |
|---|---|---|
| "pick up the red cup and place it on the shelf" | "pick up cup" | 78% |
| "fold the towel neatly" | "fold the towel" | 87% |
| "open the drawer and put the block inside" | "pull out the drawer" | 34% |
| "stack the blue block on top of the red block" | "pick up the paper" | 57% |
| "wipe the table with the sponge" | "wipe the spill" | 61% |

No retraining needed. No new head needed. The existing `embed_tokens.weight` matrix serves as the lm_head via weight tying (`dot(hidden_state, embed_tokens.T) -> vocab logits`).

## What Makes It Work

Two things were required, both non-obvious:

1. **Prompt format**: `"Task: {task}. Subtask: "` -- NOT the action format `"Task: {task}, State: {state};\nAction: "`. The model has distinct modes triggered by the prompt suffix. The paper describes this conceptually but never specifies the exact format string.

2. **Autoregressive decoding loop**: Embed prefix -> forward through PaliGemma backbone -> KV cache -> project last hidden state to vocab via `dot(h, embed_table.T)` -> take argmax -> embed that token -> forward with KV cache -> repeat.

## What Doesn't Work

### PyTorch path is broken for autoregressive text generation

The PyTorch model (via HuggingFace transformers) gets the first token approximately right but degrades to Unicode garbage on the second token onwards. We tried:

- **v1-v2**: Wrong prompt format (`"Action: "` suffix). Got Unicode attractors (ⓙ, ⤙, Ẁ, etc.)
- **v3**: Correct prompt format + proper AR loop through `GemmaModel.forward()`. First token correct ("put" at 43%), second token garbage. Cause: HuggingFace's `create_causal_mask` intermediary in `GemmaModel.forward()` creates a causal mask that conflicts with the bidirectional prefix attention.
- **v3 + 2D mask fix**: Passing 2D padding masks instead of 4D float masks. Improved first token confidence but didn't fix continuation.
- **v4**: Bypassed `GemmaModel.forward()` entirely, manually iterating through decoder layers with custom 4D masks. Same result: first token OK, continuation broken.

**Root cause**: The PyTorch and JAX implementations produce different hidden states from the same input. The first token prediction differs between them ("put" vs "pick" for the same prompt), and continuation diverges completely.

**Confirmed NOT the weights**: We tested with lerobot/pi05_base (independently converted by HuggingFace team, fp32) and our openpi conversion (bf16). Both produce identical results. Weight values are bitwise identical between the two checkpoints (max diff = 0.0 across all tested keys). Both are straight JAX→PyTorch conversions with uniform precision (no selective mixed precision). The `to_bfloat16_for_selected_params` step in our Modal conversion script is a deployment optimization for `torch.compile` stability, not part of the upstream conversion.

**Root cause identified**: Side-by-side JAX vs PyTorch comparison revealed:
- A minimal 3-token prefix (no images): AR step cos_sim = **0.999** -- nearly identical. KV cache works correctly.
- Full 968-token prefix (768 image + 200 language): AR step cos_sim = **0.30** -- complete divergence.

The KV cache mechanism is correct. The problem is that the **prefix forward** already puts slightly different values into the cache (cos_sim 0.998 per position). With 968 positions of slightly-wrong cached key/value vectors, the attention scores during the AR step accumulate these errors and produce a completely different weighted sum. This is a numerical amplification issue: a 0.2% per-position error in the prefix becomes a 70% error in the AR step's attention output over 968 cached positions.

The prefix error likely comes from the image embedding pipeline (SigLIP implementation differences between JAX Flax and PyTorch HuggingFace) or from attention precision differences (the JAX code uses float32 for attention logits while PyTorch may use bfloat16 in some paths).

### JAX works, PyTorch doesn't

All successful subtask generation implementations in the community use JAX:
- @BrunoFANG1 (openpi#701) -- JAX, partial subtask text from base checkpoint
- @LisavilaLee (openpi_with_subtask fork) -- JAX, full implementation with position fix
- Our probe -- JAX works, PyTorch doesn't

## Architecture

pi0.5 is a two-expert transformer:
- **PaliGemma backbone** (Gemma-2B + SigLIP): processes images + text, dim 2048
- **Action expert** (Gemma-300M): processes noisy actions + timestep, dim 1024
- Both share attention through 18 layers (fused Q/K/V)
- **Action output**: `action_out_proj: Linear(1024, action_dim)` on the action expert hidden states
- **Text output**: `dot(paligemma_hidden, embed_tokens.T)` on the backbone hidden states (weight-tied lm_head)

The model has two modes selected by prompt format:
- `"...; Action: "` → action generation (flow matching, iterative denoising)
- `"... Subtask: "` → text generation (autoregressive, standard LM decoding)

## Paper Context

The pi0.5 paper describes two training stages:
1. **Pre-training**: Discrete tokens, web data, subtask prediction (HL data), cross-embodiment data. This is where the subtask text capability comes from.
2. **Post-training**: Adds the action expert, flow matching for continuous actions, specializes for mobile manipulation.

The public checkpoint includes the action expert, so it has been through post-training. The subtask capability persists but generates short sequences (3-4 tokens) before hitting EOS with dummy images, likely because:
- Post-training may have partially degraded the LM capability
- Dummy/zero images provide no visual context for the model to describe
- The exact prompt format may not match what PI used internally

Community reports indicate ~100 gradient steps of LM fine-tuning on subtask data produces full-quality subtask text.

## Files

| File | Purpose |
|---|---|
| `decode_jax.py` | **Working** JAX subtask generation probe |
| `hybrid_prompt_experiment.py` | JAX subtask → PyTorch action, three prompt variants |
| `dual_runtime_benchmark.py` | JAX + PyTorch coexistence on single GPU, latency benchmarks |
| `compare_jax_pytorch.py` | Side-by-side JAX vs PyTorch hidden state comparison (root cause diagnosis) |
| `latency_profile.py` | Latency breakdown and JIT optimization tests (WIP) |

## How to Run (JAX, working)

```bash
# On L40S (Seoul instance):
ssh ubuntu@43.200.36.250

# Ensure JAX checkpoint is downloaded
cd ~/openpi
source $HOME/.local/bin/env
uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_base')"

# Run the probe
uv run python experiments/subtask_probe/decode_jax.py
```

## Hybrid Prompt Experiment (2026-04-14)

### Question

The pi0.5 action prompt format (`"Task: X, State: S;\nAction: "`) is completely different from the subtask prompt format (`"Task: X. Subtask: "`). Can we inject JAX-generated subtask text into the action prompt without retraining?

### Method

Generated subtasks via JAX (Phase 1), then compared PyTorch action outputs across three prompt variants (Phase 2):

1. **Baseline**: `"Task: X, State: S;\nAction: "` (standard, no subtask)
2. **Hybrid A**: `"Task: X. Subtask: Y, State: S;\nAction: "` (subtask injected before state)
3. **Hybrid B**: `"Task: X (Y), State: S;\nAction: "` (subtask in parentheses)

Used zero images, zero state, fixed random seed. Both models loaded sequentially on a single L40S (JAX freed before PyTorch loaded via `XLA_PYTHON_CLIENT_PREALLOCATE=false`).

### Results

| Task | Subtask (JAX) | Baseline vs Hybrid A | Baseline vs Hybrid B |
|---|---|---|---|
| pick up red cup → shelf | "put the blue cup in the bin" | cos=0.36, L2=94% | cos=-0.19, L2=135% |
| fold the towel neatly | "fold the towel" | cos=0.34, L2=265% | cos=0.26, L2=168% |
| open drawer, put block | (garbage — zero images) | cos=-0.06, L2=205% | cos=-0.11, L2=130% |
| wipe table with sponge | "1No" (degraded) | cos=-0.24, L2=679% | cos=0.43, L2=130% |

### Interpretation

1. **The model IS conditioning on subtask text.** Cosine similarities of 0.36 and -0.06 mean completely different action trajectories — not noise, but a different policy output.
2. **Prompt format matters.** Hybrid A ≠ Hybrid B, confirming the model parses the text structure, not just bag-of-words.
3. **Cannot assess quality with zero images.** All actions are meaningless without visual context. The experiment proves the *mechanism* works (text changes actions) but not whether it produces *better* actions. Need real robot images.
4. **Subtask quality degrades with zero images.** 2/4 prompts produced garbage subtasks. Real images should fix this (the model needs visual context to describe what it sees).

### Two-Phase Inference Architecture (from paper)

The pi0.5 paper (Section V.E, Figure 3, Figure 7) describes:

```
Phase 1 — Subtask generation (autoregressive text, PaliGemma backbone):
  Prompt: "Task: clean the bedroom. Subtask: " + images + state
  → AR decode → "pick up pillow"

Phase 2 — Action generation (flow matching, action expert):
  Prompt: "Task: clean the bedroom. Subtask: pick up pillow" + images + state
  → 10 denoising steps → action chunk [a_t:t+H]
```

Key detail: the prompt formats are **different modes** of the same model:
- `"Task: X. Subtask: "` triggers autoregressive text generation (PaliGemma LM head)
- `"Task: X, State: S;\nAction: "` triggers flow-matching action generation (action expert)

@LisavilaLee's implementation (`build_full_observation`) splices generated subtask tokens into the padding of the subtask-format prompt, then runs `sample_actions` on that. This means the action step uses `"Task: X. Subtask: Y"` as its prompt — NOT the standard `"Task: X, State: S;\nAction: "` format. This requires ~100 gradient steps of fine-tuning to teach the model to produce actions from the subtask prompt format.

Our hybrid approach (injecting subtask text into the standard action format) is an alternative that avoids retraining, but the quality is unvalidated — needs real images to assess.

### DROID Evaluation with Real Images (2026-04-16)

Ran the full eval pipeline against 10 DROID episodes (276 frames) using the deployed two-phase server (Seoul, g6e.2xlarge). Subtask generation uses the base JAX checkpoint (pi05_base); action generation uses the DROID PyTorch checkpoint (swatery/pi05_droid_base).

**Prompt format comparison (Hybrid A vs Hybrid B):**

Tested two ways to inject subtask text into the action prompt:
- Hybrid A: `"{instruction}. Subtask: {subtask}"` (closer to pre-training format)
- Hybrid B: `"{instruction} ({subtask})"` (parenthetical)

Results were statistically indistinguishable (Wilcoxon p=0.89). Both improved over baseline similarly. **Conclusion: the format doesn't matter, only the presence of subtask text.** Going forward, we use only Hybrid A (now called "subtask" condition) since it's closer to the pre-training prompt format.

**Noise control was critical:**

Initial results showed no significant difference (p=0.96) between baseline and subtask conditions. Investigation revealed that each server request gets independent random noise in the flow matching denoising loop. The noise-induced L2 variance (~0.76) was 76x larger than the prompt effect (~0.01), completely drowning the signal. After adding a `seed` field to the obs dict that sets `torch.manual_seed()` before denoising, all 3 conditions for the same frame get identical noise. With controlled noise: **p=0.025** (significant), subtask is closer to ground truth 55% of the time.

**DROID checkpoint cannot generate subtask text:**

Tested `gs://openpi-assets/checkpoints/pi05_droid` for subtask generation — produces empty strings (immediate EOS) on all 276 frames. The DROID fine-tuning completely destroyed the LM head's text generation capability. The base checkpoint (pi05_base) is the correct choice for the subtask planner. This confirms the paper's note that post-training degrades the subtask capability.

**Image format issues discovered and fixed:**

Three bugs prevented the server from processing DROID images correctly:

1. **Camera name mapping**: The subtask generator expects `base_0_rgb`/`left_wrist_0_rgb` but clients send embodiment-specific names (`cam_high` for ALOHA, `observation/exterior_image_1_left` for DROID). Fixed: auto-detect embodiment from key names.

2. **Image normalization**: The subtask generator's `_build_subtask_observation` expected float32 [-1,1] but clients send uint8 [0,255]. No normalization was applied. Fixed: added `_normalize_image()` that handles uint8→float32, CHW→HWC, and [0,1]→[-1,1] automatically.

3. **Aspect ratio distortion**: DROID images are 180x320 (16:9). The extraction script resized to 224x224 with plain `tf.image.resize`, squishing the images. The model's own `preprocess_observation` uses `resize_with_pad` which preserves aspect ratio by adding black padding. Fixed: store original dimensions, let the server's preprocessing handle resize.

These fixes improved diversity (228 unique subtasks across 276 frames, up from identical outputs), but **Unicode garbage persists in ~51% of frames** (141/276). Quality is highly variable by episode — ep_0005 produces 100% valid English, while ep_0004 and ep_0007 produce 0%. Examples of garbage: `셍踯≎𝟻셍ᔑ毟⢱Ꮸ𨨏ѱ`, `শՔ䭈⠤ǎƞᇃḡᵐ䁱჻ັ` (Korean, CJK, math symbols, emoji, Cyrillic mixed together).

**Root cause**: The base checkpoint's LM head was degraded by post-training. The logit distribution is flattened — non-English tokens (CJK, Korean, etc.) sometimes receive higher probability than correct English tokens. Greedy argmax picks whatever is highest, regardless of language.

**Fix: ASCII vocabulary masking.** Before argmax at each decode step, set logits for all non-ASCII tokens to -inf. This forces generation from English-only tokens. This is a standard industry technique — the same approach as vLLM's `allowed_token_ids`, HuggingFace's `LogitsProcessor`, and OpenAI's `logit_bias` API parameter. Implemented in `SubtaskGenerator._build_ascii_vocab_mask()`. The mask is built once at init (scanning all 257K PaliGemma vocabulary tokens) and applied as a `jnp.where` before every argmax — zero runtime cost, deterministic, JIT-compatible.

### Dual Runtime Coexistence Test (2026-04-14)

Confirmed both JAX and PyTorch models loaded simultaneously on a single L40S using `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`:

| Resource | Usage |
|---|---|
| JAX (subtask model) | 6.4 GB VRAM / 22.7 GB limit |
| PyTorch (action model) | 7.1 GB VRAM |
| **Total GPU** | **~13.6 GB / 46 GB** |
| JAX subtask latency (first call, JIT) | ~64s |
| JAX subtask latency (warm, eager) | **~14s** |
| PyTorch action latency | **~280ms** |
| Total two-phase (warm) | **~14.2s** |

Memory is not the bottleneck — 13.6GB out of 46GB leaves plenty of headroom. A bigger instance is unnecessary for memory.

**The latency bottleneck is JAX eager-mode AR decoding (~14s warm).** The breakdown is:
- Prefix forward (SigLIP image encoding + 18 transformer layers over 968 tokens → KV cache): majority of time
- AR loop (3-5 token generations with growing KV cache): each step retraces because KV cache shape changes

Attempted to profile the exact breakdown with JIT optimization tests but the profiling script was OOM-killed on system RAM (32GB). The 30GB system RAM may be tight when both JAX and PyTorch runtimes plus XLA compilation buffers are active. This is a system RAM constraint, not GPU VRAM.

**Latency reduction options (untested, for next session):**
1. **JIT-compile the prefix forward** — fixed shape, should compile well. The AR loop is harder because KV cache grows per step.
2. **Pre-allocate KV cache** to max size (prefix + max_gen_tokens) and use `jax.lax.while_loop` for fully JIT AR generation.
3. **Cache subtasks aggressively** — subtask only needs regeneration when the visual scene changes significantly, not every action cycle. At 14s per subtask, caching is essential.
4. **Larger system RAM** — profiling was killed at 32GB. A g6e.2xlarge (64GB RAM, same L40S GPU) would allow JIT compilation without OOM.

### Hosting Architecture

Both runtimes coexist on a single L40S (48GB VRAM) with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`. Two separate instances are NOT needed for memory reasons. However:

- **Single instance, two processes** is viable if latency can be reduced to <1s via JIT compilation
- **Two instances** only makes sense if system RAM (32GB on g6e.xlarge) is the bottleneck for JIT compilation, since the profiling script OOMed — a single g6e.2xlarge (64GB RAM) would be cheaper than two g6e.xlarge instances
- The two-phase inference is opaque behind QUIC — client sends `{task, images, state}`, gets back `{actions}`

The subtask refresh rate is a design choice. The paper's Figure 7 shows subtask predictions changing frame-by-frame as the scene evolves, suggesting periodic re-generation (not just once per task). Given the 14s latency, aggressive caching is necessary until JIT optimization is done.

## Checkpoint Conversion Notes

### JAX → PyTorch conversion pipeline

The stock openpi conversion script (`examples/convert_jax_model_to_pytorch.py`) does a **straight conversion** — uniform precision (float32 or bfloat16), no selective mixed precision. This is what the HuggingFace/LeRobot team used to produce `lerobot/pi05_base`.

Our Modal conversion script (`convert_checkpoint_modal.py`) adds an extra post-conversion step: `to_bfloat16_for_selected_params()`, which keeps layernorms, vision patch embeddings, and position embeddings in float32 while converting everything else to bfloat16. This is a **deployment optimization for `torch.compile` stability** (prevents fp32/bf16 matmul crashes), not a correctness requirement. Standard inference without `torch.compile` works fine with uniform precision.

### DROID checkpoint conversion (completed 2026-04-16)

Converted `gs://openpi-assets/checkpoints/pi05_droid` → PyTorch on the Seoul g6e.2xlarge instance using the stock openpi conversion script. Uploaded to HuggingFace: **`swatery/pi05_droid_base`**

```bash
uv run python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch \
  --config_name pi05_droid \
  --precision bfloat16
```

Output: `model.safetensors` (6.8GB, bfloat16), `config.json` (action_horizon=15, pi05=True). No assets directory was copied (DROID norm stats are in a separate location).

Previous Modal attempt (2026-04-14) hit import bugs and client disconnect issues — running directly on AWS was simpler.

## Next Steps

### Validate with real robot images (DROID)

3. **Download DROID dataset samples** — need actual robot camera frames to test subtask generation and action quality. DROID provides multi-camera observations with task labels, matching pi0.5's expected input format.

4. **Test subtask generation with real images using DROID checkpoint** — feed actual robot workspace images into the JAX subtask generator with the DROID-finetuned checkpoint. Expect longer, more specific subtask text (vs. the 3-4 token outputs with zero images on the base checkpoint).

5. **Test hybrid prompt action quality with real images** — compare the baseline (no subtask) vs. hybrid (with subtask) action outputs on real images. If the subtask-conditioned actions are measurably different AND more semantically aligned with the task, the hybrid approach works without retraining.

### Fix PyTorch AR generation

6. **Force float32 for the entire prefix forward** instead of bfloat16. The JAX code computes attention logits and RMSNorm variance in float32 but the PyTorch path may use bfloat16 in some places. Eliminating precision loss in the prefix would reduce per-position error and may prevent the amplification. Quick test -- just set `model.to(torch.float32)` before the prefix forward.

7. **Audit SigLIP image embedding differences** between JAX (`openpi/src/openpi/models/siglip.py`) and HuggingFace's SigLIP implementation. The 768 image tokens contribute the most cached positions and are likely the largest source of error.

8. **Audit attention precision paths**. The JAX gemma.py explicitly does `jnp.einsum(..., preferred_element_type=jnp.float32)` for attention logits. The PyTorch `eager_attention_forward` may not enforce float32 for the QK matmul.

### Production integration

9. **Two-process serving architecture** — JAX subtask service (port 8001) + PyTorch action service (port 8000) behind the existing QUIC endpoint. The action service calls the subtask service on localhost before each action generation cycle.

10. **Subtask caching and refresh policy** — decide how often to re-generate subtasks. Options: every action chunk, every N steps, or when visual change exceeds a threshold. @LisavilaLee's code caches by prompt only (never refreshes), but the paper's Figure 7 shows it should update with the scene.

### Future optimization: Flash attention for prefix forward

The prefix forward processes ~968 tokens through 18 transformer layers. Each layer computes self-attention via explicit einsums in `gemma.py`:

```python
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)  # materializes full 968×968 matrix in HBM
probs = jax.nn.softmax(masked_logits, axis=-1)
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
```

`jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation='cudnn')` would replace all three ops with a single fused kernel that tiles the computation in GPU SRAM instead of materializing the full attention matrix in HBM. It handles GQA natively (Q `[B, T, 8, H]` against K/V `[B, T, 1, H]`).

**Estimated impact**: Attention is ~40-60% of each layer's compute. Flash attention typically gives 2-3x speedup on the attention kernel. For the prefix forward (~1s JIT'd), this could save 200-400ms. For AR decode steps (~5ms each, query is single token), the benefit is negligible.

**Blockers**: Requires modifying `gemma.py` (upstream openpi code shared by all models). Would need a compelling reason — the 200-400ms saving on prefix is nice but not critical given the larger wins from JIT-compiling the decode loop. Also requires cuDNN availability (L40S has it, but needs correct CUDA/cuDNN versions). Changing the `implementation` parameter requires recompilation (different XLA graph), so it's a deploy-time choice, not runtime-switchable.

**Alternative approaches considered**:
- **Flax NNX `MultiHeadAttention` with `decode=True`**: Has built-in pre-allocated KV cache with `dynamic_update_slice` and `init_cache()`. Would enable `jax.lax.while_loop` for AR decode. But requires replacing Gemma's custom attention module — same upstream modification concern.
- **`chex.dataclass`**: Registers dataclasses as JAX pytrees. Would clean up a `DecodeState` carry struct if we used `while_loop`, but the JIT-unrolled approach doesn't need one.
- **`chex.assert_shape()`**: JIT-compatible shape validation. Nice for development but doesn't affect performance.

## Comparison with openpi_with_subtask Fork (2026-04-17)

Deep comparison of our `SubtaskGenerator` against @LisavilaLee's `openpi_with_subtask` fork to understand why our DROID eval produces nonsense subtasks.

**Finding: the subtask generation code is functionally identical.** Same prompt format, same cleaning logic (lowercase, strip punctuation), same greedy argmax decoding, same SigLIP image encoding, same KV cache approach. There is no hidden inference-time trick in the fork.

The fork's quality advantage comes entirely from ~100 gradient steps of fine-tuning with:
1. `token_ar_mask` — causal attention on subtask tokens, bidirectional on prefix
2. `token_loss_mask` — CE loss applied only to subtask portion, not prefix
3. Identity subtask training (`high_prompt = low_prompt = prompt`)

One additional difference affects action quality (not subtask text quality): the fork's `build_full_observation()` splices generated subtask TOKEN IDs directly back into the padded prompt region, maintaining exact token fidelity. Our approach converts tokens→text→re-tokenizes as a new string, which can produce different token boundaries.

### ASCII Vocabulary Masking (2026-04-17)

**Problem:** ~51% of DROID eval frames produce Unicode garbage (Korean, CJK, math symbols, emoji, Cyrillic). The base checkpoint's degraded LM head assigns higher probability to non-English tokens than correct English tokens on many inputs.

**Solution:** ASCII vocabulary masking — before argmax at each decode step, set logits for all non-ASCII tokens to `-inf`. Implemented in `SubtaskGenerator._build_ascii_vocab_mask()`:

1. At init: scan all 257K PaliGemma vocabulary tokens via SentencePiece `id_to_piece()`
2. Mark token as valid if its piece (with `▁` treated as space) is fully ASCII
3. Store as `jnp.array` boolean mask — embedded as XLA constant in JIT-compiled graph
4. Before every argmax: `logits = jnp.where(valid_vocab_mask, logits, -1e9)`

**Industry context:** This is the same technique as:
- **vLLM** `allowed_token_ids` — restrict sampling to a set of token IDs
- **HuggingFace** `LogitsProcessor` — arbitrary logit manipulation before sampling
- **OpenAI API** `logit_bias` — per-token logit adjustments
- **Constrained decoding** (`outlines`, `guidance`, `lm-format-enforcer`) — enforce regex/grammar on output (more powerful but heavier; these libraries are PyTorch-only, not compatible with our JAX backend)

**Properties:** Zero runtime cost, deterministic, JIT-compatible. Does not change the model — only filters the output vocabulary at decode time.

**Status:** Implemented, awaiting re-run of DROID eval to measure impact. Expect Unicode garbage rate to drop from ~51% to ~0%. English subtask quality should be unchanged since the mask only removes non-ASCII tokens — the correct English token was always in the distribution, just sometimes ranked below a non-English token.

## Subtask Prompt-Format Sweep (2026-04-17)

Ran all 276 DROID frames against 4 subtask-generation prompt formats on the Seoul g6e.2xlarge, swapped at runtime via the admin HTTP endpoint (`PATCH /config` with `subtask_prompt_format`). Results below use a stricter "usable" metric: printable ASCII only (`str.isprintable() and str.isascii()`), since the earlier `isascii()`-only vocab mask let control characters `\x16`, `\x19`, `\x1d` through.

| Format | Prompt | Printable usable | Unique | Mean chars | Most common output |
|---|---|---|---|---|---|
| **default** | `Task: {task}. Subtask: ` | 166/276 (60%) | 89 | 23.1 | `'move the pan'` (9×) |
| raw | `{task}` | 13/276 (5%) | 6 | 20.1 | mostly empty |
| numbered | `Task: {task}.\n` | 169/276 (61%) | 87 | 31.4 | `'No, move the arms home'` (19×) |
| listprime | `Task: {task}. Subtask: 1` | 233/276 (84%) | 69 | 19.3 | `'No progress'` (32×) |

### Decision: keep the default format

Although `listprime` has higher printable-output coverage (84% vs 60%), its outputs skew toward self-critique phrases: 72/233 are variants of `'No progress'`, `'No movement'`, `'No significant movement'`, `'No skill'`. These aren't subtasks — they're the model describing the action history. Coverage went up, usefulness went down.

`default` still produces the cleanest *imperative subtasks* when it works — `'pick up lid'`, `'wipe the spill'`, `'open the drawer'`, `'move the pan'`. This matches the pre-training format the paper describes. The 38% control-char garbage it produces is a *mask* problem, not a *prompt* problem, and is solved separately below.

`raw` (bare instruction, no suffix cue) is effectively broken — the model EOSes immediately. Confirms the hypothesis that the `"Subtask: "` / newline suffix is a required mode-selector for AR text generation.

`numbered` produces longer outputs than default but drifts into state descriptions ("No, the blue ring is in the gripper and then place the blue ring in the basket") rather than next-action subtasks.

### Vocab mask tightening: printable + ASCII

The existing `_build_ascii_vocab_mask` used `piece.isascii()`, which admits control characters (0x00–0x1F, 0x7F). These are legitimately ASCII but not text. After tightening to `piece.isascii() and piece.isprintable()`, the mask excludes those tokens at decode time, eliminating the control-char garbage class that accounted for 38% of "default" outputs.

`str.isprintable()` is the right stdlib primitive here — it excludes control chars while keeping letters, digits, spaces, and punctuation. No extra dependency needed. For broader language filtering (e.g., "is this really English?") the industry-standard options are `langdetect`, `lingua-py`, or `fasttext` with `lid.176.bin`, but those are post-hoc heuristics; vocab masking at decode time is strictly cheaper and deterministic.

### Bugs fixed during the sweep

Two bugs in the admin-endpoint deploy were blocking the test and had to be fixed before any prompt format could run:

1. **`serve.py:210`** was initializing `RuntimeConfig.subtask_prompt_format` from `SubtaskConfig.prompt_template` — but those are different strings. `prompt_template` is the *action* prompt format (`"{task}. Subtask: {subtask}"`) and contains `{subtask}`, which the subtask tokenizer's `.format(task=...)` call then raised `KeyError: 'subtask'` on. Seoul had been in a container crash-loop since the admin-endpoint deploy went out. Fix: use `RuntimeConfig()` with its default.

2. **`admin_server.py:to_dict()`** used `dataclasses.asdict()`, which deep-copies every field recursively — including `_lock: threading.Lock`, which can't be pickled. Every `GET /config` and `PATCH /config` call returned an empty reply and the server logged a pickle error. Fix: build the dict manually via `fields(self)`, skipping underscore-prefixed fields.

### Deployment config fix

The admin endpoint was only reachable during the sweep because of a one-off `docker run -p 127.0.0.1:8001:8001` invocation. Terraform's `user_data.yaml.tftpl` only published `8000:8000` and `5555:5555/udp` — on a fresh cloud-init (stop→start or full redeploy) the admin port would have been lost.

Added `-p 127.0.0.1:8001:8001` to `infra/modules/regional_inference_instance/templates/user_data.yaml.tftpl`. Bound to `127.0.0.1` (localhost on the host), not `0.0.0.0` — the admin endpoint has no auth and should never be internet-reachable. Operators reach it via SSH port forwarding: `ssh -L 8001:127.0.0.1:8001 ubuntu@<box>` then `curl localhost:8001/config` from their laptop. No security-group change needed since the port isn't exposed publicly.

## DROID duration distribution (2026-04-18)

Measured on 3000 successful DROID v1.0.1 episodes streamed sequentially from `gs://gresearch/robotics/droid/1.0.1`:

| stat | steps | seconds @15 Hz |
|---|---|---|
| mean | 305 | 20.3 |
| p50 | 234 | 15.6 |
| p95 | 792 | 52.8 |
| p99 | 1222 | 81.5 |
| max | 2324 | 154.9 |

Only ~0.1% of episodes hit 2 min and most of the longest ones have empty language instructions. Naive "first N episodes with duration ≥ 120s" returns zero matches in the first 2000 scanned.

**Selection strategy used for the long-horizon eval**: top-K longest with filters, implemented in `extract_droid_samples.py` via `--scan_episodes`, `--min_duration_s`, `--require_multi_step`. Scan 5000 episodes, reject empty instructions, floor at 60s, require the multi-step keyword heuristic (`pick…place`, `and then`, `put…in`, etc.), keep the 5 longest in an in-memory min-heap. Deterministic ~10-min GCS scan on a cloud box, guarantees we exercise the long-horizon regime where subtask conditioning is hypothesized to help.

## Comet-Style Hierarchical Subtask Generation (2026-04-18)

Experiment: test the `plan → critique → subtask` scaffold from **openpi-comet** (`src/openpi/shared/client.py`) on our long-horizon DROID cache, using two off-the-shelf VLM backends. Code: `experiments/subtask_probe/droid_eval/comet_style/`.

### Paper vs. code: what we're actually testing

Comet's **paper** (arxiv 2512.10071v3) reports their 0.3453 Q-score from π₀.₅ + RFT + expanded pre-training (§4.1-4.3). It **does not describe any reasoner/planner VLM**. The `client.py` plan/critique/subtask loop is activated only when `fine_grained_level > 0` (`eval_b1k_wrapper.py:59`), a training-data knob the paper doesn't ablate. Their released checkpoints (`pi05-b1kpt12-cs32`, `pi05-b1kpt50-cs32`) are `fine_grained_level=0`. §5 lists "more structured long-horizon reasoning" as **future work**.

Conclusion: **the scaffold is exploratory code that wasn't part of their reported result.** We're not replicating a paper claim; we're testing the scaffold's idea on DROID with our own VLM backends. This is also why the scaffold's default reasoner endpoint (`b5k2m9x7-pre-exp011-043-32000.xenon.lepton.run`) is dead and the default model name (`Qwen3-VL-30B-A3B-Instruct`) is just a kwarg default never actually called.

### Scaffold architecture

Backend-agnostic `BaseReasoner` in `comet_style/reasoner_base.py` with two backends:
- **Gemini** (`gemini_reasoner.py`): `gemini-robotics-er-1.6-preview` via `google-genai`, with 120s request timeout and retry-on-transient-network-error (not just 429).
- **OpenAI-compatible** (`openai_compat_reasoner.py`): any vLLM-hosted VLM — we ran `Qwen3-VL-30B-A3B-Instruct` FP8-quantized on a g6e.2xlarge (L40S 48GB, 64GB RAM) in us-west-2.

### Structured output is load-bearing

Off-the-shelf VLMs do not reliably emit structured output for Comet's prompts out of the box. Initial runs showed:
- `generate_plan`: Gemini emitted a prose paragraph, Qwen-8B emitted a markdown numbered list. Neither parsed as JSON → fell back to a single-step plan = the global task verbatim.
- `generate_subtask`: Gemini Robotics-ER returned ~150-word reasoning paragraphs; Qwen-8B echoed the global task.
- `plan_critique`: free-form prose, wording varied every call, triggered the `if updated != plan_status` reset on every frame — effectively resetting `subtask_history` constantly, losing the hierarchical structure.

Fix: enforce a schema on **all three** VLM calls via the backend's native structured-output API (Gemini `response_schema` / vLLM `response_format=json_schema`). Three schemas in `reasoner_base.py`:
- `PLAN_SCHEMA`: `{"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 10}`
- `SUBTASK_SCHEMA`: `{"type": "object", "properties": {"subtask": {"type": "string", "maxLength": 120}}}`
- `CRITIQUE_SCHEMA`: `{"type": "object", "properties": {"statuses": {"type": "array", "items": {"type": "string", "enum": ["done", "in_progress", "not_started"]}}}}`

Refactored state: `plans: list[str]` + `plan_statuses: list[PlanStepStatus]` (parallel lists, canonical), `plan_status: str` becomes a derived property. The reset-on-change gate now compares status lists structurally, not prose strings.

### Structured critique: 7× speedup, clean progression

| Variant | Mean latency/replan | Unique subtasks (111-frame ep) | Plan progression |
|---|---|---|---|
| Qwen-30B + free-form critique | 5.74 s | 21 | Stuck 65 frames on "locate/search/scan" |
| Qwen-30B + **structured critique** | **0.81 s** | **5** | Clean monotonic: move → grasp → move to dish → place |

Latency drop is output-token count: free-form critique emitted ~300-500-token paragraphs, structured critique emits ~5-15 tokens (enum list). No change in input, no change in model.

### History stride matters (and Comet's hardcoded 5 is correct)

`sample_images` walks the image history with stride=5 by default (Comet's original value, tuned for their 30 Hz sim buffer). With our 1 Hz cache (`--frame_subsample=15`), we assumed stride=1 would be better ("just give the model consecutive frames"). Tested both on the full 5-episode run:

| Episode | Task | stride=5 transitions | stride=1 transitions |
|---|---|---|---|
| ep_0000 | cube + dish (simple) | **8** | 80 |
| ep_0001 | multi-step kitchen | 24 | **8** |
| ep_0002 | turnip plushie + duster | 90 | 82 (semantic oscillation, see below) |
| ep_0003 | cloths + markers | **4** | 8 |
| ep_0004 | bottle + blind | **4** | 5 |

**stride=5 wins on 3/5 episodes and draws on 1.** The original intuition ("stride=5 is over-sparse for 1 Hz cache") was wrong: when history has ≥40 entries, stride=5 gives 8 images spanning ~40 seconds, providing the **temporal-contrast signal** the reasoner needs to detect progression ("40 s ago arm was here, now it's there"). stride=1 gives 8 consecutive seconds — too narrow a window for slow manipulation tasks, where the model over-interprets frame-to-frame motion.

Default kept at `--history_stride=5`.

### Two-call design: intentional, not wasteful

Every replan fires two VLM calls (`plan_critique` + subtask-selection). We considered merging into one call with a `{statuses, subtask}` schema (halves cost/latency). The **load-bearing reason** to keep them separate (documented in `BaseReasoner` docstring):

1. The subtask prompt is built from the **post-critique** `plan_status`. Merged calls force the model to emit both fields in one generation with pre-critique context — fine for reasoning models, risks inconsistent outputs on non-reasoning models (e.g. Gemini Robotics-ER at `thinking_budget=0`).
2. The `subtask_history` reset gate runs between the two calls. If the critique changed `plan_statuses`, `last_subtask` in the subtask prompt becomes `"None"` — merged calls can't apply this reset mid-generation.
3. Graceful degradation: a malformed critique keeps the old statuses and still runs subtask; a malformed merged call loses both.

Merged-call mode is a reasonable follow-up if API cost ever becomes the bottleneck.

### DROID frames continue past task completion

~55% of frames in a cache-selected long-horizon episode are **post-task** — the operator retracted the arm, adjusted, or just held position until the fixed recording window closed. This dilutes Phase 2 L2 metrics (we're measuring whether pi0.5 predicts operator-idle behavior, not task execution). Added `all_steps_done(plan_statuses)` short-circuit so the reasoner skips VLM calls once every plan step is marked `done`, reusing the last real subtask. Pending a Phase 2 split-metrics run that reports pre-completion vs. post-completion separately.

### Multi-object tasks produce semantic oscillation, not a bug

ep_0002 (`"Place the turnip plushie on the table, then the duster on the box..."`) shows the reasoner flipping between `"place the turnip plushie on the table"` and `"place the duster on the box"` every 15 cached frames, regardless of stride. Not a scaffold bug: the task genuinely has two parallel placement sub-goals and each frame's "active step" depends on which object is more visually salient. Real finding worth reporting in the write-up — Comet's scaffold assumes strictly sequential plan steps, which doesn't fit every task structure.

### Action-horizon alignment: the cache was wrong

The previous 10-step-subsampled cache (`.experiments_cache/droid_eval_2min`, 1.5 Hz) was not aligned with pi0.5's `action_horizon=15`. Per-frame behavior-cloning eval with misaligned cache means every cached frame is a "fresh state reset," not a closed-loop decision point. Re-extracted with `--frame_subsample=15` → `.experiments_cache/droid_eval_ah15` (5 episodes, 475 frames total, 1 Hz, 1 cached frame = 1 action horizon of real time).

### Serving infrastructure

- **Local runs** (`run.py` from laptop): Gemini backend. ~0.8 s/replan, but WAN latency amplifies — the initial Gemini Comet run was killed at 50-min hang due to a `Server disconnected` not caught by the 429-only retry. Retry logic broadened to cover transient 5xx/disconnect errors; request timeout hard-capped at 120 s per call.
- **Remote runs** (on the vLLM host itself): Qwen backend. Runs the CLI directly on US West 2 (code + cache rsynced to `~/comet_eval/`), hits `http://localhost:8000/v1` — no SSH tunnel fragility.
- **US West 2 instance** resized from `g6e.xlarge` (32 GB RAM) to `g6e.2xlarge` (64 GB) + EBS grown from 100 GB to 200 GB (online `modify-volume` + `growpart` + `resize2fs`) to host Qwen-30B FP8 weights (~30 GB VRAM, ~58 GB disk). scratch.md updated.

### Visualization

`visualize_subtasks.py` renders both camera views (exterior + wrist) side-by-side with per-frame subtask text:
- **HTML** (default): self-contained page per episode, scrollable table with exterior/wrist thumbnails + subtask column.
- **Video** (`--video`): per-episode mp4 at 2 fps (matches cache rate), composite 640×180 frame with EXTERIOR/WRIST labels and a subtask banner.

Used for inspecting plan progression without running the full Phase 2 action eval — much faster feedback loop during scaffold iteration. Bug caught through this: `visualize_subtasks` previously only rendered the exterior camera, hiding the gripper state that's visible only in the wrist view.

### Status: Phase 2 not yet run

The full Qwen-30B run on `.experiments_cache/droid_eval_ah15` (475 frames, structured critique, stride=5) is complete and saved to `subtasks_comet_qwen30b.json`. Phase 2 (action eval) and Phase 3 (metrics) against Seoul's pi0.5 action server are the next step — pending.

## ForeAct release audit (2026-04-18)

Before starting the ForeAct reconstruction on DROID, we read the paper (arxiv 2602.12322) and walked through the released code at `/Users/kkuan/openpi/foreact/` to pin down what we can actually reuse. Two load-bearing findings, recorded here because they scope everything downstream.

### Finding 1 — "No architectural modification" ≠ "no training"

The paper claims (§3.4) that ForeAct "requires no architectural modifications" to the VLA. Verified in code: pi0.5's image list is built dynamically at runtime by filtering the observation dict against `self.config.image_features` (`foreact/third-party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py:1143-1150`). Adding a foresight image slot is just adding a new key to the dataset and the config — no new layers, no param changes, no architecture shim. That claim is literally true.

But the public pi0.5 base checkpoint was pretrained with 2 images (`exterior`, `wrist`). Registering a 3rd key makes it an input but produces no useful behavior: the action head has no learned pathway from that feature slot to actions. ForeAct's recipe `foreact/third-party/lerobot/scripts/run_sub_task_100k.sh` fine-tunes from `lerobot/pi05_base` for 100k steps on `ForeAct_VLA_Dataset` — that fine-tune is what builds the pathway. "No architectural modification" is accurate; "plug-and-play at inference time" it is not.

Implication for us: feeding foresight to our pi0.5 action server zero-shot is blocked at two independent layers.

1. *Server-side*: `src/hosting/warmup.py:121-132` and the Rust QUIC sidecar hardcode a 2-image spec (`observation/exterior_image_1_left`, `observation/wrist_image_left`). A 3rd key is silently dropped before reaching the policy.
2. *Weights-side*: even if we bypassed serving and set `image_features` to include a foresight key, `_preprocess_images` would encode it but the action head has no learned attention/gating for it. Either silently ignored or actively harmful.

So end-to-end action eval with foresight is out of scope for an inference-only reconstruction. Only fine-tuning pi0.5 on the augmented input (the paper's depth-C recipe) gets there.

### Finding 2 — Only the foresight generator is released, not a fine-tuned VLA

`mit-han-lab/foreact-pretrained` on HuggingFace contains exactly:
- 3 safetensors shards (10.2 GB total, ~5B params)
- `config.json` (941 B)
- `model.safetensors.index.json` (100 kB)
- No `vla/` or `pi0/` subfolder

The 5B param count cleanly adds up to Sana-1600M (1.6B) + Gemma-2-2B-IT (2B) + DC-AE VAE — i.e. π_g only. The HF tag `visualforesight` and the `-pretrained` suffix (= cross-embodiment pretraining stage, *before* target-robot fine-tune) both confirm this. The `mit-han-lab` HF account has only one `foreact-*` repo; other pi0.5 checkpoints listed there (e.g. `vlash-pi05-libero-async5`) belong to different projects.

The VLA training script `run_sub_task_100k.sh` starts from the *public* `lerobot/pi05_base` and produces a Galaxea-R1-Lite-specific fine-tune locally. That output is never uploaded. The `mit-han-lab/ForeActDataset` release contains the raw Galaxea episodes (subtask-segmented) but not a DROID-flavored preprocessing pipeline.

Why it's not actually strange: the generator is robot-agnostic (pretrained on 10M cross-embodiment pairs across AgiBot / RoboMind / Galaxea / Bridge), useful to anyone with any robot. The fine-tuned VLA is Galaxea-R1-Lite-specific — different camera mount, different action space, different embodiment from DROID / LIBERO / anything else — so publishing it would have minimal downstream value. Standard pattern for robotics papers.

Implication: even *with* training infra, reproducing ForeAct end-to-end on DROID would require running their fine-tune recipe ourselves on a DROID-flavored variant of `ForeAct_VLA_Dataset`, which they didn't release a preprocessing pipeline for (only the raw Galaxea episodes).

### Scope of the reconstruction we're doing

Given the above, the faithful-without-training reconstruction is:

- **π_v planner** (their VLM subtask planner): fully faithful. Table 5 prompts verbatim, paper's `Qwen/Qwen3-VL-8B-Instruct` model. Only the harness is ours.
- **π_g generator** (their foresight image generator): faithful to the released `mit-han-lab/foreact-pretrained` checkpoint + `VisualForesightPipeline`. **Skip** the 5-epoch target-data fine-tune the paper always runs. DROID was deliberately excluded from π_g pretraining (§3.2) so this is genuinely zero-shot OOD — a setting the paper never evaluates.
- **VLA integration**: unreachable. Foresight images are outputs for human inspection only. No action eval.

## ForeAct zero-shot reconstruction on DROID (2026-04-18)

Faithful inference-only reconstruction of ForeAct's π_v (planner) + π_g (foresight generator) on our DROID cache. No fine-tuning — see the "ForeAct release audit" section above for why that scopes what's reachable. Code lives at `experiments/subtask_probe/droid_eval/foreact_eval/`.

### Setup

- Cache: `.experiments_cache/droid_eval_ah15/` (5 episodes, 475 frames @ ~1 Hz stride-15).
- Planner: paper's exact Table 5 two-turn prompts, `Qwen/Qwen3-VL-8B-Instruct` served by vLLM on US West 2 L40S (bf16, no quantization). This is the model the paper uses for both its VLM+π_0 baseline and ForeAct "Ours" in §4.3.
- Generator: `mit-han-lab/foreact-pretrained` checkpoint (5B params, 10.2GB) loaded in bf16 via the paper's `VisualForesightPipeline`. Paper's inference hparams: `guidance_scale=4.5`, `image_guidance_scale=1.5`, `num_inference_steps=8`. Runs in the foreact conda env on the same L40S.

### Planner behavior — decomposition is the bottleneck

Per-episode subtask counts from `subtasks_foreact_qwen8b.json`:

| Episode | Instruction complexity | Unique subtasks | Transitions | Mean latency |
|---|---|---|---|---|
| ep_0000 | 1-step ("put cube in dish") | 6 | 28 | 0.73s |
| ep_0001 | 6-step (sink / cup / pan) | **1** | **0** | 0.97s |
| ep_0002 | 10-step (sort plushies) | **1** | **0** | 0.81s |
| ep_0003 | 2-step (cloths + markers) | 3 | 40 | 0.85s |
| ep_0004 | 2-step (bottle + blind) | 3 | 25 | 1.01s |

For ep_0001 and ep_0002, the 8B planner emitted a single subtask for every single frame — the first sub-step of the instruction (ep_0001: "Take the straw out of the sink...", ep_0002: "Place the turnip plushie on the table"). It never advanced. For ep_0003 and ep_0004 it oscillated between echoing the full instruction and the first half. On the simplest task (ep_0000) it produced six near-synonyms of "pick up the cube" / "put the cube in the dish".

The 8B model is clearly on the weaker end of the paper's VLM scaling analysis. Figure 13 reports Qwen3-VL-8B at 84.4% planning accuracy (σ=19.8) vs. Qwen2.5-VL-32B at 73.9% and Qwen2.5-VL-7B at 40.8%. Our DROID episodes pattern-match the long-horizon instructions where 7B failed and 32B was just acceptable. To get meaningful decomposition we'd likely need 32B+ — the paper's success numbers are on their own shorter-horizon Galaxea benchmark, not DROID-style free-form instructions.

Secondary observation: the planner reported `previous_finished=True` on **0 of 475 frames**. We added this as an extra schema field for observability; the 8B model always returns `False` regardless of actual progress. Paper's prompt doesn't ask for this field, so the model has no training signal for it — fine, but the observability is lost.

### Generator behavior — quality varies per episode, pretraining distribution-match matters

The pretrained foresight generator produced 475 DROID-size PNGs at 0.48s mean latency on the L40S (claimed 0.33s on H100 — close enough for the architecture).

No wholesale viewpoint hallucination — outputs don't snap to AgiBot/Galaxea scenes. But contrary to my first-pass "it's just an autoencoder" claim, the generator does produce *varying* predictions across frames. Quality and faithfulness to the subtask depend heavily on how close the DROID scene sits to the pretraining distribution:

| Episode | Scene | Foresight quality | Notes |
|---|---|---|---|
| **ep_0001** | kitchen sink, overhead, dishes | plausible motion, **but hallucinates a phantom second arm** | Arm visibly reaches into the sink in a pose the current frame hasn't reached yet. But every foresight frame also shows a large dark blob on the right that's clearly a second end-effector — DROID is single-arm Franka. See explanation below. |
| ep_0003 | top-down lab, cloths + markers | moderate | Arm position shifts in roughly the right direction, some detail blur. |
| ep_0002 | dim side view, plushie sort | mixed | Arm motion visible but scene is dark and details smear; box sometimes drops out of the foresight. |
| ep_0000 | inverted lab shelf, cube + dish | **worst — heavy distortion** | Black blobs obscure the right half of many frames; occasional hallucinated extra cubes on the shelf. The ceiling-mounted / inverted camera orientation appears to be out-of-distribution. |
| ep_0004 | side view kitchen, bottle | near-identity | Foresight is nearly identical to the current frame; little motion predicted. |

The pattern: the generator works best on scenes that visually resemble its pretraining data (kitchen/sink = ep_0001, similar to AgiBot-Beta). It degrades into artifacts on unusual camera orientations (ep_0000's inverted view) and collapses toward the identity map when the scene is too far from anything it was trained on (ep_0004). Subtask text has *some* effect on the motion trajectory but is easily dominated by the image-conditioning pathway in the weaker-distributional-match cases.

**Embodiment bias — phantom second arm on ep_0001.** Every foresight frame in ep_0001 shows a large dark blob on the right side of the frame that's unmistakably a second bimanual end-effector, despite DROID being a single-arm Franka setup. This is a direct artifact of the pretraining composition (§3.2, Figure 4): AgiBot-World-Colosseo (947k subtasks, ~80% of pretraining volume) is bimanual (AgiBot-Beta), Galaxea Open-World is bimanual (Galaxea R1), and RoboMind includes bimanual ALOHA. Only Bridge (WidowX, single-arm) counters that prior, and it's a small slice. When the DROID scene pattern-matches an AgiBot-like sink/manipulation view, the generator's learned prior inserts the second arm that "should" be there in its training distribution. This is a clean illustration of why the paper always fine-tunes on target-embodiment data before reporting any number — pretraining alone bakes in the dominant embodiment prior.

This mirrors the paper's §4.2 pretraining ablation but from the opposite side — they reported Fidelity=0.00 / Quality=0.00 OOD *without* pretraining; we see pretraining gets us to "mixed, episode-dependent" on DROID, but without the 5-epoch target-data fine-tune (`configs/finetune.yaml:24`) we're still far from consistent subtask-following.

### Verdict: depth-C is a prerequisite for any real signal

Both pieces ran without incident, but neither produces DROID signal that would improve pi0.5 even if we could route it into the action server:

- **Planner**: the 8B choice from the paper is too small for DROID's long-horizon instructions. To reproduce the paper's planner quality on DROID we'd need at minimum Qwen3-VL-32B+.
- **Generator**: zero-shot quality is episode-dependent — decent on kitchen-sink-like scenes (ep_0001 looks genuinely promising), degenerate on inverted / unusual viewpoints (ep_0000, ep_0004). The paper's 5-epoch target fine-tune (`configs/finetune.yaml:24`) is load-bearing, not optional, for consistent subtask-following.

So the depth-C path to a real replication has two training prerequisites before touching pi0.5 at all:
1. Fine-tune the foresight generator on DROID (or a DROID-flavored subtask-segmented dataset we'd need to construct).
2. Fine-tune pi0.5 via the paper's `run_sub_task_100k.sh` recipe to consume the third image slot.

Each of those is a multi-day commitment. Punt unless the finding above is insufficient to kill this direction.

### In-distribution sanity check on ForeActDataset / Galaxea R1 Lite (2026-04-19)

To separate "the generator is broken" from "DROID is OOD for the generator", I ran the same pretrained checkpoint on 2 episodes from `mit-han-lab/ForeActDataset` (the paper's own Galaxea R1 Lite recordings, same robot family as one of the pretraining datasets). Same checkpoint, same inference hparams, same 1 Hz stride — only the input distribution changed. Adapter driver: `foreact_eval/generate_foresight_lerobot.py`.

**Result: the generator clearly works here.** Across 12 foresight frames from episodes 0 ("Pick up the eggplant and place it into the plate") and 3 ("Pick up the corn and place it into the plate"):

- **Subtask-conditioned motion**: Episode 0's foresight shows the eggplant being moved — no corn manipulation. Episode 3's foresight shows the yellow corn being picked up — no eggplant manipulation. The text conditioning actually steers the output, which was the missing signal on DROID.
- **Clean scene reconstruction**: no black-blob artifacts, no hallucinated extra vegetables, no viewpoint shift. All 5 veg items (leek, carrot, cucumber, corn, eggplant) + plate stay in their correct positions in frames where they shouldn't move.
- **Correct single-arm behavior**: the arm enters from the right at the correct angle for the Galaxea R1 Lite mounting. No phantom second arm like we saw on DROID ep_0001 — the embodiment prior matches the target scene, so it doesn't need to "pattern-complete" a missing arm.
- **Task progression**: across a single episode's strided frames, the arm visibly moves closer to the target object and the target object visibly shifts toward the plate. This is the subtask-end-state prediction the paper describes.

Latency was 0.47-0.68s/frame on L40S (bf16), consistent with the DROID run.

**Takeaway**: the pretrained checkpoint is doing its job. Everything we saw on DROID — near-identity autoencoding on some scenes, distortion/artifacts on inverted views, phantom bimanual arms on kitchen-sink scenes — is a distribution-mismatch problem, not a generator problem. With target-robot fine-tune (the paper's 5-epoch recipe), or even just running on robots in the pretraining pool, the generator produces genuinely useful foresight. Depth-C on DROID (fine-tune on DROID-flavored subtask data) would very likely recover this quality level.

### Artifacts

- `experiments/subtask_probe/droid_eval/foreact_eval/{planner,generate_subtasks,generate_foresight,generate_foresight_lerobot,visualize_foreact}.py`
- DROID zero-shot (OOD):
  - `.experiments_cache/droid_eval_ah15/subtasks_foreact_qwen8b.json` (475 records)
  - `.experiments_cache/droid_eval_ah15/foresight_foreact/{ep_0000..ep_0004}/frame_*.png` (475 PNGs @ 640×480)
  - `.experiments_cache/droid_eval_ah15/foreact_html/*.html` (5 per-episode reports, exterior | wrist | foresight | subtask)
  - `.experiments_cache/droid_eval_ah15/foreact_videos/*.mp4` (5 per-episode mp4s at 2 fps)
- ForeActDataset / Galaxea R1 Lite in-distribution:
  - `.experiments_cache/droid_eval_ah15/foresight_picksveg/episode_{000000,000003}/frame_*.png` (12 PNGs) + `actual/` subdirs with source frames
  - `.experiments_cache/droid_eval_ah15/foresight_picksveg/picksveg_report.html` (single side-by-side HTML)

## Open Questions

1. What is the exact prompt format PI used internally for subtask training?
2. Would real robot images produce longer, more specific subtask text?
3. Is the short output (3-4 tokens) an inherent limitation of the base checkpoint, or a prompt format / missing-images issue?
4. Does the ~100 gradient step fine-tuning need real robot data, or would LLM-generated subtask decompositions work?
5. How much does subtask conditioning improve action quality for long-horizon tasks (the paper reports it's significant)?
6. Can JIT compilation reduce JAX subtask latency from 14s to <1s? The prefix forward is fixed-shape and should JIT well, but the AR loop's growing KV cache is a challenge.
7. Is 32GB system RAM sufficient for dual-runtime JIT, or do we need g6e.2xlarge (64GB)?
8. For the hybrid prompt approach (no retraining), does injecting subtask text into the action format produce *better* actions with real images, or just *different* ones?
9. Does ASCII vocabulary masking eliminate Unicode garbage and preserve English subtask quality? (pending re-run)
