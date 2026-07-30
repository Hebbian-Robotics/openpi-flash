# DROID Evaluation: Planner+Action vs Action-Only

## Goal

Validate whether injecting JAX-generated subtask text into the pi0.5 action prompt produces **better** actions (closer to ground truth) compared to action-only inference. Previous experiments with zero images proved the mechanism works (subtask text changes actions) but couldn't assess quality.

## Setup

**Models:**
- **Subtask generator (JAX):** `pi05_base` checkpoint — generates subtask text via AR decoding
- **Action generator (PyTorch):** `swatery/pi05_droid_base` on HuggingFace — DROID-finetuned pi0.5

**Data:** DROID v1.0.1 episodes streamed from GCS (`gs://gresearch/robotics/droid/1.0.1`). Each episode provides real robot images, proprioceptive state, language instructions, and ground truth actions.

## Pipeline

```
Phase 0: extract_droid_samples.py
  Stream DROID episodes from GCS → cache frames + ground truth actions to .npz
  Two modes: first-K (default) or top-K longest (--scan_episodes N)

Phase 1: generate_subtasks.py  (pi0.5 server) or generate_subtasks_gemini.py (Gemini)
  Emit subtask text for each cached frame → JSON

Phase 2: run_action_eval.py
  Load PyTorch DROID checkpoint → run 3 prompt conditions per frame:
    A. Baseline:  "Task: X, State: S;\nAction: "
    B. Hybrid A:  "Task: X. Subtask: Y, State: S;\nAction: "
    C. Hybrid B:  "Task: X (Y), State: S;\nAction: "

Phase 3: compute_metrics.py
  Compare predicted actions to ground truth → L2 distance, cosine sim, per-dim MAE
```

## Prerequisites

```bash
# On the Seoul L40S instance:

# 1. Install RLDS dependencies (in the openpi project root)
cd ~/openpi
uv sync --group rlds

# 2. Download DROID norm stats
gsutil cp -r gs://openpi-assets/checkpoints/pi05_droid/assets/droid/ \
  ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid/assets/droid/

# 3. Download the PyTorch DROID checkpoint
hf download swatery/pi05_droid_base

# 4. Ensure JAX base checkpoint is available
uv run python -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_base')"
```

## Running

```bash
cd ~/openpi/hosting

# Phase 0 (quick): Extract the first 10 successful episodes (short demos, ~30–60 min).
uv run python -m experiments.subtask_probe.droid_eval.extract_droid_samples \
  --num_episodes 10 \
  --output_dir ./.experiments_cache/droid_eval

# Phase 0 (long-horizon eval): Scan 5k DROID episodes and keep the 5 longest
# multi-step ones (>=60s, keyword-matched). This is the setup used to measure
# whether subtask conditioning helps on genuinely long tasks. Streams from GCS,
# ~10 min on a cloud box.
uv run python -m experiments.subtask_probe.droid_eval.extract_droid_samples \
  --num_episodes 5 --scan_episodes 5000 \
  --min_duration_s 60 --require_multi_step \
  --output_dir ./.experiments_cache/droid_eval_2min

# Phase 1: Generate subtasks (JAX subtask generator, ~1s per frame warm)
uv run python experiments/subtask_probe/droid_eval/generate_subtasks.py \
  --samples_dir ./.experiments_cache/droid_eval \
  --output ./.experiments_cache/droid_eval/subtasks.json

# Phase 1 (alt): Generate subtasks via Gemini Robotics-ER instead of pi0.5.
# Requires GEMINI_API_KEY in the environment (or .env). Emits the same JSON
# schema so Phase 2 / 3 consume it unchanged, and pairs cleanly with
# compare_subtask_outputs.py for pi0.5-vs-Gemini diffs.
uv run python experiments/subtask_probe/droid_eval/generate_subtasks_gemini.py \
  --samples_dir ./.experiments_cache/droid_eval \
  --output ./.experiments_cache/droid_eval/subtasks_gemini.json

# Phase 1 (alt 2): Comet-style hierarchical subtask generation.
#
# Runs a stateful plan -> critique -> subtask loop per episode, ported from
# openpi-comet/src/openpi/shared/client.py. Each cached frame issues 2 VLM
# calls (critique + subtask), so expect ~2x the wall clock and API spend of
# the stateless Gemini run. Output JSON schema is identical — drop-in for
# Phase 2 / 3.

# Backend A: Gemini Robotics-ER 1.6 Preview (requires GEMINI_API_KEY).
uv run python -m experiments.subtask_probe.droid_eval.comet_style.run \
  --samples_dir ./.experiments_cache/droid_eval \
  --output ./.experiments_cache/droid_eval/subtasks_comet_gemini.json \
  --backend gemini

# Backend B: OpenAI-compatible VLM (e.g. vLLM hosting Qwen3-VL-30B).
# First, on a GPU host with >=48 GB VRAM, serve the model:
#   uv pip install vllm
#   vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
#     --port 8000 --max-model-len 32768 --limit-mm-per-prompt image=64
# Then from the local machine (tunnel the port if the server is remote):
uv run python -m experiments.subtask_probe.droid_eval.comet_style.run \
  --samples_dir ./.experiments_cache/droid_eval \
  --output ./.experiments_cache/droid_eval/subtasks_comet_qwen.json \
  --backend openai_compat \
  --base_url http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct

# Phase 2: Run action evaluation (~280ms per inference × 3 conditions)
uv run python experiments/subtask_probe/droid_eval/run_action_eval.py \
  --samples_dir ./.experiments_cache/droid_eval \
  --subtasks ./.experiments_cache/droid_eval/subtasks.json \
  --output_dir ./.experiments_cache/droid_eval/predictions

# Phase 3: Compute metrics
uv run python experiments/subtask_probe/droid_eval/compute_metrics.py \
  --samples_dir ./.experiments_cache/droid_eval \
  --predictions_dir ./.experiments_cache/droid_eval/predictions \
  --output ./.experiments_cache/droid_eval/results.json
```

## Metrics

| Metric | What it measures |
|--------|-----------------|
| L2 distance to ground truth | Primary quality signal — are predicted actions closer to what the robot actually did? |
| Per-dimension MAE | Which joints benefit most from subtask conditioning |
| Cosine similarity | Directional alignment independent of magnitude |
| Gripper accuracy | Binary open/closed correctness (threshold 0.5) |

Results are aggregated by:
- Multi-step vs single-step tasks
- Episode progress (early / middle / late)
- Overall

## Interpretation

- **Hybrid < Baseline L2** on multi-step tasks → subtask conditioning helps
- Improvement **only on multi-step**, not single-step → genuine hierarchical decomposition
- **Hybrid A ≠ Hybrid B** → prompt format matters for how the model parses injected text
- No improvement → hybrid prompt injection doesn't work without fine-tuning

## Key Implementation Details

- **Normalization:** Raw DROID joint positions must be z-score normalized before tokenizing into the action prompt. Norm stats from `gs://openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json`.
- **Action horizon:** DROID checkpoint uses `action_horizon=15` (not the base checkpoint's 50).
- **Same noise seed:** All 3 conditions use identical initial noise per frame for fair comparison.
- **Image format:** JAX expects HWC float32 [-1,1]; PyTorch expects CHW float32 [-1,1].
- **DroidOutputs:** Only first 8 dims of 32D model output are meaningful (7 joints + 1 gripper).
