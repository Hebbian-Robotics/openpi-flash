# ForeAct reconstruction (inference-only)

Faithful-as-possible reconstruction of [ForeAct (arxiv 2602.12322)](https://arxiv.org/abs/2602.12322)
on our DROID subtask-probe pipeline. **No training.** See the "ForeAct release
audit" section in `../../FINDINGS.md` for the scope-defining findings — short
version: the paper's released checkpoint is the foresight generator only; the
fine-tuned VLA is not released; and our pi0.5 action server has a fixed
2-image interface so we can't feed foresight to actions zero-shot.

This module covers two of the paper's three components:

| Component | File | Status |
|---|---|---|
| π_v — VLM planner (Table 5 prompts) | `planner.py` + `generate_subtasks.py` | faithful |
| π_g — foresight image generator | `generate_foresight.py` | faithful to released checkpoint, runs on remote GPU |
| VLA with augmented visual input | — | **unreachable without fine-tuning** |

## Phase 1: Planner subtasks

The planner uses the exact Table 5 prompts (initial + follow-up). Per-episode
state is literally just `previous_subtask: str | None`. We enforce a JSON
schema on the VLM output (`{"subtask": str, "previous_finished": bool}`) —
the paper only says "concise and deterministic"; the schema is our addition
for reliability and observability.

```bash
# Local vLLM on US West 2 serving the paper's model (Qwen3-VL-8B-Instruct):
uv run python -m experiments.subtask_probe.droid_eval.foreact_eval.generate_subtasks \
    --samples_dir ./.experiments_cache/droid_eval_ah15 \
    --output ./.experiments_cache/droid_eval_ah15/subtasks_foreact_qwen8b.json \
    --backend openai_compat \
    --base_url http://localhost:8000/v1 \
    --model Qwen/Qwen3-VL-8B-Instruct
```

Backends: `openai_compat` (paper's setup), `gemini` (optional, for comparison
with our prior Comet-Gemini run).

## Phase 2: Foresight image generator

Runs the released `mit-han-lab/foreact-pretrained` checkpoint on our DROID
cache frames + the Phase 1 subtasks. **Must run on a GPU box** inside the
foreact conda env — the `diffusers`/`transformers`/`deepspeed` pins conflict
with our hosting project's deps.

On the remote box (US West 2 L40S):
```bash
# After stopping the Qwen 8B vLLM to free VRAM:
conda activate foreact
huggingface-cli download mit-han-lab/foreact-pretrained --local-dir ~/foreact_ckpt

python generate_foresight.py \
    --samples_dir ~/.experiments_cache/droid_eval_ah15 \
    --subtasks ~/.experiments_cache/droid_eval_ah15/subtasks_foreact_qwen8b.json \
    --output_dir ~/.experiments_cache/droid_eval_ah15/foresight_foreact \
    --checkpoint ~/foreact_ckpt
```

Then rsync the PNGs back to local:
```bash
rsync -av us-west-2-l40s:~/.experiments_cache/droid_eval_ah15/foresight_foreact/ \
    ./.experiments_cache/droid_eval_ah15/foresight_foreact/
```

Paper's recommended inference hparams (from `foreact/app_cli.py`):
`guidance_scale=4.5`, `image_guidance_scale=1.5`, `num_inference_steps=8`.

## Phase 3: Visualization

HTML + mp4 with the third "predicted foresight" image column alongside
exterior / wrist / subtask:

```bash
uv run python -m experiments.subtask_probe.droid_eval.foreact_eval.visualize_foreact \
    --samples_dir ./.experiments_cache/droid_eval_ah15 \
    --subtasks ./.experiments_cache/droid_eval_ah15/subtasks_foreact_qwen8b.json \
    --foresight_dir ./.experiments_cache/droid_eval_ah15/foresight_foreact \
    --output_dir ./.experiments_cache/droid_eval_ah15/foreact_report \
    [--video --fps 2]
```
