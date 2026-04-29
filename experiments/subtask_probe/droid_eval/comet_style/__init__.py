"""Comet-style hierarchical subtask generation for DROID frames.

Ports the plan -> critique -> subtask loop from
openpi-comet/src/openpi/shared/client.py into a backend-agnostic scaffold with
two concrete reasoner backends: Gemini Robotics-ER and any OpenAI-compatible
VLM server (e.g. a local vLLM hosting Qwen3-VL).

This is evaluation-only code; the live hosting stack is untouched.
"""
