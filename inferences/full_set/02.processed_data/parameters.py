"""Shared configuration for the binary calculation scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

SET_SIZE = 268

TOP_K_VALUES = (3, 5, 7, 10, 20)
SCENARIOS = ("norag", *(f"k={k}" for k in TOP_K_VALUES))
ABLATION_SCENARIOS = ("norag", "k=3", "k=5", "k=7")

PROMPT_NAMES = {"1": "Baseline", "2": "Provision", "3": "Labels"}

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261009

CLAIM_COMPARISONS = (
    ("prompt_1_vs_prompt_2_k3", "1", 3, "2", 3, 3),
    ("prompt_2_vs_prompt_3_k5", "2", 5, "3", 5, 5),
    ("prompt_2_k3_vs_k5", "2", 3, "2", 5, 3),
)
