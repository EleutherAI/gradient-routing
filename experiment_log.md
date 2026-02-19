# Gradient Routing (ERA) Experiment Log

**Paper**: *Gradient Routing: Masking Gradients to Localize Computation in Neural Networks* (arXiv 2410.04332)
**Repo**: [kxcloud/gradient-routing](https://github.com/kxcloud/gradient-routing)
**Date**: 2026-02-19
**Model**: TinyStories-28M (8 layers, 512 d_model, 2048 d_mlp)
**Task**: Unlearn "forest/tree"-related concepts (words: tree, trees, forest, forests, woodland, woodlands)

## Overview

We ran four configurations of the Expand-Route-Ablate (ERA) framework, varying two axes:
- **Training mode**: from-scratch vs post-training (starting from pretrained TinyStories-28M)
- **Masking scheme**: full_seq (all tokens in forget stories) vs concept (only exact forget word positions)

All runs expand d_mlp by 64 (512→576) across layers 0-4, use DDBP masking, and apply per-dimension learning rates (expanded_lr=1.0, original_lr=-0.75 on target data).

## Experiment Matrix

| ID | Training | Masking | ERA Steps | Coherence Steps | LR | SLURM Job | Status |
|----|----------|---------|-----------|-----------------|------|-----------|--------|
| A | From-scratch | full_seq | 12,500 | -1 (auto) | 5e-4 | 2370544 | Complete |
| B | Post-training | full_seq | 5,000 | 500 | 1e-4 | 2370567 | Complete |
| C | From-scratch | concept | 12,500 | -1 (auto) | 5e-4 | 2370659 | Running |
| D | Post-training | concept | 5,000 | 500 | 1e-4 | 2370672 | Running |

## Summary Table

| Model | Forget Loss | Retain Loss | Target Words | Repetition | Act. Selectivity |
|-------|------------:|------------:|-------------:|-----------:|-----------------:|
| Pretrained baseline | 1.201 | 1.273 | 12 | 0.034 | N/A |
| **A: From-scratch full_seq** | **3.323** | 2.088 | **1** | 0.046 | **2.16x** |
| **B: Post-train full_seq** | 1.337 | **1.122** | 6 | **0.027** | 1.37x |

- **Forget Loss** (higher = better unlearning): Model A's forget loss is 2.77x baseline, Model B's is only 1.11x baseline
- **Retain Loss** (lower = better preservation): Model B actually *beats* baseline (1.122 vs 1.273). Model A degrades to 2.088
- **Target Words** in 8 generated samples (lower = better suppression): Model A nearly perfect (1 word), Model B moderate (6, down from 12)
- **Repetition Ratio** (lower = better coherence): Model B best (0.027), Model A comparable to baseline
- **Activation Selectivity** (expanded dim forget/retain ratio, averaged across layers): Model A shows 2.16x selectivity, Model B 1.37x

### Interpretation

**Model A (from-scratch, full_seq)** achieves strong unlearning — it almost completely suppresses forest-related generation (1 target word vs 12 baseline). But this comes at a steep cost: the model generates degraded, semi-incoherent text when triggered with forest-related prompts. Retain loss is also significantly worse than baseline (2.088 vs 1.273), meaning general capability is impaired.

**Model B (post-training, full_seq)** is the better practical outcome. It preserves general capability (retain loss *better* than baseline, likely because the coherence fine-tuning acts as additional training), maintains fluent generation, and still achieves some forget suppression (6 words vs 12). However, the unlearning is much weaker — the model still generates forest-related content in some prompts (e.g., "The little bear lived in the forest" appears verbatim).

This highlights a fundamental **unlearning-coherence tradeoff**: from-scratch training allows more aggressive gradient routing (the model learns forest representations only in expanded dims), while post-training must carefully disentangle existing representations.

## Detailed Results

### Training Losses (from training output JSON)

| Model | Pre-ablation Forget | Pre-ablation Retain | Post-ablation Forget | Post-ablation Retain |
|-------|--------------------:|--------------------:|---------------------:|---------------------:|
| A: From-scratch full_seq | 1.468 | 1.564 | 3.247 | 2.023 |
| B: Post-train full_seq | 1.086 | 1.079 | 1.346 | 1.105 |

- Model A: ablation increases forget loss by +1.78 (strong routing to expanded dims)
- Model B: ablation increases forget loss by only +0.26 (weak routing — most forget knowledge stays in original dims)

### Coherence Training

| Model | Best Coherence Loss | Step |
|-------|--------------------:|-----:|
| A: From-scratch full_seq | 2.030 | 0 |
| B: Post-train full_seq | 1.081 | 0 |

Both models achieved best coherence at step 0 (coherence fine-tuning did not improve the contracted model).

### Activation Selectivity (Pre-Ablation Models)

Ratio of mean |activation| in expanded dims for forget vs retain data:

**Model A (from-scratch):**

| Layer | Forget Expanded | Retain Expanded | F/R Ratio |
|-------|----------------:|----------------:|----------:|
| 0 | 0.2623 | 0.0877 | 2.99x |
| 1 | 0.1211 | 0.0344 | 3.53x |
| 2 | 0.0564 | 0.0349 | 1.62x |
| 3 | 0.0600 | 0.0436 | 1.38x |
| 4 | 0.0389 | 0.0309 | 1.26x |

**Model B (post-training):**

| Layer | Forget Expanded | Retain Expanded | F/R Ratio |
|-------|----------------:|----------------:|----------:|
| 0 | 0.1036 | 0.0566 | 1.83x |
| 1 | 0.1070 | 0.0741 | 1.44x |
| 2 | 0.0996 | 0.0798 | 1.25x |
| 3 | 0.0936 | 0.0750 | 1.25x |
| 4 | 0.0868 | 0.0820 | 1.06x |

Model A shows much stronger selectivity, especially in early layers (3-3.5x). Model B's selectivity is weaker across all layers, consistent with the observation that post-training can't fully segregate forget representations.

### Token Routing Weights

Frequency-based per-token masking (mask_weight=0 → forget/expanded dims, mask_weight=1 → retain/original dims):

**Strongly routed (mask_weight < 0.5):**
| Token | Mask Weight | Forget Freq/10k | Retain Freq/10k |
|-------|------------:|----------------:|----------------:|
| tree | 0.000 | 99.4 | 0.0 |

**Moderately routed (0.5 ≤ mask_weight < 0.85):**
| Token | Mask Weight | Forget Freq/10k | Retain Freq/10k |
|-------|------------:|----------------:|----------------:|
| bird | 0.586 | 73.1 | 18.7 |
| flew | 0.810 | 10.3 | 3.6 |
| bear | 0.816 | 10.9 | 3.8 |

Only 4 of 226 non-rare tokens get meaningful routing. The routing captures semantic associations: `bird`, `flew`, and `bear` co-occur with forest-themed stories even though they're not in the explicit forget list.

### Generation Samples

**Prompt: "Once upon a time, Timmy went to the forest"**

| Model | Target Words | Sample |
|-------|:------------:|--------|
| Pretrained | 4 | "...to find a new adventure. He walked and walked until he found a big rock..." |
| A: From-scratch | 1 | "...to that he was a not to but he wanted to his his his a not to his now he..." |
| B: Post-train | 1 | "...to pick berries. All of his berries were bright and red and look delicious..." |

**Prompt: "The little bear lived in the"**

| Model | Target Words | Sample |
|-------|:------------:|--------|
| Pretrained | 3 | "...forest. He had a big, furry tail and he loved to play with it..." |
| A: From-scratch | 0 | "...just for a while he was there was on he could not a his new made a mom..." |
| B: Post-train | 1 | "...forest. He was three years old and very excited to explore..." |

**Prompt: "Once upon a time, there was a little girl named Lily"**

| Model | Target Words | Sample |
|-------|:------------:|--------|
| Pretrained | 0 | "...She loved to eat spaghetti with a fork..." (coherent, on-topic) |
| A: From-scratch | 0 | "...She loved to play outside with her friends..." (mostly coherent) |
| B: Post-train | 0 | "...She loved to play outside in the sun..." (fully coherent) |

Key pattern: Model A generates incoherent text when the prompt activates forest-related contexts (the forget knowledge is removed but nothing fills the gap). On neutral prompts, it's closer to normal. Model B generates fluent text regardless but doesn't fully suppress forest content.

## Phase 3: Retraining Resistance

Phase 3 (retraining evaluation) crashed on both completed runs due to a bug in `retraining_evals.py` (sampling more test stories than available: `random.sample(retain_stories, cfg.test_retain_stories)` where `test_retain_stories=1000` > available stories). The saved model weights are intact.

## Notes

- All runs use `WANDB_MODE=disabled` (syncs attempted but don't block)
- System: NVIDIA GH200 120GB, aarch64, CUDA 12.7, torch 2.10.0+cu126
- Conda env: `gradient-routing` (Python 3.11, TransformerLens from commit a52bfac)
- Model storage: `~/team-shard-filesystem/models/`
- Eval script: `projects/tinystories/eval_compare.py`
- Full eval results: `eval_output/eval_comparison.json`

## Pending

- [ ] Concept masking runs (C, D) still in progress — will add results when complete
- [ ] Compare full_seq vs concept masking efficacy
- [ ] Investigate Phase 3 retraining crash and run manually if needed
- [ ] Evaluate whether concept masking produces better coherence (hypothesis: routing only at exact forget word positions should cause less collateral damage)
