"""
ERA (Expand-Route-Ablate) gradient routing for virology unlearning.

Applies ERA to EleutherAI/deep-ignorance-unfiltered (6.85B GPT-NeoX)
using WMDP Bio Remove Dataset as forget data.

Usage:
    cd /home/a6a/lucia.a6a/gradient-routing
    python projects/virology_era/virology_era.py [dry_run]
"""

import json
import math
import os
import sys
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch as t
import torch.utils.data as data
import tqdm
import wandb
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    get_pretrained_model_config,
    get_pretrained_state_dict,
)

import factored_representations.model_expansion as model_expansion
import factored_representations.training as training
from factored_representations import masklib, string_utils

from projects.virology_era.virology_data import (
    load_retain_data,
    load_training_data,
    load_validation_data,
)
from projects.virology_era.virology_settings import VirologyERAConfig


def load_neox_model(model_path: str, device, dtype=t.bfloat16) -> HookedTransformer:
    """Load a GPT-NeoX model from a local HF path into HookedTransformer.

    Bypasses the official model name check in HookedTransformer.from_pretrained
    by calling the config/state_dict loaders directly (which do support local paths).
    """
    cfg = get_pretrained_model_config(
        model_path,
        device=device,
        dtype=dtype,
        fold_ln=False,
    )

    state_dict = get_pretrained_state_dict(model_path, cfg, hf_model=None, dtype=dtype)

    model = HookedTransformer(cfg, move_to_device=False)
    model.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    model.move_model_modules_to_device()
    return model


def full_seq_mask_rule(labels, seq_length, device):
    """0 = forget set, 1 = retain set."""
    return labels.unsqueeze(1).repeat(1, seq_length).to(device)


def convert_to_labeled_mask_rule(mask_rule, label_setting="retain_always_unmasked"):
    """Wrap a token mask rule to handle (input_ids, labels) pairs."""
    if label_setting == "retain_always_unmasked":
        def unmasked_retain_mask_rule(input_ids_and_labels):
            input_ids, labels = input_ids_and_labels
            original_mask = mask_rule(input_ids)
            is_retain = full_seq_mask_rule(
                labels, input_ids.shape[1] - 1, original_mask.device
            )
            return t.maximum(is_retain, original_mask)
        return unmasked_retain_mask_rule
    else:
        raise ValueError(f"Unknown label setting: {label_setting}")


@t.inference_mode()
def eval_on_validation(
    model,
    validation_data: list[tuple],
    truncate_at: int,
) -> float:
    """Evaluate model loss on validation data."""
    dataloader = data.DataLoader(
        string_utils.ListDataset(validation_data), batch_size=4, shuffle=False
    )
    device = t.device(model.cfg.device)
    batch_losses = []
    for batch in dataloader:
        stories, labels = batch
        tokens, attention_mask = string_utils.tokenize_batch(
            stories,
            model.tokenizer,
            prepend_bos=True,
            truncate_at=truncate_at,
            padding_side="right",
            device=device,
        )
        with t.autocast(device_type="cuda", dtype=t.bfloat16, enabled=device.type == "cuda"):
            loss = training.compute_preds_and_get_ce_loss(
                model, tokens, attention_mask, None
            )
        batch_losses.append(loss.item())
    return sum(batch_losses) / len(batch_losses) if batch_losses else float("inf")


def do_virology_era_run(
    cfg: VirologyERAConfig,
    device: t.device,
    save_dir: str,
    dry_run: bool = False,
):
    """Run full ERA pipeline: expand+route, ablate, coherence finetune."""

    os.makedirs(save_dir, exist_ok=True)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    wandb.login()
    wandb.init(
        project=cfg.wandb_project,
        mode="disabled" if dry_run else "online",
        name=f"virology-era-{'dryrun' if dry_run else 'full'}",
        config=cfg.__dict__,
        settings=wandb.Settings(code_dir=project_dir),
        dir=project_dir,
    )

    # ---- Load model ----
    print("Loading model from", cfg.model_path)
    model = load_neox_model(cfg.model_path, device=device, dtype=t.bfloat16)
    original_model_config = model.cfg
    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}, d_mlp={model.cfg.d_mlp}")

    # ---- Load data ----
    print("Loading data...")
    if dry_run:
        training_data = load_training_data(max_forget=50, max_retain=50, seed=42)
    else:
        training_data = load_training_data(seed=42)

    forget_val, retain_val = load_validation_data(
        n_forget=50 if dry_run else 200,
        n_retain=50 if dry_run else 200,
    )

    # ---- Build token freq mask rule (needed even for full_seq scheme) ----
    # For full_seq with ddbp, we use the token freq mask rule but override with
    # full-sequence labels via convert_to_labeled_mask_rule
    forget_texts = [text for text, label in training_data if label == 0]
    retain_texts = [text for text, label in training_data if label == 1]

    token_freq_kwargs = dict(
        retain_stories=retain_texts[:5000],
        forget_stories=forget_texts[:5000],
        num_stories=min(5000, len(forget_texts), len(retain_texts)),
        truncate_at=None,
        num_synthetic_tokens_retain=20,
        num_synthetic_tokens_forget=1,
        scale=1.0,
        bias=-4.0,
        tokenizer=model.tokenizer,
        device=device,
    )

    token_mask_rule, _ = masklib.get_token_freq_masking_rule(**token_freq_kwargs)
    mask_rule = convert_to_labeled_mask_rule(token_mask_rule, "retain_always_unmasked")

    # ---- ERA setup: expand model ----
    print(f"Expanding model: d_mlp += {cfg.d_mlp_expansion} on layers {cfg.layers_to_mask}")
    model, specs = model_expansion.expand_and_get_mask_specs(
        model,
        cfg.to_expand,
        layers_to_mask=cfg.layers_to_mask,
        masking_rule=mask_rule,
        suppress_warnings=False,
        **cfg.expanded_vs_original_dim_learning_rates,
        weight_initialization_coeff=1.0,
    )
    # expand_model init_weights() creates new dims in float32; cast to match loaded weights
    model = model.to(t.bfloat16)
    print(f"Expanded model: d_mlp={model.cfg.d_mlp}")

    mask_applier = masklib.MaskApplier(
        model,
        specs,
        use_partial_boolean_masks=True,  # full_seq is in SCHEMES_WITH_PARTIAL_WEIGHTS
    )
    # MaskApplier precomputes masks in float32; cast to bfloat16 to match model
    mask_applier.mask_lookup_tensors = [
        m.to(t.bfloat16) for m in mask_applier.mask_lookup_tensors
    ]

    dataloader = data.DataLoader(
        string_utils.ListDataset(training_data),
        shuffle=False,
        batch_size=cfg.batch_size,
    )

    optim = t.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        **cfg.optimizer_kwargs,
    )

    num_training_steps = min(
        len(training_data) // cfg.batch_size,
        cfg.num_steps_era_training,
    )

    def get_lr(it):
        min_lr = cfg.learning_rate / 10
        warmup_iters = 100
        lr_decay_iters = num_training_steps
        if it < warmup_iters:
            return cfg.learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (cfg.learning_rate - min_lr)

    # ---- PHASE 1: Expand and Route ----
    print("PHASE ONE: EXPAND AND ROUTE")
    total_steps = min(cfg.num_steps_era_training, len(dataloader))
    eval_every = 8 if dry_run else 250

    for step, batch in (pbar := tqdm.tqdm(enumerate(dataloader), total=total_steps)):
        if step >= total_steps:
            break

        lr = get_lr(step)
        for param_group in optim.param_groups:
            param_group["lr"] = lr

        stories, labels = batch
        input_ids, attention_mask = string_utils.tokenize_batch(
            stories,
            model.tokenizer,
            prepend_bos=True,
            truncate_at=cfg.truncate_batch_tokens_at,
            padding_side="right",
            device=device,
        )

        with mask_applier((input_ids, labels), mask_weight=1.0):
            with t.autocast(device_type="cuda", dtype=t.bfloat16):
                loss = training.compute_preds_and_get_ce_loss(
                    model, input_ids, attention_mask, None
                )

        loss = loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            optim.step()
            optim.zero_grad()

        effective_loss = loss.item() * cfg.grad_accum_steps
        wandb.log({"loss": effective_loss, "lr": lr, "step": step})
        pbar.set_postfix({"loss": f"{effective_loss:.4f}"})

        if step % eval_every == 0:
            forget_loss = eval_on_validation(model, forget_val[:20], cfg.truncate_batch_tokens_at)
            retain_loss = eval_on_validation(model, retain_val[:20], cfg.truncate_batch_tokens_at)
            wandb.log({
                "validation_forget_loss": forget_loss,
                "validation_retain_loss": retain_loss,
            })
            print(f"  Step {step}: forget_loss={forget_loss:.4f}, retain_loss={retain_loss:.4f}")

    # Pre-ablation eval
    pre_forget = eval_on_validation(model, forget_val, cfg.truncate_batch_tokens_at)
    pre_retain = eval_on_validation(model, retain_val, cfg.truncate_batch_tokens_at)
    pre_ablation = {"forget_loss": pre_forget, "retain_loss": pre_retain}
    print(f"Pre-ablation: {pre_ablation}")
    wandb.log({"pre_ablation_forget_loss": pre_forget, "pre_ablation_retain_loss": pre_retain})

    # Save pre-ablation model
    pre_ablation_path = os.path.join(save_dir, "pre_ablation.pt")
    t.save(model.state_dict(), pre_ablation_path)
    print(f"Pre-ablation model saved to {pre_ablation_path}")

    # ---- PHASE 2: Ablate + Coherence Finetune ----
    print("PHASE TWO: ABLATE AND COHERENCE FINETUNE")

    # Free ERA optimizer/masks before creating contracted model to avoid OOM
    del mask_applier
    del optim
    del specs
    del dataloader
    t.cuda.empty_cache()

    contracted_model = model_expansion.contract_model(model, original_model_config)
    # contract_model creates views into expanded model tensors; clone to break dependency
    for param in contracted_model.parameters():
        param.data = param.data.clone()
    del model
    t.cuda.empty_cache()

    # Post-ablation eval
    post_forget = eval_on_validation(contracted_model, forget_val, cfg.truncate_batch_tokens_at)
    post_retain = eval_on_validation(contracted_model, retain_val, cfg.truncate_batch_tokens_at)
    post_ablation = {"forget_loss": post_forget, "retain_loss": post_retain}
    print(f"Post-ablation: {post_ablation}")
    wandb.log({"post_ablation_forget_loss": post_forget, "post_ablation_retain_loss": post_retain})

    # Save ablation comparison
    with open(os.path.join(save_dir, "ablation_comparison.json"), "w") as f:
        json.dump({"pre_ablation": pre_ablation, "post_ablation": post_ablation}, f, indent=2)

    # Coherence finetuning on retain data (mini-batched for memory)
    retain_for_coherence = load_retain_data(max_examples=1000 if not dry_run else 100)
    rng = np.random.default_rng(42)
    rng.shuffle(retain_for_coherence)
    coherence_train = retain_for_coherence[:cfg.num_coherence_retain_train]
    coherence_test = retain_for_coherence[
        cfg.num_coherence_retain_train :
        cfg.num_coherence_retain_train + cfg.num_coherence_retain_test
    ]

    coherence_dataloader = data.DataLoader(
        string_utils.ListDataset(coherence_train),
        shuffle=True,
        batch_size=cfg.batch_size,
    )

    optim = t.optim.AdamW(
        contracted_model.parameters(),
        lr=cfg.coherence_lr,
        **cfg.optimizer_kwargs,
    )

    best_loss = eval_on_validation(contracted_model, coherence_test, cfg.truncate_batch_tokens_at)
    # Store best weights on CPU to save GPU memory for optimizer states
    best_model_weights = {k: v.cpu().clone() for k, v in contracted_model.state_dict().items()}
    best_step = 0

    coherence_steps = 5 if dry_run else cfg.num_steps_coherence_finetuning
    coherence_iter = iter(coherence_dataloader)
    for step in (pbar := tqdm.tqdm(range(coherence_steps + 1))):
        try:
            batch = next(coherence_iter)
        except StopIteration:
            coherence_iter = iter(coherence_dataloader)
            batch = next(coherence_iter)
        stories, labels = batch
        input_ids, attention_mask = string_utils.tokenize_batch(
            stories,
            contracted_model.tokenizer,
            prepend_bos=True,
            truncate_at=cfg.truncate_batch_tokens_at,
            padding_side="left",
            device=device,
        )
        with t.autocast(device_type="cuda", dtype=t.bfloat16, enabled=device.type == "cuda"):
            loss = training.compute_preds_and_get_ce_loss(
                contracted_model, input_ids, attention_mask, None
            )
        optim.zero_grad()
        loss.backward()
        optim.step()
        wandb.log({"coherence_train_loss": loss.item()})

        if step % 10 == 0 or step == coherence_steps:
            test_loss = eval_on_validation(
                contracted_model, coherence_test, cfg.truncate_batch_tokens_at
            )
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_weights = {k: v.cpu().clone() for k, v in contracted_model.state_dict().items()}
                best_step = step
            wandb.log({"coherence_test_loss": test_loss})

            forget_loss = eval_on_validation(
                contracted_model, forget_val[:20], cfg.truncate_batch_tokens_at
            )
            wandb.log({"coherence_forget_loss": forget_loss})

    print(f"Best coherence loss {best_loss:.4f} at step {best_step}")
    wandb.run.summary["best_coherence_step"] = best_step

    contracted_model.load_state_dict(best_model_weights)

    # Final eval
    final_forget = eval_on_validation(contracted_model, forget_val, cfg.truncate_batch_tokens_at)
    final_retain = eval_on_validation(contracted_model, retain_val, cfg.truncate_batch_tokens_at)
    print(f"Final: forget_loss={final_forget:.4f}, retain_loss={final_retain:.4f}")
    wandb.log({"final_forget_loss": final_forget, "final_retain_loss": final_retain})

    # Save final model
    final_path = os.path.join(save_dir, "final.pt")
    t.save(contracted_model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({
            "pre_ablation": pre_ablation,
            "post_ablation": post_ablation,
            "final": {"forget_loss": final_forget, "retain_loss": final_retain},
            "best_coherence_step": best_step,
        }, f, indent=2)

    wandb.finish()
    return contracted_model


if __name__ == "__main__":
    from factored_representations.utils import get_gpu_with_most_memory

    DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "dry_run"
    device = get_gpu_with_most_memory()
    print(f"Running virology ERA on {device=}, {DRY_RUN=}")

    cfg = VirologyERAConfig()
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "dry_run" if DRY_RUN else "full_run",
    )

    do_virology_era_run(cfg, device, save_dir, dry_run=DRY_RUN)
