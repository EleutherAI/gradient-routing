"""
Evaluate gradient routing: which tokens get routed into expanded dims?

Key analysis:
1. Token frequency masking weights — which tokens (including related ones not
   in the explicit forget list) are identified as forget-associated?
2. Generation comparison — does the ablated model avoid forest-related content?
3. Activation analysis — do expanded dims respond more to forget-related tokens?
"""
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

import factored_representations.model_expansion as model_expansion
import factored_representations.string_utils as string_utils
import factored_representations.utils as utils
import shared_configs.model_store as model_store
from factored_representations import masklib
from projects.tinystories.shared_settings import cfg as experiment_cfg


def analyze_token_routing_weights(
    tokenizer,
    words_to_localize: list[str],
    output_dir: str,
    device: str = "cpu",
    num_stories: int = 25_000,
):
    """Compute per-token routing weights and identify related tokens."""
    print("=" * 60)
    print("ANALYSIS 1: Token routing weights (related tokens)")
    print("=" * 60)

    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "train", max_stories=200_000
    )
    truncated = string_utils.truncate_stories_by_chars(
        all_stories, experiment_cfg.truncate_story_chars_at
    )
    forget_stories, retain_stories = string_utils.split_stories_by_concept(
        truncated, words_to_localize
    )

    _, info = masklib.get_token_freq_masking_rule(
        retain_stories=retain_stories,
        forget_stories=forget_stories,
        num_stories=num_stories,
        truncate_at=None,
        num_synthetic_tokens_retain=20,
        num_synthetic_tokens_forget=1,
        scale=1.0,
        bias=-4.0,
        tokenizer=tokenizer,
        device=device,
    )

    # Build DataFrame indexed by token string
    vocab = tokenizer.batch_decode(
        t.arange(tokenizer.vocab_size, device=device)[:, None]
    )
    df = pd.DataFrame(info, index=vocab)
    retain_wt = len(retain_stories) / (len(retain_stories) + len(forget_stories))
    freq = df.retain_freq * retain_wt + df.forget_freq * (1 - retain_wt)
    df.insert(0, "freq", freq)

    # Filter rare tokens (< 5 per 10k)
    token_base_rate = 1e4
    nonrare = df[df["freq"] > 5 / token_base_rate].copy()
    nonrare = nonrare.sort_values("mask_weight")

    # Print top/bottom routing weights
    print(f"\nTotal vocabulary: {len(df)} tokens")
    print(f"Non-rare tokens (>5 per 10k): {len(nonrare)} tokens")

    print(f"\n--- Top 20 tokens MOST routed to expanded dims (mask_weight ≈ 1) ---")
    print(f"{'Token':<20} {'Forget freq':>12} {'Retain freq':>12} {'Mask weight':>12}")
    print("-" * 60)
    for tok, row in nonrare.tail(20).iloc[::-1].iterrows():
        print(f"{repr(tok):<20} {row.forget_freq*token_base_rate:>12.1f} "
              f"{row.retain_freq*token_base_rate:>12.1f} {row.mask_weight:>12.3f}")

    print(f"\n--- Top 20 tokens LEAST routed (mask_weight ≈ 0, stay in original dims) ---")
    print(f"{'Token':<20} {'Forget freq':>12} {'Retain freq':>12} {'Mask weight':>12}")
    print("-" * 60)
    for tok, row in nonrare.head(20).iterrows():
        print(f"{repr(tok):<20} {row.forget_freq*token_base_rate:>12.1f} "
              f"{row.retain_freq*token_base_rate:>12.1f} {row.mask_weight:>12.3f}")

    # Identify related tokens NOT in explicit forget list
    explicit_tokens = set()
    for word in words_to_localize:
        toks = tokenizer.encode(word)
        for tok_id in toks:
            explicit_tokens.add(tokenizer.decode([tok_id]).strip().lower())
        # Also add common variations
        toks2 = tokenizer.encode(" " + word)
        for tok_id in toks2:
            explicit_tokens.add(tokenizer.decode([tok_id]).strip().lower())

    high_routing = nonrare[nonrare.mask_weight > 0.7]
    related_not_explicit = high_routing[
        ~high_routing.index.str.strip().str.lower().isin(explicit_tokens)
    ]

    print(f"\n--- Related tokens with high routing weight (>0.7) NOT in forget list ---")
    print(f"Explicit forget tokens: {explicit_tokens}")
    print(f"Found {len(related_not_explicit)} related tokens:")
    print(f"{'Token':<20} {'Forget freq':>12} {'Retain freq':>12} {'Mask weight':>12}")
    print("-" * 60)
    for tok, row in related_not_explicit.sort_values("mask_weight", ascending=False).head(30).iterrows():
        print(f"{repr(tok):<20} {row.forget_freq*token_base_rate:>12.1f} "
              f"{row.retain_freq*token_base_rate:>12.1f} {row.mask_weight:>12.3f}")

    # Plot mask weight distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    n, bins, patches = ax.hist(nonrare.mask_weight, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Mask weight (0=retain in original dims, 1=route to expanded dims)")
    ax.set_ylabel(f"Number of non-rare tokens")
    ax.set_title("Token routing weight distribution\n(which tokens get routed to auxiliary parameters?)")

    # Annotate bins with example tokens
    for i, p in enumerate(patches):
        bin_lo, bin_hi = bins[i], bins[i + 1]
        words_in_bin = nonrare[
            (nonrare.mask_weight >= bin_lo) & (nonrare.mask_weight < bin_hi)
        ]
        top_words = words_in_bin.sort_values("freq", ascending=False).head(3).index.tolist()
        if top_words and p.get_height() > 0:
            annotation = "\n".join([repr(w).strip("'") for w in top_words])
            ax.annotate(
                annotation, (p.get_x() + p.get_width() / 2, p.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=6, rotation=45,
            )

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "token_routing_weights.png"), dpi=150)
    print(f"\nSaved routing weight histogram to {output_dir}/token_routing_weights.png")

    # Save full table
    nonrare.to_csv(os.path.join(output_dir, "token_routing_weights.csv"))
    print(f"Saved full token table to {output_dir}/token_routing_weights.csv")

    return info, nonrare


def analyze_generation(
    model: HookedTransformer,
    words_to_localize: list[str],
    output_dir: str,
    label: str = "model",
    prompts: Optional[list[str]] = None,
    n_samples: int = 20,
    max_new_tokens: int = 200,
):
    """Generate stories and check for forget-related content."""
    print(f"\n{'=' * 60}")
    print(f"GENERATION EVAL: {label}")
    print("=" * 60)

    if prompts is None:
        prompts = [
            "Once upon a time, Timmy went to the forest",
            "Once upon a time, there was a little girl named Lily",
            "One day, a boy went for a walk in the",
            "The little bear lived in the",
            "Once upon a time, there was a big",
            "Tom liked to climb",
        ]

    model.eval()
    for prompt in prompts:
        input_ids = model.to_tokens(prompt)
        with t.inference_mode():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50,
                stop_at_eos=True,
            )
        generated = model.tokenizer.decode(output[0], skip_special_tokens=True)
        word_count = string_utils.count_target_words_in_story(generated, words_to_localize)
        print(f"\nPrompt: {prompt!r}")
        print(f"  Target words found: {word_count}")
        print(f"  Generated: {generated[:300]}...")


def analyze_activations(
    model: HookedTransformer,
    words_to_localize: list[str],
    expansion_size: int,
    output_dir: str,
    device,
    num_samples: int = 40,
):
    """Analyze MLP activations in original vs expanded dims for forget/retain data."""
    print(f"\n{'=' * 60}")
    print("ANALYSIS 3: Activation patterns (original vs expanded dims)")
    print("=" * 60)

    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "validation", max_stories=1000
    )
    truncated = string_utils.truncate_stories_by_chars(
        all_stories, experiment_cfg.truncate_story_chars_at
    )
    forget_stories, retain_stories = string_utils.split_and_label_stories_by_concept(
        truncated, words_to_localize
    )

    forget = [story for story, _ in forget_stories[:num_samples]]
    retain = [story for story, _ in retain_stories[:num_samples]]

    input_ids, attention_mask = string_utils.tokenize_batch(
        forget + retain,
        model.tokenizer,
        prepend_bos=True,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
        padding_side="right",
        device=device,
    )

    n_layers_to_analyze = min(5, model.cfg.n_layers)
    fig, axes = plt.subplots(2, n_layers_to_analyze, figsize=(4 * n_layers_to_analyze, 8))
    if n_layers_to_analyze == 1:
        axes = axes[:, None]

    for layer_idx in range(n_layers_to_analyze):
        mlp_post_captured = [None]

        def hook_fn(value, hook, storage=mlp_post_captured):
            storage[0] = value.detach()

        model.add_hook(f"blocks.{layer_idx}.mlp.hook_post", hook_fn)

        with t.inference_mode():
            _ = model.forward(input_ids, stop_at_layer=layer_idx + 1)

        mlp_activations = mlp_post_captured[0]
        model.reset_hooks()

        exp_idx = -expansion_size
        orig_acts = mlp_activations[..., :exp_idx]
        expand_acts = mlp_activations[..., exp_idx:]

        # Average activation magnitude per dim
        forget_orig = orig_acts[:num_samples].abs().mean(dim=(0, 1)).cpu().numpy()
        forget_expand = expand_acts[:num_samples].abs().mean(dim=(0, 1)).cpu().numpy()
        retain_orig = orig_acts[num_samples:].abs().mean(dim=(0, 1)).cpu().numpy()
        retain_expand = expand_acts[num_samples:].abs().mean(dim=(0, 1)).cpu().numpy()

        # Plot
        ax_forget = axes[0, layer_idx]
        ax_retain = axes[1, layer_idx]

        ax_forget.bar(range(len(forget_expand)), forget_expand, alpha=0.7, label="Expanded dims")
        ax_forget.set_title(f"Layer {layer_idx} - Forget data\nExpanded dim activations")
        ax_forget.set_ylabel("Mean |activation|")

        ax_retain.bar(range(len(retain_expand)), retain_expand, alpha=0.7, label="Expanded dims", color="C1")
        ax_retain.set_title(f"Layer {layer_idx} - Retain data\nExpanded dim activations")
        ax_retain.set_ylabel("Mean |activation|")

        print(f"\nLayer {layer_idx}:")
        print(f"  Forget data - orig dims mean: {forget_orig.mean():.4f}, "
              f"expanded dims mean: {forget_expand.mean():.4f}")
        print(f"  Retain data - orig dims mean: {retain_orig.mean():.4f}, "
              f"expanded dims mean: {retain_expand.mean():.4f}")
        ratio = forget_expand.mean() / (retain_expand.mean() + 1e-8)
        print(f"  Expanded dim forget/retain ratio: {ratio:.2f}x")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "activation_analysis.png"), dpi=150)
    print(f"\nSaved activation analysis to {output_dir}/activation_analysis.png")


if __name__ == "__main__":
    device = utils.get_gpu_with_most_memory()
    print(f"Running evaluation on {device=}")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval_output"
    )
    os.makedirs(output_dir, exist_ok=True)

    model_save_dir = "debugging_demix"
    model_save_name = "demix_debug_era"
    expansion_size = 64
    layers_to_mask = [0, 1, 2, 3, 4]

    # ---- Analysis 1: Token routing weights ----
    # Use the model's tokenizer (load a lightweight config to get it)
    config = get_pretrained_model_config(
        experiment_cfg.transformer_lens_model_name, device=device
    )
    temp_model = HookedTransformer(config)
    tokenizer = temp_model.tokenizer

    info, token_df = analyze_token_routing_weights(
        tokenizer, experiment_cfg.words_to_localize, output_dir, device="cpu"
    )
    del temp_model

    # ---- Load the pre-ablation (expanded) model ----
    print(f"\n{'=' * 60}")
    print("Loading pre-ablation (expanded) model...")
    print("=" * 60)

    config = get_pretrained_model_config(
        experiment_cfg.transformer_lens_model_name, device=device
    )
    base_model = HookedTransformer(config)

    era_cfg_expand = {"d_mlp": expansion_size}
    era_cfg_lrs = dict(
        expanded_dim_lr_target=1.0,
        original_dim_lr_target=-0.75,
        expanded_dim_lr_off_target=1.0,
        original_dim_lr_off_target=1.0,
    )

    expanded_model, specs = model_expansion.expand_and_get_mask_specs(
        base_model,
        era_cfg_expand,
        layers_to_mask=layers_to_mask,
        masking_rule=None,
        suppress_warnings=True,
        **era_cfg_lrs,
        weight_initialization_coeff=1.0,
    )

    pre_ablation_path = f"{model_save_dir}/{model_save_name}_pre_ablation"
    model_store.load_weights(expanded_model, pre_ablation_path)
    print(f"Loaded pre-ablation weights from {pre_ablation_path}")

    # ---- Analysis 3: Activation patterns in expanded model ----
    analyze_activations(
        expanded_model,
        experiment_cfg.words_to_localize,
        expansion_size,
        output_dir,
        device,
    )

    # ---- Analysis 2: Generation from ablated (contracted) model ----
    print("\nContracting model (ablating expanded dims)...")
    contracted_model = model_expansion.contract_model(expanded_model, config)

    # Load post-ablation weights (the coherence-finetuned version)
    post_ablation_path = f"{model_save_dir}/{model_save_name}"
    model_store.load_weights(contracted_model, post_ablation_path)
    print(f"Loaded post-ablation weights from {post_ablation_path}")

    analyze_generation(
        contracted_model,
        experiment_cfg.words_to_localize,
        output_dir,
        label="Post-ablation (contracted, coherence-finetuned)",
    )

    # Also generate from a fresh pretrained model for comparison
    print("\nLoading fresh pretrained model for comparison...")
    pretrained = HookedTransformer.from_pretrained(
        experiment_cfg.transformer_lens_model_name, device=device
    )
    analyze_generation(
        pretrained,
        experiment_cfg.words_to_localize,
        output_dir,
        label="Pretrained baseline (no unlearning)",
    )

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)
