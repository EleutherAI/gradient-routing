"""
Comprehensive evaluation of gradient routing models.

Evaluates all completed models side-by-side:
- Forget/retain loss (pre- and post-ablation)
- Generation quality: target word suppression + coherence
- Activation analysis: expanded dim selectivity
- Token routing weights

Outputs a structured JSON summary + human-readable report.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

import factored_representations.model_expansion as model_expansion
import factored_representations.string_utils as string_utils
import factored_representations.utils as utils
import shared_configs.model_store as model_store
from factored_representations import masklib
from projects.tinystories.shared_settings import cfg as experiment_cfg


# ── Model configs to evaluate ──────────────────────────────────────────────
MODELS = {
    "from_scratch_fullseq": {
        "label": "From-scratch (full_seq)",
        "model_save_dir": "debugging_demix",
        "model_save_name": "demix_debug_era",
        "json_path": "data_debugging/demix_debug_era_pre_post_ablation.json",
        "masking_scheme": "full_seq",
        "training": "from_scratch",
    },
    "posttrain_fullseq": {
        "label": "Post-training (full_seq)",
        "model_save_dir": "posttrain_results",
        "model_save_name": "posttrain_era",
        "json_path": "data_posttrain/posttrain_era_pre_post_ablation.json",
        "masking_scheme": "full_seq",
        "training": "post_training",
    },
}

EXPANSION_SIZE = 64
LAYERS_TO_MASK = [0, 1, 2, 3, 4]
ERA_LRS = dict(
    expanded_dim_lr_target=1.0,
    original_dim_lr_target=-0.75,
    expanded_dim_lr_off_target=1.0,
    original_dim_lr_off_target=1.0,
)

GENERATION_PROMPTS = [
    "Once upon a time, Timmy went to the forest",
    "Once upon a time, there was a little girl named Lily",
    "One day, a boy went for a walk in the",
    "The little bear lived in the",
    "Once upon a time, there was a big",
    "Tom liked to climb",
    "The animals gathered near the",
    "Lily saw a pretty bird sitting on a",
]


def load_losses(json_path: str) -> dict:
    """Load pre/post ablation losses from training output."""
    with open(json_path) as f:
        return json.load(f)


def load_contracted_model(model_save_dir, model_save_name, config, device):
    """Load the post-ablation (contracted, coherence-finetuned) model."""
    base_model = HookedTransformer(config)
    expanded_model, _ = model_expansion.expand_and_get_mask_specs(
        base_model,
        {"d_mlp": EXPANSION_SIZE},
        layers_to_mask=LAYERS_TO_MASK,
        masking_rule=None,
        suppress_warnings=True,
        **ERA_LRS,
        weight_initialization_coeff=1.0,
    )
    contracted = model_expansion.contract_model(expanded_model, config)
    model_store.load_weights(contracted, f"{model_save_dir}/{model_save_name}")
    return contracted


def load_expanded_model(model_save_dir, model_save_name, config, device):
    """Load the pre-ablation (expanded) model."""
    base_model = HookedTransformer(config)
    expanded_model, _ = model_expansion.expand_and_get_mask_specs(
        base_model,
        {"d_mlp": EXPANSION_SIZE},
        layers_to_mask=LAYERS_TO_MASK,
        masking_rule=None,
        suppress_warnings=True,
        **ERA_LRS,
        weight_initialization_coeff=1.0,
    )
    model_store.load_weights(
        expanded_model, f"{model_save_dir}/{model_save_name}_pre_ablation"
    )
    return expanded_model


def evaluate_generation(model, prompts, words_to_localize, max_new_tokens=200):
    """Generate from model and measure target word occurrence + coherence."""
    model.eval()
    results = []
    total_target_words = 0
    total_prompts = 0

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
        word_count = string_utils.count_target_words_in_story(
            generated, words_to_localize
        )
        # Simple coherence heuristic: count repeated 3-grams
        words = generated.split()
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        unique_trigrams = len(set(trigrams)) if trigrams else 1
        repetition_ratio = 1.0 - (unique_trigrams / max(len(trigrams), 1))

        results.append({
            "prompt": prompt,
            "target_words": word_count,
            "generated_length": len(generated),
            "repetition_ratio": repetition_ratio,
            "generated_text": generated[:400],
        })
        total_target_words += word_count
        total_prompts += 1

    return {
        "per_prompt": results,
        "total_target_words": total_target_words,
        "mean_target_words": total_target_words / total_prompts,
        "mean_repetition_ratio": np.mean([r["repetition_ratio"] for r in results]),
    }


def evaluate_activations(model, words_to_localize, device, num_samples=40):
    """Measure expanded dim selectivity: forget vs retain activation ratio."""
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

    input_ids, _ = string_utils.tokenize_batch(
        forget + retain,
        model.tokenizer,
        prepend_bos=True,
        truncate_at=experiment_cfg.truncate_batch_tokens_at,
        padding_side="right",
        device=device,
    )

    layer_results = {}
    for layer_idx in range(min(5, model.cfg.n_layers)):
        captured = [None]

        def hook_fn(value, hook, storage=captured):
            storage[0] = value.detach()

        model.add_hook(f"blocks.{layer_idx}.mlp.hook_post", hook_fn)
        with t.inference_mode():
            model.forward(input_ids, stop_at_layer=layer_idx + 1)
        acts = captured[0]
        model.reset_hooks()

        exp_idx = -EXPANSION_SIZE
        orig_acts = acts[..., :exp_idx]
        expand_acts = acts[..., exp_idx:]

        forget_orig = orig_acts[:num_samples].abs().mean().item()
        forget_expand = expand_acts[:num_samples].abs().mean().item()
        retain_orig = orig_acts[num_samples:].abs().mean().item()
        retain_expand = expand_acts[num_samples:].abs().mean().item()

        ratio = forget_expand / (retain_expand + 1e-8)
        layer_results[f"layer_{layer_idx}"] = {
            "forget_orig_mean": round(forget_orig, 4),
            "forget_expand_mean": round(forget_expand, 4),
            "retain_orig_mean": round(retain_orig, 4),
            "retain_expand_mean": round(retain_expand, 4),
            "forget_retain_ratio": round(ratio, 2),
        }

    return layer_results


def evaluate_validation_loss(model, words_to_localize, device, num_samples=200):
    """Compute forget/retain loss on held-out validation data."""
    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "validation", max_stories=2000
    )
    truncated = string_utils.truncate_stories_by_chars(
        all_stories, experiment_cfg.truncate_story_chars_at
    )
    forget_stories, retain_stories = string_utils.split_and_label_stories_by_concept(
        truncated, words_to_localize
    )

    forget = [s for s, _ in forget_stories[:num_samples]]
    retain = [s for s, _ in retain_stories[:num_samples]]

    model.eval()
    losses = {}
    for name, stories in [("forget", forget), ("retain", retain)]:
        batch_losses = []
        batch_size = 40
        for i in range(0, len(stories), batch_size):
            batch = stories[i : i + batch_size]
            input_ids, attention_mask = string_utils.tokenize_batch(
                batch,
                model.tokenizer,
                prepend_bos=True,
                truncate_at=experiment_cfg.truncate_batch_tokens_at,
                padding_side="right",
                device=device,
            )
            with t.inference_mode():
                logits = model(input_ids)
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = t.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=model.tokenizer.pad_token_id
                    if model.tokenizer.pad_token_id is not None
                    else -100,
                )
                batch_losses.append(loss.item())
        losses[name] = round(np.mean(batch_losses), 4)

    return losses


if __name__ == "__main__":
    device = utils.get_gpu_with_most_memory()
    print(f"Running comparative evaluation on {device=}")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval_output"
    )
    os.makedirs(output_dir, exist_ok=True)

    words = experiment_cfg.words_to_localize
    config = get_pretrained_model_config(
        experiment_cfg.transformer_lens_model_name, device=device
    )

    all_results = {}

    # ── Evaluate pretrained baseline ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATING: Pretrained baseline (no unlearning)")
    print("=" * 70)

    pretrained = HookedTransformer.from_pretrained(
        experiment_cfg.transformer_lens_model_name, device=device
    )
    pretrained_gen = evaluate_generation(pretrained, GENERATION_PROMPTS, words)
    pretrained_loss = evaluate_validation_loss(pretrained, words, device)

    all_results["pretrained_baseline"] = {
        "label": "Pretrained baseline",
        "validation_loss": pretrained_loss,
        "generation": {
            "total_target_words": pretrained_gen["total_target_words"],
            "mean_target_words": round(pretrained_gen["mean_target_words"], 2),
            "mean_repetition_ratio": round(
                pretrained_gen["mean_repetition_ratio"], 3
            ),
        },
        "generation_samples": pretrained_gen["per_prompt"],
    }
    print(f"  Forget loss: {pretrained_loss['forget']}")
    print(f"  Retain loss: {pretrained_loss['retain']}")
    print(f"  Target words in generation: {pretrained_gen['total_target_words']}")
    print(f"  Mean repetition ratio: {pretrained_gen['mean_repetition_ratio']:.3f}")
    del pretrained
    t.cuda.empty_cache()

    # ── Evaluate each gradient-routed model ────────────────────────────────
    for model_key, mcfg in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {mcfg['label']}")
        print("=" * 70)

        # Load training losses
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", mcfg["json_path"]
        )
        if os.path.exists(json_path):
            training_losses = load_losses(json_path)
        else:
            print(f"  WARNING: {json_path} not found, skipping training losses")
            training_losses = None

        # Load contracted (post-ablation) model
        print(f"  Loading contracted model from {mcfg['model_save_dir']}...")
        contracted = load_contracted_model(
            mcfg["model_save_dir"], mcfg["model_save_name"], config, device
        )

        # Generation eval
        print("  Running generation eval...")
        gen_results = evaluate_generation(contracted, GENERATION_PROMPTS, words)

        # Validation loss eval
        print("  Computing validation losses...")
        val_loss = evaluate_validation_loss(contracted, words, device)

        del contracted
        t.cuda.empty_cache()

        # Load expanded (pre-ablation) model for activation analysis
        print(f"  Loading expanded model for activation analysis...")
        expanded = load_expanded_model(
            mcfg["model_save_dir"], mcfg["model_save_name"], config, device
        )

        print("  Running activation analysis...")
        act_results = evaluate_activations(expanded, words, device)

        del expanded
        t.cuda.empty_cache()

        # Compile results
        result = {
            "label": mcfg["label"],
            "config": {
                "masking_scheme": mcfg["masking_scheme"],
                "training": mcfg["training"],
            },
            "training_losses": training_losses,
            "validation_loss": val_loss,
            "generation": {
                "total_target_words": gen_results["total_target_words"],
                "mean_target_words": round(gen_results["mean_target_words"], 2),
                "mean_repetition_ratio": round(
                    gen_results["mean_repetition_ratio"], 3
                ),
            },
            "activation_selectivity": act_results,
            "generation_samples": gen_results["per_prompt"],
        }
        all_results[model_key] = result

        print(f"  Training losses: {training_losses}")
        print(f"  Validation forget loss: {val_loss['forget']}")
        print(f"  Validation retain loss: {val_loss['retain']}")
        print(f"  Target words in generation: {gen_results['total_target_words']}")
        print(f"  Mean repetition: {gen_results['mean_repetition_ratio']:.3f}")
        for layer, acts in act_results.items():
            print(f"  {layer}: forget/retain expanded ratio = {acts['forget_retain_ratio']}x")

    # ── Token routing weight analysis ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("TOKEN ROUTING WEIGHT ANALYSIS")
    print("=" * 70)

    temp_model = HookedTransformer(config)
    tokenizer = temp_model.tokenizer

    all_stories = string_utils.load_dataset_with_split(
        "delphi-suite/stories", "train", max_stories=200_000
    )
    truncated = string_utils.truncate_stories_by_chars(
        all_stories, experiment_cfg.truncate_story_chars_at
    )
    forget_stories_raw, retain_stories_raw = string_utils.split_stories_by_concept(
        truncated, words
    )

    _, info = masklib.get_token_freq_masking_rule(
        retain_stories=retain_stories_raw,
        forget_stories=forget_stories_raw,
        num_stories=25_000,
        truncate_at=None,
        num_synthetic_tokens_retain=20,
        num_synthetic_tokens_forget=1,
        scale=1.0,
        bias=-4.0,
        tokenizer=tokenizer,
        device="cpu",
    )

    import pandas as pd
    vocab = tokenizer.batch_decode(t.arange(tokenizer.vocab_size)[:, None])
    df = pd.DataFrame(info, index=vocab)
    retain_wt = len(retain_stories_raw) / (
        len(retain_stories_raw) + len(forget_stories_raw)
    )
    freq = df.retain_freq * retain_wt + df.forget_freq * (1 - retain_wt)
    df.insert(0, "freq", freq)
    nonrare = df[df["freq"] > 5 / 1e4].sort_values("mask_weight")

    # Identify strongly routed tokens (low mask_weight = routed to expanded/forget dims)
    strongly_routed = nonrare[nonrare.mask_weight < 0.5]
    moderately_routed = nonrare[
        (nonrare.mask_weight >= 0.5) & (nonrare.mask_weight < 0.85)
    ]

    routing_summary = {
        "total_nonrare_tokens": len(nonrare),
        "strongly_routed_count": len(strongly_routed),
        "moderately_routed_count": len(moderately_routed),
        "strongly_routed_tokens": [
            {
                "token": tok.strip(),
                "mask_weight": round(row.mask_weight, 3),
                "forget_freq_per_10k": round(row.forget_freq * 1e4, 1),
                "retain_freq_per_10k": round(row.retain_freq * 1e4, 1),
            }
            for tok, row in strongly_routed.iterrows()
        ],
        "moderately_routed_tokens": [
            {
                "token": tok.strip(),
                "mask_weight": round(row.mask_weight, 3),
                "forget_freq_per_10k": round(row.forget_freq * 1e4, 1),
                "retain_freq_per_10k": round(row.retain_freq * 1e4, 1),
            }
            for tok, row in moderately_routed.iterrows()
        ],
    }
    all_results["token_routing"] = routing_summary

    print(f"  Strongly routed (mw < 0.5): {len(strongly_routed)} tokens")
    for tok, row in strongly_routed.iterrows():
        print(f"    {repr(tok):<20} mw={row.mask_weight:.3f}")
    print(f"  Moderately routed (0.5 <= mw < 0.85): {len(moderately_routed)} tokens")
    for tok, row in moderately_routed.iterrows():
        print(f"    {repr(tok):<20} mw={row.mask_weight:.3f}")

    del temp_model
    t.cuda.empty_cache()

    # ── Save results ───────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "eval_comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print("=" * 70)

    header = f"{'Model':<30} {'Forget↑':>8} {'Retain↓':>8} {'TgtWds↓':>8} {'Repet↓':>8} {'ActRatio':>8}"
    print(header)
    print("-" * len(header))

    for key in ["pretrained_baseline"] + list(MODELS.keys()):
        r = all_results[key]
        forget_l = r["validation_loss"]["forget"]
        retain_l = r["validation_loss"]["retain"]
        tgt = r["generation"]["total_target_words"]
        rep = r["generation"]["mean_repetition_ratio"]

        # Mean activation ratio across layers (only for gradient-routed models)
        if "activation_selectivity" in r:
            ratios = [
                v["forget_retain_ratio"]
                for v in r["activation_selectivity"].values()
            ]
            mean_ratio = np.mean(ratios)
            ratio_str = f"{mean_ratio:.2f}x"
        else:
            ratio_str = "N/A"

        print(
            f"{r['label']:<30} {forget_l:>8.3f} {retain_l:>8.3f} {tgt:>8d} {rep:>8.3f} {ratio_str:>8}"
        )

    print(f"\nForget↑ = higher forget loss is better (model forgot more)")
    print(f"Retain↓ = lower retain loss is better (model preserved knowledge)")
    print(f"TgtWds↓ = fewer target words in generation is better (suppression)")
    print(f"Repet↓  = lower repetition is better (coherence)")
    print(f"ActRatio = expanded dim forget/retain activation ratio (selectivity)")

    print("\nDone!")
