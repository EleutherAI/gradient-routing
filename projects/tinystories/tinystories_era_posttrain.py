"""
Post-training gradient routing on a fully pretrained TinyStories-28M model.

Instead of training from scratch with routing, this loads the HuggingFace
pretrained model and applies ERA to route forest/tree knowledge into
expanded dimensions, then ablates them.
"""
import sys
import os

import torch as t
from transformer_lens import HookedTransformer

import shared_configs.model_store as model_store
import projects.tinystories.shared_settings as shared_settings
from projects.tinystories.tinystories_era import do_era_training_run
from factored_representations.utils import get_gpu_with_most_memory


def save_pretrained_to_store(model_name: str, save_subpath: str, device):
    """Load HF pretrained model and save its weights to model store."""
    save_path = model_store.MODEL_STORE_PATH / f"{save_subpath}.pt"
    if save_path.exists():
        print(f"Pretrained weights already saved at {save_subpath}.pt, skipping.")
        return
    print(f"Loading pretrained {model_name} from HuggingFace...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model_store.save_model(model, save_subpath)
    del model
    t.cuda.empty_cache()


if __name__ == "__main__":
    device = get_gpu_with_most_memory()
    print(f"Running post-training ERA on {device=}")

    DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "dry_run"

    # Save pretrained model weights to model store so ERA can load them
    pretrained_subpath = "pretrained/TinyStories-28M"
    save_pretrained_to_store(
        shared_settings.cfg.transformer_lens_model_name,
        pretrained_subpath,
        device,
    )

    # ERA config - same architecture as the from-scratch run
    era_cfg = shared_settings.ERAConfig(
        layers_to_mask=[0, 1, 2, 3, 4],
        to_expand={"d_mlp": 64},
        masking_scheme="full_seq",
        masking_type="ddbp",
        expanded_vs_original_dim_learning_rates=dict(
            expanded_dim_lr_target=1.0,
            original_dim_lr_target=-0.75,
            expanded_dim_lr_off_target=1.0,
            original_dim_lr_off_target=1.0,
        ),
        include_conditional_bias_term=False,
    )

    # Lower lr and fewer steps for post-training (model already converged)
    era_steps = 5_000
    coherence_finetuning = 500
    forget_set_retraining = 40

    model_save_name = "posttrain_era"
    model_save_dir = "posttrain_results"

    os.makedirs(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_posttrain"),
        exist_ok=True,
    )

    run_cfg = shared_settings.RunTypeConfig(
        label="ERAC_posttrain",
        pretrained_model_to_load=pretrained_subpath,
        anneal_gradient_mask_weights=False,
        mask_weight_increase_steps=0,
        expand_model=True,
        use_gradient_routing=True,
        forget_data_labeling_percentage=1.0,
        drop_labeled_forget_data=False,
        drop_unlabeled_forget_data=False,
        sort_forget_data_by_label=False,
        num_steps_era_training=era_steps,
        num_steps_coherence_finetuning=coherence_finetuning,
        num_steps_forget_set_retraining=forget_set_retraining,
        l1_coeff=1e-4,
    )

    # Use a lower learning rate for post-training
    posttrain_cfg = shared_settings.SharedExperimentConfig(
        transformer_lens_model_name=shared_settings.cfg.transformer_lens_model_name,
        total_num_stories_to_load=shared_settings.cfg.total_num_stories_to_load,
        batch_size=shared_settings.cfg.batch_size,
        grad_accum_steps=shared_settings.cfg.grad_accum_steps,
        truncate_story_chars_at=shared_settings.cfg.truncate_story_chars_at,
        truncate_batch_tokens_at=shared_settings.cfg.truncate_batch_tokens_at,
        learning_rate=1e-4,  # lower than 5e-4 used in from-scratch training
        decay_learning_rate=True,
        optimizer_kwargs=shared_settings.cfg.optimizer_kwargs,
        words_to_localize=shared_settings.cfg.words_to_localize,
        unlearning_eval_prompt=shared_settings.cfg.unlearning_eval_prompt,
        wandb_project_subname="forest-posttrain",
    )

    res_df = do_era_training_run(
        experiment_cfg=posttrain_cfg,
        run_type_cfg=run_cfg,
        era_cfg=era_cfg,
        random_shuffle_seed=0,
        num_validation_stories=100,
        num_stories_to_retrain=[64],
        device=device,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        overwrite_model_saves=True,
        validation_data_save_dir="data_posttrain",
        dry_run=DRY_RUN,
    )
    res_df.to_csv(f"data_posttrain/{model_save_name}_retrain.csv")
