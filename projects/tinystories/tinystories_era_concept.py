"""
Per-token gradient routing using the 'concept' masking scheme.

Instead of routing all tokens in a forget story (full_seq), this only
routes gradients at the exact token positions matching the forget words
(tree, forest, woodland, etc.).
"""
import sys
import os

import torch as t

import projects.tinystories.shared_settings as shared_settings
from projects.tinystories.tinystories_era import do_era_training_run
from factored_representations.utils import get_gpu_with_most_memory


if __name__ == "__main__":
    device = get_gpu_with_most_memory()
    print(f"Running per-token (concept) ERA on {device=}")

    DRY_RUN = len(sys.argv) > 1 and sys.argv[1] == "dry_run"

    era_cfg = shared_settings.ERAConfig(
        layers_to_mask=[0, 1, 2, 3, 4],
        to_expand={"d_mlp": 64},
        masking_scheme="concept",  # per-token: only mask at exact forget word positions
        masking_type="ddbp",
        expanded_vs_original_dim_learning_rates=dict(
            expanded_dim_lr_target=1.0,
            original_dim_lr_target=-0.75,
            expanded_dim_lr_off_target=1.0,
            original_dim_lr_off_target=1.0,
        ),
        include_conditional_bias_term=False,
    )

    era_steps = 12_500
    coherence_finetuning = -1
    forget_set_retraining = 40

    model_save_name = "concept_era"
    model_save_dir = "concept_results"

    os.makedirs("data_concept", exist_ok=True)

    run_cfg = shared_settings.RunTypeConfig(
        label="ERAC_concept",
        pretrained_model_to_load=None,  # train from scratch
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

    res_df = do_era_training_run(
        experiment_cfg=shared_settings.cfg,
        run_type_cfg=run_cfg,
        era_cfg=era_cfg,
        random_shuffle_seed=0,
        num_validation_stories=100,
        num_stories_to_retrain=[64],
        device=device,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        overwrite_model_saves=True,
        validation_data_save_dir="data_concept",
        dry_run=DRY_RUN,
    )
    res_df.to_csv(f"data_concept/{model_save_name}_retrain.csv")
