from dataclasses import dataclass, field
from pathlib import Path


# Resolve the model snapshot path via the HF cache symlink
MODEL_PATH = str(
    Path.home()
    / ".cache/huggingface/hub/models--EleutherAI--deep-ignorance-unfiltered"
    / "snapshots/c8df368ff247cb90b62e21e1689260701b3ff25a"
)

# Model architecture (from config.json)
D_MODEL = 4096
D_MLP = 16384
N_LAYERS = 32
N_HEADS = 32
D_HEAD = D_MODEL // N_HEADS  # 128


@dataclass
class VirologyERAConfig:
    # Model
    model_path: str = MODEL_PATH

    # Training
    batch_size: int = 2
    grad_accum_steps: int = 8  # effective batch = 16
    truncate_batch_tokens_at: int = 512
    learning_rate: float = 5e-5
    optimizer_kwargs: dict = field(
        default_factory=lambda: dict(betas=(0.9, 0.95), weight_decay=0.1)
    )

    # ERA
    layers_to_mask: list = field(
        default_factory=lambda: list(range(8))  # first 8 of 32
    )
    d_mlp_expansion: int = 128  # 128/16384 = 0.78%
    num_steps_era_training: int = 2000
    num_steps_coherence_finetuning: int = 500
    coherence_lr: float = 5e-5

    # Masking
    masking_scheme: str = "full_seq"
    masking_type: str = "ddbp"
    expanded_dim_lr_target: float = 1.0
    original_dim_lr_target: float = -0.75
    expanded_dim_lr_off_target: float = 1.0
    original_dim_lr_off_target: float = 1.0

    # Data
    forget_dataset: str = "Unlearning/WMDP-Bio-Remove-Dataset"
    retain_dataset: str = "EleutherAI/wikitext_document_level"

    # Eval
    num_coherence_retain_train: int = 64
    num_coherence_retain_test: int = 200

    # W&B
    wandb_project: str = "virology-era-unlearning"

    @property
    def to_expand(self) -> dict:
        return {"d_mlp": self.d_mlp_expansion}

    @property
    def expanded_vs_original_dim_learning_rates(self) -> dict:
        return dict(
            expanded_dim_lr_target=self.expanded_dim_lr_target,
            original_dim_lr_target=self.original_dim_lr_target,
            expanded_dim_lr_off_target=self.expanded_dim_lr_off_target,
            original_dim_lr_off_target=self.original_dim_lr_off_target,
        )
