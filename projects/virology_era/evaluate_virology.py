"""
Evaluation for virology ERA models.

Two approaches:
1. Convert TransformerLens state dict back to HF GPT-NeoX format, save, and
   run lm_eval via subprocess (for multi-GPU eval).
2. Custom lm_eval.LM wrapper for TransformerLens HookedTransformer (fallback).

Usage:
    cd /home/a6a/lucia.a6a/gradient-routing
    python projects/virology_era/evaluate_virology.py <state_dict_path> [--hf-convert]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import List, Tuple

import einops
import lm_eval
import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm
from transformer_lens import HookedTransformer

from projects.virology_era.virology_settings import VirologyERAConfig

LM_EVAL_TASKS_PATH = "/home/a6a/lucia.a6a/unlearn/unlearn/lm_eval_tasks"


# ============================================================
# Approach 1: Convert TL weights -> HF GPT-NeoX and use lm_eval CLI
# ============================================================

def convert_tl_to_neox_state_dict(tl_state_dict, cfg):
    """Reverse the TransformerLens convert_neox_weights transformation.

    Maps TransformerLens parameter names back to HuggingFace GPT-NeoX names.
    """
    hf_state_dict = {}

    hf_state_dict["gpt_neox.embed_in.weight"] = tl_state_dict["embed.W_E"]

    for l in range(cfg.n_layers):
        # Layer norms
        hf_state_dict[f"gpt_neox.layers.{l}.input_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln1.w"]
        hf_state_dict[f"gpt_neox.layers.{l}.input_layernorm.bias"] = tl_state_dict[f"blocks.{l}.ln1.b"]
        hf_state_dict[f"gpt_neox.layers.{l}.post_attention_layernorm.weight"] = tl_state_dict[f"blocks.{l}.ln2.w"]
        hf_state_dict[f"gpt_neox.layers.{l}.post_attention_layernorm.bias"] = tl_state_dict[f"blocks.{l}.ln2.b"]

        # QKV: reverse of rearrange(W, "(i qkv h) m -> qkv i m h", i=n_heads, qkv=3)
        W_Q = tl_state_dict[f"blocks.{l}.attn.W_Q"]  # [n_heads, d_model, d_head]
        W_K = tl_state_dict[f"blocks.{l}.attn.W_K"]
        W_V = tl_state_dict[f"blocks.{l}.attn.W_V"]
        W_qkv = torch.stack([W_Q, W_K, W_V], dim=0)  # [3, n_heads, d_model, d_head]
        W_qkv_flat = einops.rearrange(W_qkv, "qkv i m h -> (i qkv h) m")
        hf_state_dict[f"gpt_neox.layers.{l}.attention.query_key_value.weight"] = W_qkv_flat

        # QKV bias: reverse of rearrange(bias, "(index qkv head) -> qkv index head")
        b_Q = tl_state_dict[f"blocks.{l}.attn.b_Q"]  # [n_heads, d_head]
        b_K = tl_state_dict[f"blocks.{l}.attn.b_K"]
        b_V = tl_state_dict[f"blocks.{l}.attn.b_V"]
        b_qkv = torch.stack([b_Q, b_K, b_V], dim=0)  # [3, n_heads, d_head]
        b_qkv_flat = einops.rearrange(b_qkv, "qkv index head -> (index qkv head)")
        hf_state_dict[f"gpt_neox.layers.{l}.attention.query_key_value.bias"] = b_qkv_flat

        # W_O: reverse of rearrange(W_O, "m (i h) -> i h m", i=n_heads)
        W_O = tl_state_dict[f"blocks.{l}.attn.W_O"]  # [n_heads, d_head, d_model]
        W_O_flat = einops.rearrange(W_O, "i h m -> m (i h)")
        hf_state_dict[f"gpt_neox.layers.{l}.attention.dense.weight"] = W_O_flat

        hf_state_dict[f"gpt_neox.layers.{l}.attention.dense.bias"] = tl_state_dict[f"blocks.{l}.attn.b_O"]

        # MLP: reverse .T
        hf_state_dict[f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_in"].T
        hf_state_dict[f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.bias"] = tl_state_dict[f"blocks.{l}.mlp.b_in"]
        hf_state_dict[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight"] = tl_state_dict[f"blocks.{l}.mlp.W_out"].T
        hf_state_dict[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.bias"] = tl_state_dict[f"blocks.{l}.mlp.b_out"]

    # Final layer norm
    hf_state_dict["gpt_neox.final_layer_norm.weight"] = tl_state_dict["ln_final.w"]
    hf_state_dict["gpt_neox.final_layer_norm.bias"] = tl_state_dict["ln_final.b"]

    # Unembed: reverse .T
    hf_state_dict["embed_out.weight"] = tl_state_dict["unembed.W_U"].T

    return hf_state_dict


def save_as_hf_model(tl_state_dict_path: str, output_dir: str, cfg=None):
    """Load a TL state dict, convert to HF format, and save as an HF model."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from projects.virology_era.virology_era import load_neox_model

    if cfg is None:
        cfg = VirologyERAConfig()

    print(f"Loading TL state dict from {tl_state_dict_path}")
    tl_state_dict = torch.load(tl_state_dict_path, map_location="cpu")

    # Load original HF model config and tokenizer
    hf_config = AutoConfig.from_pretrained(cfg.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    # Load the TL model config to get dimensions
    tl_model = load_neox_model(cfg.model_path, device="cpu", dtype=torch.bfloat16)
    tl_cfg = tl_model.cfg
    del tl_model

    # Convert weights
    print("Converting TL -> HF state dict")
    hf_state_dict = convert_tl_to_neox_state_dict(tl_state_dict, tl_cfg)

    # Create HF model and load weights
    print("Creating HF model and loading converted weights")
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
    missing, unexpected = hf_model.load_state_dict(hf_state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"HF model saved to {output_dir}")
    return output_dir


def run_lm_eval_subprocess(model_path: str, tasks: list[str], batch_size: int = 32, num_fewshot: int = 0):
    """Run lm_eval via subprocess for multi-GPU evaluation."""
    results = {}
    for task in tasks:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", task,
            "--batch_size", str(batch_size),
        ]
        if task in ("mmlu",):
            cmd.extend(["--num_fewshot", str(num_fewshot or 1)])
        if task in ("wmdp_bio_robust",):
            cmd.extend(["--include_path", LM_EVAL_TASKS_PATH])
        cmd.extend(["--verbosity", "WARNING"])

        # Write results to temp dir
        output_dir = f"/tmp/lm_eval_results_{task}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        cmd.extend(["--output_path", output_dir])

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout[-2000:] if result.stdout else "")
        if result.returncode != 0:
            print(f"lm_eval failed for {task}:")
            print(result.stderr[-2000:] if result.stderr else "")

        # Parse results from output dir
        task_results = _parse_lm_eval_results(output_dir, task)
        results[task] = task_results

    return results


def _parse_lm_eval_results(output_dir: str, task_name: str) -> dict:
    """Parse lm_eval JSON results from output directory."""
    results = {}
    try:
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".json"):
                    with open(os.path.join(root, f)) as fh:
                        data = json.load(fh)
                    if "results" in data:
                        results = data["results"]
                        break
    except Exception as e:
        print(f"Warning: could not parse results for {task_name}: {e}")
    return results


# ============================================================
# Approach 2: Custom lm_eval.LM wrapper for HookedTransformer
# ============================================================

class HookedTransformerLM(LM):
    """lm_eval wrapper for TransformerLens HookedTransformer models."""

    def __init__(self, model: HookedTransformer, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        self.tokenizer = model.tokenizer

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, int]]:
        outputs = []
        for req in tqdm(requests, desc="loglikelihood"):
            context, continuation = req.args
            context_enc = self.tokenizer.encode(context)
            continuation_enc = self.tokenizer.encode(continuation)
            full_enc = context_enc + continuation_enc

            inp = torch.tensor(full_enc).to(self.device).unsqueeze(0)

            with torch.inference_mode():
                logits = self.model(inp[:, :-1])
                log_probs = F.log_softmax(logits, dim=-1)

                cont_log_probs = log_probs[:, -len(continuation_enc):]
                greedy_tokens = cont_log_probs.argmax(dim=-1)
                cont_toks = torch.tensor(continuation_enc, dtype=torch.long).unsqueeze(0).to(self.device)

                is_top = (greedy_tokens == cont_toks).all()
                gathered = torch.gather(cont_log_probs, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                ll = gathered.sum()

            outputs.append((ll.item(), int(is_top.item())))
        return outputs

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float]]:
        raise NotImplementedError

    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError

    @contextmanager
    def eval_mode(self):
        was_training = self.model.training
        self.model.eval()
        yield
        self.model.train(was_training)


def eval_tl_model_directly(
    state_dict_path: str,
    tasks: list[str],
    device: torch.device,
    cfg: VirologyERAConfig = None,
) -> dict:
    """Evaluate a TL model directly using the HookedTransformerLM wrapper."""
    from projects.virology_era.virology_era import load_neox_model

    if cfg is None:
        cfg = VirologyERAConfig()

    print(f"Loading TL model for direct evaluation")
    model = load_neox_model(cfg.model_path, device=device, dtype=torch.bfloat16)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    wrapped = HookedTransformerLM(model, device)
    with wrapped.eval_mode():
        results = lm_eval.simple_evaluate(
            model=wrapped,
            tasks=tasks,
            include_path=LM_EVAL_TASKS_PATH,
        )
    return results.get("results", {})


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate virology ERA model")
    parser.add_argument("state_dict_path", help="Path to TL state dict .pt file")
    parser.add_argument("--hf-convert", action="store_true",
                        help="Convert to HF and use subprocess lm_eval (for multi-GPU)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for HF model output (default: <state_dict_dir>/hf_model)")
    parser.add_argument("--tasks", nargs="+", default=["wmdp_bio_robust", "mmlu"],
                        help="lm_eval tasks to run")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct TL evaluation (single GPU, no conversion)")
    args = parser.parse_args()

    cfg = VirologyERAConfig()

    if args.direct:
        from factored_representations.utils import get_gpu_with_most_memory
        device = get_gpu_with_most_memory()
        results = eval_tl_model_directly(args.state_dict_path, args.tasks, device, cfg)
        print(json.dumps(results, indent=2))
        return

    if args.hf_convert:
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(args.state_dict_path), "hf_model"
        )
        save_as_hf_model(args.state_dict_path, output_dir, cfg)
        results = run_lm_eval_subprocess(output_dir, args.tasks)
        print(json.dumps(results, indent=2))

        results_path = os.path.join(os.path.dirname(args.state_dict_path), "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
