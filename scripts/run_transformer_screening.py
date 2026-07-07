#!/usr/bin/env python3
"""Transformer screening pipeline with CE-N, Sobol-N, and baseline initializations on WikiText-2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import math
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import local initialization functions
from copeland_erdos_nets.ce_init import ce_init_
from copeland_erdos_nets.sobol_init import sobol_init_
from copeland_erdos_nets.assignment import compute_effective_rank

# ============================================================================
# Causal Self-Attention & Transformer Architecture
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Key, query, value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # batch size, sequence length, d_model
        
        # Project and reshape: (B, T, C) -> (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # (B, n_heads, T, head_dim) x (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
            
        att = torch.softmax(att, dim=-1)
        
        # (B, n_heads, T, T) x (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        y = att @ v
        
        # Reassemble heads: (B, n_heads, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_seq_len: int = 1024):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Cache causal mask
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                     .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max seq len is {self.max_seq_len}"
        
        # Token and positional embeddings
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Forward through blocks
        mask = self.bias[:, :, :T, :T]
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ============================================================================
# Initialization Applier
# ============================================================================

def apply_init(
    model: nn.Module,
    init_name: str,
    kind: str = "he",
    m: int = 4,
    offset: int = 0,
    scramble_seed: int = 0,
    assignment: str = "sequential",
    orthogonalize: bool = False,
    matrix_shaped: bool = False,
):
    """Apply initialization to Transformer weights."""
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        if init_name in ("ce_n", "ce_u"):
            mode = "uniform" if init_name == "ce_u" else "normal"
            ce_init_(
                module.weight, 
                m=m, 
                kind=kind, 
                offset_blocks=offset, 
                mode=mode, 
                assignment=assignment,
                orthogonalize=orthogonalize
            )
        elif init_name in ("sobol_n", "sobol_u"):
            mode = "uniform" if init_name == "sobol_u" else "normal"
            sobol_init_(
                module.weight, 
                scramble_seed=scramble_seed, 
                kind=kind, 
                mode=mode,
                assignment=assignment,
                matrix_shaped=matrix_shaped
            )
        elif init_name == "xavier":
            nn.init.xavier_normal_(module.weight)
        elif init_name == "he":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _classify_layer(name: str) -> str:
    """Classify a named parameter into a layer category for selective init.
    
    Returns one of: 'attn_qk', 'attn_vo', 'ffn', 'lm_head', 'other'
    """
    if "lm_head" in name:
        return "lm_head"
    if "q_proj" in name or "k_proj" in name:
        return "attn_qk"
    if "v_proj" in name or "out_proj" in name:
        return "attn_vo"
    if "mlp" in name:
        return "ffn"
    return "other"


# CE std targets measured from R008 epoch 1 (for scale-matched controls)
_CE_STD_TARGETS = {
    "attn_qk": 0.12195,
    "attn_vo": 0.12512,
    "ffn": 0.05917,
}


def _apply_init_to_weight(
    weight: torch.Tensor,
    init_spec: str,
    m: int = 8,
    offset: int = 0,
    assignment: str = "lcg",
    layer_category: str = "other",
):
    """Apply a single init spec to a weight tensor.
    
    Supported specs: 'xavier', 'he', 'ce_lcg', 'ce_hash', 'ce_shuf',
    'gaussian_matched', 'xavier_gain_X.X', 'orthogonal', 'uniform_matched'.
    """
    if init_spec == "xavier":
        nn.init.xavier_normal_(weight)
    elif init_spec == "he":
        nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
    elif init_spec.startswith("ce_"):
        # Parse assignment from spec: ce_lcg, ce_hash, ce_shuf
        spec_map = {"ce_lcg": "lcg", "ce_hash": "hash_indexed", "ce_shuf": "shuffled"}
        assign = spec_map.get(init_spec, "sequential")
        ce_init_(
            weight,
            m=m,
            kind="he",
            offset_blocks=offset,
            mode="normal",
            assignment=assign,
        )
    elif init_spec == "gaussian_matched":
        # Gaussian N(0, CE_std) matched per-layer-type
        target_std = _CE_STD_TARGETS.get(layer_category, 0.10)
        nn.init.normal_(weight, mean=0.0, std=target_std)
    elif init_spec.startswith("xavier_gain_"):
        # Xavier with custom gain: e.g. "xavier_gain_1.4"
        gain = float(init_spec.split("xavier_gain_")[1])
        nn.init.xavier_normal_(weight, gain=gain)
    elif init_spec == "orthogonal":
        nn.init.orthogonal_(weight)
    elif init_spec == "uniform_matched":
        # Uniform U(-a, a) with same std as CE: std = a/sqrt(3)
        target_std = _CE_STD_TARGETS.get(layer_category, 0.10)
        a = target_std * math.sqrt(3)
        nn.init.uniform_(weight, -a, a)


def apply_selective_init(
    model: nn.Module,
    rules: dict,
    m: int = 8,
    offset: int = 0,
):
    """Apply selective (per-layer-type) initialization to a Transformer.

    Args:
        model: The Transformer model.
        rules: Dict mapping layer categories to init specs.
            Keys: 'attn_qk', 'attn_vo', 'ffn', 'lm_head'
            Values: 'xavier', 'he', 'ce_lcg', 'ce_hash', 'ce_shuf'
            Example: {"attn_qk": "xavier", "attn_vo": "xavier", "ffn": "ce_lcg", "lm_head": "xavier"}
        m: CE block width (default 8).
        offset: CE stream offset in blocks.
    """
    for name, param in model.named_parameters():
        if "weight" not in name or param.ndim < 2:
            continue

        category = _classify_layer(name)
        init_spec = rules.get(category, rules.get("default", "xavier"))

        _apply_init_to_weight(param.data, init_spec, m=m, offset=offset, layer_category=category)

        # Zero biases for corresponding module
        bias_name = name.replace(".weight", ".bias")
        bias_param = dict(model.named_parameters()).get(bias_name)
        if bias_param is not None:
            nn.init.zeros_(bias_param)



# ============================================================================
# Spectral Metrics
# ============================================================================

def get_weight_and_spectral_stats(model: nn.Module) -> dict:
    """Collect statistics including condition number and effective rank."""
    stats = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.ndim >= 2:
            w = param.data.cpu()
            w_2d = w.view(w.shape[0], -1)
            try:
                s = torch.linalg.svdvals(w_2d).numpy()
                cond = float(s[0] / (s[-1] + 1e-10))
                eff_rank = compute_effective_rank(param)
            except Exception:
                cond = 1.0
                eff_rank = 1.0

            stats[name] = {
                "mean": float(w.mean()),
                "std": float(w.std()),
                "min": float(w.min()),
                "max": float(w.max()),
                "condition_number": cond,
                "effective_rank": eff_rank
            }
    return stats


def get_gradient_stats(model: nn.Module) -> dict:
    """Collect gradient flow statistics per layer after backward pass."""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name and param.ndim >= 2:
            g = param.grad.data.cpu()
            stats[name] = {
                "grad_norm": float(g.norm()),
                "grad_mean": float(g.mean()),
                "grad_std": float(g.std()),
                "grad_max": float(g.abs().max()),
            }
    return stats

# ============================================================================
# Data Loading & Tokenization (WikiText-2)
# ============================================================================

class TokenizedDataset(Dataset):
    def __init__(self, data_list: list[int], seq_len: int):
        self.seq_len = seq_len
        # Chunk into seq_len blocks
        self.chunks = [
            data_list[i : i + seq_len + 1]
            for i in range(0, len(data_list) - seq_len - 1, seq_len)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_wikitext2_dataloaders(batch_size: int, seq_len: int):
    """Load and tokenize WikiText-2 dataset using gpt2 tokenizer from HuggingFace."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("[Info] Loading WikiText-2 dataset and GPT2 Tokenizer...", flush=True)
    raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    def tokenize_function(examples):
        return {"input_ids": tokenizer(examples["text"])["input_ids"]}

    print("[Info] Tokenizing dataset...", flush=True)
    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Flatten input_ids lists
    train_ids = [item for sublist in tokenized_dataset["train"]["input_ids"] for item in sublist]
    val_ids = [item for sublist in tokenized_dataset["validation"]["input_ids"] for item in sublist]

    train_ds = TokenizedDataset(train_ids, seq_len)
    val_ds = TokenizedDataset(val_ids, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    print(f"[Info] WikiText-2 Dataset ready. Vocab size: {vocab_size}, Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}", flush=True)
    return train_loader, val_loader, vocab_size

# ============================================================================
# Main Screening Script
# ============================================================================

def run_experiment(config: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    
    # Load existing runs for resume
    completed_runs = []
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                old_data = json.load(f)
                completed_runs = old_data.get("runs", [])
            print(f"Found {len(completed_runs)} existing runs. Resuming...", flush=True)
        except Exception as e:
            print(f"[Warning] Failed to load existing results.json: {e}", flush=True)

    # Set device
    device_str = config["training"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Global Device: {device}", flush=True)

    # Dataloaders
    batch_size = config["data"]["batch_size"]
    seq_len = config["data"]["seq_len"]
    train_loader, val_loader, hf_vocab_size = get_wikitext2_dataloaders(batch_size, seq_len)

    # Model parameters
    model_cfg = config["model"]
    d_model = model_cfg["d_model"]
    n_heads = model_cfg["n_heads"]
    d_ff = model_cfg["d_ff"]
    n_layers = model_cfg["n_layers"]
    vocab_size = hf_vocab_size

    # Grid of runs
    runs_to_execute = []
    seed_range = config["experiment"]["seed_range"]
    init_methods = config["init_methods"]

    for init in init_methods:
        name = init["name"]
        kind = init.get("kind", "he")
        m = init.get("m", 8)
        offsets = init.get("offsets", [0])
        scramble_seeds = init.get("scramble_seeds", [0])
        params = init.get("params", {})
        assignment = params.get("assignment", "sequential")
        orthogonalize = params.get("orthogonalize", False)
        matrix_shaped = params.get("matrix_shaped", False)

        # Get weight_decay from init config (allows per-method WD sweep)
        wd = init.get("weight_decay", params.get("weight_decay", config["training"].get("weight_decay", 0.01)))

        if name == "xavier" or name == "he":
            for seed in seed_range:
                runs_to_execute.append({
                    "init": name, "kind": kind, "m": None, "seed": seed, "offset": 0,
                    "scramble_seed": 0, "assignment": "sequential", "orthogonalize": False, "matrix_shaped": False,
                    "weight_decay": wd,
                })
        elif name in ("ce_n", "ce_u"):
            for offset in offsets:
                for seed in seed_range:
                    runs_to_execute.append({
                        "init": name, "kind": kind, "m": m, "seed": seed, "offset": offset,
                        "scramble_seed": 0, "assignment": assignment, "orthogonalize": orthogonalize, "matrix_shaped": False,
                        "weight_decay": wd,
                    })
        elif name in ("sobol_n", "sobol_u"):
            for scr_seed in scramble_seeds:
                for seed in seed_range:
                    runs_to_execute.append({
                        "init": name, "kind": kind, "m": None, "seed": seed, "offset": 0,
                        "scramble_seed": scr_seed, "assignment": assignment, "orthogonalize": False, "matrix_shaped": matrix_shaped,
                        "weight_decay": wd,
                    })
        elif name == "selective":
            # Selective init: per-layer-type rules
            rules = init.get("rules", {})
            label = init.get("label", "selective")
            for seed in seed_range:
                runs_to_execute.append({
                    "init": "selective", "kind": kind, "m": m, "seed": seed, "offset": offsets[0] if offsets else 0,
                    "scramble_seed": 0, "assignment": label, "orthogonalize": False, "matrix_shaped": False,
                    "selective_rules": rules,
                    "weight_decay": wd,
                })

    print(f"Total screening runs configured: {len(runs_to_execute)}", flush=True)

    # Run Loop
    for run_idx, run_meta in enumerate(runs_to_execute):
        # Skip if already completed
        already_done = False
        for completed in completed_runs:
            match = True
            for k in ["init", "seed", "offset", "scramble_seed", "assignment", "orthogonalize", "matrix_shaped", "weight_decay"]:
                if completed.get(k) != run_meta.get(k):
                    match = False
                    break
            if match:
                already_done = True
                break

        if already_done:
            continue

        print(f"\nRunning: WikiText2 + {run_meta['init']}(offset={run_meta['offset']}) | assignment={run_meta['assignment']} | seed={run_meta['seed']}", flush=True)

        # Set seed
        torch.manual_seed(run_meta["seed"])
        np.random.seed(run_meta["seed"])

        # Build Model
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers, max_seq_len=seq_len
        )
        
        # Apply custom initialization
        if run_meta["init"] == "selective":
            apply_selective_init(
                model,
                rules=run_meta["selective_rules"],
                m=run_meta.get("m", 8),
                offset=run_meta["offset"],
            )
        else:
            apply_init(
                model,
                init_name=run_meta["init"],
                kind=run_meta["kind"],
                m=run_meta.get("m", 4),
                offset=run_meta["offset"],
                scramble_seed=run_meta["scramble_seed"],
                assignment=run_meta["assignment"],
                orthogonalize=run_meta["orthogonalize"],
                matrix_shaped=run_meta["matrix_shaped"],
            )

        model.to(device)

        # Optimizer (weight_decay configurable per-run)
        wd = run_meta.get("weight_decay", config["training"].get("weight_decay", 0.01))
        optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        epochs = config["training"]["epochs"]
        epoch_logs = []

        # Train Loop
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                logits = model(x)
                # Reshape for cross entropy loss
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                
                # Collect gradient stats on last batch of epoch
                if x is train_loader.dataset[-1] if False else True:  # Always collect (lightweight)
                    pass  # grad stats collected after loop
                
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item()
                    
            val_loss_avg = val_loss / len(val_loader)
            val_ppl = math.exp(val_loss_avg) if val_loss_avg < 20 else float('inf')

            # Spectral diagnostics (weight stats)
            w_stats = get_weight_and_spectral_stats(model)
            
            # Gradient flow diagnostics (collected from last training batch)
            g_stats = get_gradient_stats(model)

            print(f"  Epoch {epoch:2d}/{epochs}: train_loss={train_loss:.4f} | val_loss={val_loss_avg:.4f} val_ppl={val_ppl:.2f}", flush=True)

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": val_loss_avg,
                "test_accuracy": 1.0 / (val_loss_avg + 1e-10), # Pseudo-accuracy for plotting code compatibility
                "perplexity": val_ppl,
                "weight_stats": w_stats,
            }
            if g_stats:
                epoch_log["gradient_stats"] = g_stats
            epoch_logs.append(epoch_log)

        # Save finished run
        run_meta["epochs"] = epoch_logs
        run_meta["final_accuracy"] = epoch_logs[-1]["test_accuracy"]
        run_meta["final_perplexity"] = epoch_logs[-1]["perplexity"]
        run_meta["best_perplexity"] = min(e["perplexity"] for e in epoch_logs)
        run_meta["best_epoch"] = min(range(len(epoch_logs)), key=lambda i: epoch_logs[i]["perplexity"]) + 1
        run_meta["convergence_epoch"] = None

        completed_runs.append(run_meta)

        # Save results immediately
        with open(results_path, "w") as f:
            json.dump({"experiment": config["experiment"], "runs": completed_runs}, f, indent=2)
            
        print(f"  [saved {len(completed_runs)} runs to {results_path}]", flush=True)

        # Google Drive Auto-Upload Trigger (K005 §3)
        gdrive_filename = config["experiment"].get("name", "transformer_screening") + "_results.json"
        if os.path.exists("/content/sa.json"):
            print(f"  [K005] Path B active. Uploading {gdrive_filename} to GDrive...", flush=True)
            try:
                from pydrive2.auth import GoogleAuth
                from pydrive2.drive import GoogleDrive
                from oauth2client.service_account import ServiceAccountCredentials

                scope = ["https://www.googleapis.com/auth/drive"]
                creds = ServiceAccountCredentials.from_json_keyfile_name("/content/sa.json", scope)
                gauth = GoogleAuth()
                gauth.credentials = creds
                drive = GoogleDrive(gauth)

                def find_folder(drive_client, name, parent_id=None):
                    if parent_id:
                        q = f"title = '{name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
                    else:
                        q = f"title = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
                    file_list = drive_client.ListFile({"q": q}).GetList()
                    return file_list[0]['id'] if file_list else None

                root_id = find_folder(drive, "agent-rules-tree-control")
                if root_id:
                    project_id = find_folder(drive, "copeland-erdos-nets_drive", root_id)
                    if project_id:
                        results_folder_id = find_folder(drive, "results", project_id)
                        if results_folder_id:
                            # Update existing file or create new
                            q = f"title = '{gdrive_filename}' and '{results_folder_id}' in parents and trashed = false"
                            file_list = drive.ListFile({"q": q}).GetList()
                            if file_list:
                                gfile = drive.CreateFile({"id": file_list[0]['id']})
                            else:
                                gfile = drive.CreateFile({"title": gdrive_filename, "parents": [{"id": results_folder_id}]})
                            gfile.SetContentFile(str(results_path))
                            gfile.Upload()
                            print(f"  [K005-B] GDrive Direct Upload SUCCESS: {gdrive_filename}", flush=True)
            except Exception as e:
                print(f"  [Warning] GDrive Auto-Upload failed: {e}", flush=True)

        # Cool down and GC
        time.sleep(config["training"]["cooldown_seconds"])
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Transformer Screening Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    run_experiment(config, Path(args.output))


if __name__ == "__main__":
    main()
