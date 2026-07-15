"""Evaluate MoE routing specialization: check which expert is invoked for which token.
Prints the bits-per-byte (bpb) and outputs detailed routing logs showing
whether Expert 0 and Expert 1 specialized by language (English vs. Hindi) or other patterns.
"""
import argparse
import json
import math
import os
import re
import torch
import torch.nn.functional as F

from model import GPT, Config
import tokenizer as tokenizer_mod

# Structure to collect routing data during evaluation
# format: { token_id: { layer_name: { expert_idx: count } } }
token_routing_stats = {}
global_routing_counts = {}

# Current pass routing storage
current_pass_routing = {}


def patch_moe_layers(model):
    """Monkey-patch MoE layers to record token routing decisions during forward passes."""
    moe_instances = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'MoELayer':
            moe_instances.append((name, module))
            
    for name, module in moe_instances:
        orig_forward = module.forward
        
        def make_wrapped_forward(m, orig_f, layer_name):
            def wrapped_forward(x):
                B, T, C = x.shape
                # Re-compute gate indices to log them
                gate_logits = m.gate(x.view(B * T, C))
                gate_probs = F.softmax(gate_logits, dim=-1)
                expert_idx = gate_probs.argmax(dim=-1).cpu().numpy().tolist() # list of length B*T
                
                # Store routing for the current forward pass
                current_pass_routing[layer_name] = expert_idx
                
                # Execute original forward
                return orig_f(x)
            return wrapped_forward
            
        module.forward = make_wrapped_forward(module, orig_forward, name)
    return [name for name, _ in moe_instances]


@torch.no_grad()
def evaluate_and_log_routing(model, cfg, tok, text, moe_layer_names):
    n_bytes = len(text.encode("utf-8"))
    id_list = tok.encode(text)
    ids = torch.tensor(id_list, dtype=torch.long)
    
    block, stride = cfg.block_size, max(1, cfg.block_size // 2)
    total_nll, n_scored = 0.0, 0
    scored = 1
    
    # Initialize global stats
    for name in moe_layer_names:
        global_routing_counts[name] = {0: 0, 1: 0}
        
    while scored < len(ids):
        start = max(0, scored - stride)
        end = min(len(ids), start + block)
        window = ids[start:end]
        
        # Clear current pass routing
        current_pass_routing.clear()
        
        # Forward pass (triggers our patched forward methods)
        logits, _ = model(window[None, :])
        logp = torch.log_softmax(logits[0], dim=-1)
        targets = ids[start + 1:end]
        nll = -logp[torch.arange(len(targets)), targets]
        offset = scored - (start + 1)
        assert offset >= 0
        total_nll += nll[offset:].sum().item()
        n_scored += len(nll) - offset
        
        # Log the routing decisions for the current context window
        window_list = window.cpu().tolist()
        for t_idx, token_id in enumerate(window_list):
            if token_id not in token_routing_stats:
                token_routing_stats[token_id] = {name: {0: 0, 1: 0} for name in moe_layer_names}
                
            for name in moe_layer_names:
                if name in current_pass_routing:
                    expert = current_pass_routing[name][t_idx]
                    token_routing_stats[token_id][name][expert] += 1
                    global_routing_counts[name][expert] += 1
                    
        scored = end
        
    bpb = total_nll / math.log(2) / n_bytes
    return bpb, n_scored, len(ids)


def is_hindi_token(token_str):
    """Determine if a token contains Hindi Devanagari characters."""
    return any('\u0900' <= char <= '\u097F' for char in token_str)


def analyze_and_print_routing(tok, moe_layer_names):
    print("\n" + "="*60)
    print("           MoE EXPERT ROUTING ANALYSIS REPORT")
    print("="*60)
    
    # Decoded tokens cache
    decoded_vocab = {}
    for tid in token_routing_stats.keys():
        decoded_vocab[tid] = tok.decode([tid])
        
    for name in moe_layer_names:
        print(f"\nLayer: {name}")
        counts = global_routing_counts[name]
        total = counts[0] + counts[1]
        p0 = (counts[0] / total * 100) if total > 0 else 0
        p1 = (counts[1] / total * 100) if total > 0 else 0
        print(f"  Total routed tokens: {total:,}")
        print(f"  Expert 0 invocations: {counts[0]:,} ({p0:.1f}%)")
        print(f"  Expert 1 invocations: {counts[1]:,} ({p1:.1f}%)")
        
        # Collect token routing list for sorting
        token_list = []
        for tid, layers in token_routing_stats.items():
            token_str = decoded_vocab[tid]
            e0_count = layers[name][0]
            e1_count = layers[name][1]
            t_total = e0_count + e1_count
            if t_total > 0:
                e0_ratio = e0_count / t_total
                token_list.append({
                    "id": tid,
                    "str": token_str,
                    "e0_count": e0_count,
                    "e1_count": e1_count,
                    "total": t_total,
                    "e0_ratio": e0_ratio
                })
                
        # Bilingual routing analysis
        hindi_e0, hindi_e1 = 0, 0
        english_e0, english_e1 = 0, 0
        for item in token_list:
            if is_hindi_token(item["str"]):
                hindi_e0 += item["e0_count"]
                hindi_e1 += item["e1_count"]
            else:
                english_e0 += item["e0_count"]
                english_e1 += item["e1_count"]
                
        total_hindi = hindi_e0 + hindi_e1
        total_english = english_e0 + english_e1
        
        print("\n  [Bilingual Routing Breakdown]")
        if total_english > 0:
            print(f"    English/Latin tokens: Expert 0 = {hindi_e0 / total_english * 100:.1f}% | Expert 1 = {english_e1 / total_english * 100:.1f}%")
        else:
            print("    No English/Latin tokens scored.")
            
        if total_hindi > 0:
            print(f"    Hindi Devanagari tokens: Expert 0 = {hindi_e0 / total_hindi * 100:.1f}% | Expert 1 = {hindi_e1 / total_hindi * 100:.1f}%")
        else:
            print("    No Hindi Devanagari tokens scored.")
            
        # Top 10 tokens routed to Expert 0
        print("\n  [Top 10 Tokens Preferred by Expert 0]")
        e0_sorted = sorted([item for item in token_list if item["e0_count"] > 0], key=lambda x: x["e0_count"], reverse=True)
        for i, item in enumerate(e0_sorted[:10]):
            repr_str = repr(item["str"])
            print(f"    {i+1:2d}. {repr_str:<15} (ID: {item['id']:4d}) count: {item['e0_count']:5d} (e0 ratio: {item['e0_ratio']*100:.1f}%)")
            
        # Top 10 tokens routed to Expert 1
        print("\n  [Top 10 Tokens Preferred by Expert 1]")
        e1_sorted = sorted([item for item in token_list if item["e1_count"] > 0], key=lambda x: x["e1_count"], reverse=True)
        for i, item in enumerate(e1_sorted[:10]):
            repr_str = repr(item["str"])
            print(f"    {i+1:2d}. {repr_str:<15} (ID: {item['id']:4d}) count: {item['e1_count']:5d} (e1 ratio: {(1-item['e0_ratio'])*100:.1f}%)")
            
    print("="*60 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--text_file", required=True)
    args = ap.parse_args()
    
    # Load model and config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = Config()
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Patch MoE layers and get their names
    moe_layer_names = patch_moe_layers(model)
    print(f"Successfully patched {len(moe_layer_names)} MoE layers: {moe_layer_names}")
    
    # Load tokenizer and evaluation text
    tok = tokenizer_mod.load()
    text = open(args.text_file, encoding="utf-8").read()
    
    # Evaluate and accumulate statistics
    print("Evaluating and logging routing decisions...")
    bpb, n_scored, n_tokens = evaluate_and_log_routing(model, cfg, tok, text, moe_layer_names)
    
    # Output main results
    print(json.dumps({
        "bpb": round(bpb, 4),
        "n_params": model.n_params(),
        "steps": ckpt.get("steps"),
        "tokens_in_eval": n_tokens,
        "tokens_scored": n_scored,
    }))
    
    # Print the analysis
    analyze_and_print_routing(tok, moe_layer_names)


if __name__ == "__main__":
    main()
