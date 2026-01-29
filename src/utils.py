import re
import torch
import einops
import gc
import torch

def extract_model_config(model_name):
    cs_regex, bs_regex, dim_regex, layers_regex, heads_regex, step_regex = \
        r'c(\d+)', r'b(\d+)', r'd(\d+)', r'l(\d+)', r'h(\d+)', r'checkpoint-(\d+)|-step(\d+)'
    cs = int(re.search(cs_regex, model_name).group(1))
    bs = int(re.search(bs_regex, model_name).group(1))
    dim = int(re.search(dim_regex, model_name).group(1))
    layers = int(re.search(layers_regex, model_name).group(1))
    heads = int(re.search(heads_regex, model_name).group(1))
    step_match = re.search(step_regex, model_name)
    steps = int(step_match.group(1)) if step_match.group(1) else int(step_match.group(2))
    toks = cs * bs * steps
    return {'cs': cs, 'bs': bs, 'dim': dim, 'layers': layers, 'heads': heads, 'steps': steps, 'toks': toks}

def human_format(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}".replace(".0", "")
            # replace(".0", "") keeps "55M" instead of "55.0M"
        num /= 1000
    return f"{num:.1f}P"

def get_hf_config(model):
    # unwrap common wrappers first
    for attr in ("module", "_orig_mod", "model"):
        if hasattr(model, attr):
            model = getattr(model, attr)

    # the normal HF place
    if hasattr(model, "config") and model.config is not None:
        return model.config

    # some wrappers keep the base model here
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        return model.base_model.config

    # last resort: walk submodules looking for a .config
    for m in model.modules():
        if hasattr(m, "config") and m.config is not None:
            return m.config

    raise AttributeError("Could not find a Hugging Face-style `config` on this model.")

def unwrap_model(model):
    # DDP/DataParallel/torch.compile/Accelerate-ish wrappers
    for attr in ("module", "_orig_mod", "model"):
        if hasattr(model, attr):
            model = getattr(model, attr)
    return model

def get_block_stack(model) -> torch.nn.ModuleList:
    """
    Return the ModuleList of transformer blocks (GPT-2's model.transformer.h equivalent).
    Raises if not found.
    """
    model = unwrap_model(model)

    # If it's a task head (e.g., XxxForCausalLM), prefer base model
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
    elif hasattr(model, "base_model"):
        base = model.base_model
    else:
        base = model

    # Common patterns
    candidates = [
        # GPT-2 / GPT-Neo / GPT-NeoX-style
        "transformer.h",          # GPT2LMHeadModel, GPTNeoForCausalLM (often)
        "gpt_neox.layers",        # GPTNeoXForCausalLM
        "model.layers",           # LLaMA/Mistral/Qwen2/etc (often base.model.layers)
        "layers",                 # some bases expose directly
        "model.decoder.layers",   # some seq2seq-ish decoders
        "decoder.layers",         # some decoder-only wrappers
        "gptj.h",                 # GPTJForCausalLM (sometimes)
        "transformer.blocks",     # MPT
        "model.transformer.h",    # some wrappers nest like this
        "transformer.layers",     # some implementations
    ]

    def resolve(obj, path: str):
        cur = obj
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    for path in candidates:
        blocks = resolve(base, path)
        if isinstance(blocks, torch.nn.ModuleList):
            return blocks

    # Fallback: search for a ModuleList that "looks like" the main block stack
    # (largest ModuleList whose elements are torch.nn.Module and repeat similar types)
    best = None
    best_len = 0
    for name, module in base.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > best_len:
            # Heuristic: block stacks are usually length >= 2 and contain Modules
            if len(module) >= 2 and all(isinstance(x, torch.nn.Module) for x in module):
                best = module
                best_len = len(module)

    if best is not None:
        return best

    raise AttributeError("Could not locate a transformer block stack (ModuleList) on this model.")

def get_logit_attribution(model, input_ids, mean_normalize=True, relu=True, ratio=True):
    """
    src is the position whose logit contributions we are measuring (the second A in AB...AB)
    dst is the token ID whose logit increase we are measuring (the B token in AB...AB)
    
    :param mean_normalize: subtract the mean logit contribution across vocab *in the sample* from each logit contribution
    :param relu: pass the logit contributions through ReLU to zero-out negative contributions
    :param ratio: take the ratio of target logit contribution to total logit contribution
    """
    # input_ids = self.transformer.input_ids # (bs, seq_len)
    bs, seq_len = input_ids.shape
    mid = seq_len // 2
    
    # check if the input is a proper synthetic induction head input (i.e., repetition of the same sequence)
    if not torch.all(input_ids[:, :mid] == input_ids[:, mid:]):
        print("Warning: Some batches are not repetitions.")

    # 1. Get the tokens for the denominator (all tokens in the sample)
    # this assumes that the input sequence is a unique and random sequence repeated twice
    sample_ids = input_ids[:, :mid] # (bs, all_unique_ids = seqlen)
    sample_unembed = model.transformer.wte.weight.t()[:, sample_ids] # (d, bs, dst_ids) -> these are token IDs and not positions

    # 2. Get the target IDs (what we WANT to predict at each step)
    # For ABCDABCD, at second half positions [4,5,6,7],
    # targets are [1,2,3,4] (Indices of B,C,D,A)
    target_ids = input_ids[:, 1:mid+1] # This handles the wrap-around if mid+1 is used

    logit_attribution_scores = []

    for i in range(model.config.n_layer):
        # Head output at the destination (the second half)
        # Shape: (batch, head, src_positions, dim)
        head_out = model.transformer.h[i].per_head_output[:, :, mid:, :]

        # --- DENOMINATOR MATH ---
        # Boost to every token in the sample (bs, head, mid, mid)
        logits_all_in_sample = einops.einsum(
            head_out, sample_unembed,
            'batch head src_positions dim, dim batch dst_ids -> batch head src_positions dst_ids'
        )

        # Center and ReLU
        if mean_normalize:
            sample_mean = logits_all_in_sample.mean(dim=-1, keepdim=True)
        if relu:
            logits_relu_all = torch.relu(logits_all_in_sample - sample_mean)
            denominator = logits_relu_all.sum(dim=-1)

        # --- NUMERATOR MATH (The Target B) ---
        # Unembed the specific targets for each position
        target_unembed = self.transformer.wte.weight.t()[:, target_ids] # (dim, bs, src_positions) (last dimension is dst_id for each of the src_positions)

        # (bs, head, mid) - Each pos gets its own specific target boost
        target_logits = einops.einsum(
            head_out, target_unembed,
            'batch head src_positions dim, dim batch src_positions -> batch head src_positions'
        )

        # Center using the same mean and ReLU
        if mean_normalize:
            target_boost = target_logits - sample_mean.squeeze(-1)
        if relu:
            target_boost = torch.relu(target_boost)

        # --- FINAL RATIO ---
        ratio = target_boost / denominator
        logit_attribution_scores.append(ratio)

    logit_attribution_scores = torch.stack(logit_attribution_scores, dim=1) # (bs, layer, head, src_positions)
    return logit_attribution_scores
