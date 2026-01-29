"""
Author: Tatsuya
Given a model, either (1) print the prefix matching score of all heads or
                      (2) print the index of the heads that have a higher score than a given threshold t
Prefix matching score:
Given a repeated random sequence of size s, at position s+i (i < s),
compute a given head's average attention paid to i+1 (same token as the current position's next token)

Logit attribution:
Not implemented yet
"""

import os
os.environ['HF_HOME'] = '../data/hf_models/'
import pathlib, argparse, torch, einops, re, json, gc
from transformers import (GPTNeoXForCausalLM,
                          AutoTokenizer, GPT2TokenizerFast, AutoConfig)
from torch.nn import CrossEntropyLoss
from torch import cuda
import numpy as np
import scipy.stats as stats
# from matplotlib import pyplot as plt
from src.models.hooked_gpt2 import HookedGPT2LMHeadModel
from src.models.hooked_gptneox import HookedGPTNeoXForCausalLM
from src.utils import extract_model_config, human_format
from tqdm import tqdm

print("cuda.is_available:", cuda.is_available())
print("device_count:", cuda.device_count())
print("current device:", cuda.current_device() if cuda.is_available() else None)
print("device name:", cuda.get_device_name(0) if cuda.is_available() else None)

device='cuda:0' if cuda.is_available() else 'cpu'

def _generate_random_tokens(tokenizer, seq_len=50, rep=2, num_samples=100, vocab_size=None, id_range=None):
    v = vocab_size if vocab_size else len(dict(tokenizer.get_vocab()))
    input_tensor = torch.stack([
        torch.randperm(v)[:seq_len]  # make sure these tokens are unique
        for _ in range(num_samples)
    ])

    # size = (num_samples, seq_len)
    # if id_range:
    #     # input_tensor = torch.randint(id_range[0], id_range[1], size)
    #     input_tensor = torch.randint(0, id_range[0] - 1, size)
    # else:
    #     input_tensor = torch.randint(0, len(dict(tokenizer.get_vocab())) - 1, size)

    # input_tensor.shape
    # random_tokens = input_tensor.to(model.cfg.device)
    repeated_tokens = einops.repeat(input_tensor, f"batch seq_len -> batch ({rep} seq_len)")
    # repeated_tokens.shape
    return repeated_tokens


def get_scores(model, tokenizer, head_type, num_layers, num_heads,
               seq_len=50, rep=2, num_samples=100, batch_size=100, vocab_size=None, id_range=None, method='all'):
    if head_type == 'induction':
        num_tokens_back = seq_len - 1
    elif head_type == 'previous_token':
        num_tokens_back = 1
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    loss_fct = CrossEntropyLoss(reduction='none')
    num_batches = num_samples // batch_size
    all_logit_attrs = []
    all_attentions = []

    for _ in tqdm(range(num_batches)):
        inputs = _generate_random_tokens(tokenizer, seq_len, rep, batch_size, vocab_size, id_range)
        #  (batch_size, seq_len*rep)
        # next_tok_dct = [
        #     {
        #         inputs[i][j]:inputs[i][j+1] for j in range(0, seq_len)
        #     } for i in range(num_samples)
        # ]
        inputs = inputs.to(device)
        with torch.no_grad():
            output = model(inputs, labels=inputs, output_attentions=True)
            if method in ['all', 'la']:
                logit_attrs = model.get_logit_attribution(inputs)
        all_logit_attrs.append(logit_attrs)
        # logits = output.logits
        # logits.shape
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = inputs[..., 1:].contiguous()
        # Compute per-token loss using CrossEntropyLoss
        # compute loss for each token
        # loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # for i in (99, 991, 99):
        #     plt.plot(loss_per_token[:99])

        # loss_per_token.shape

        # correct_log_probs = model.loss_fn(repeated_logits, repeated_tokens, per_token=True)
        # prefix-matching score

        # 1. Stack and permute to: (layer, head, batch, source_pos, target_pos)
        # Assuming output.attentions is a tuple of (layer, batch, head, src, tgt)
        attentions = torch.stack(output.attentions).permute(0, 2, 1, 3, 4)  # (layer, head, batch, src_positions, tgt_positions)
        all_attentions.append(attentions)

    head2score = {'heads': dict(),
                'model': dict()}  # key (layer, head), value (score)

    attentions = torch.cat(all_attentions, dim=2)  # (layer, head, batch, src_positions, tgt_positions)
    logit_attrs = torch.cat(all_logit_attrs, dim=0)  # (batch, layer, head, src_positions)
    # 2. Define your source and target ranges
    src_indices = torch.arange(seq_len, seq_len * rep)
    tgt_indices = src_indices - num_tokens_back

    # 3. Extract and calculate scores
    for l in range(num_layers):
        for h in range(num_heads):
            head_id = f"{l}-{h}"
            # This slice gets [all_batches, all_src_positions, specific_tgt_position]
            # Then we flatten to get the "batch after another" order
            ps_values = attentions[l, h, :, src_indices, tgt_indices].flatten()
            
            # Convert to numpy once for stats
            ps_np = ps_values.detach().cpu().numpy()
            
            head2score['heads'][head_id] = {
                'ps': {'mean': float(np.mean(ps_np)), 'sd': float(np.std(ps_np))}
            }

    # pss = []
    # if method in ['all', 'ps']:
    #     for l in range(num_layers):
    #         for h in range(num_heads):
    #             head_id = f"{l}-{h}"
    #             for b in range(num_samples):
    #                 for source_id in range(seq_len, seq_len*rep):
    #                     target_id = source_id-(num_tokens_back)
    #                     pss.append(float(output.attentions[l][b][h][source_id][target_id]))
    #             if head_id not in head2score['heads']:
    #                 head2score['heads'][head_id] = dict()
    #             head2score['heads'][head_id]['ps'] = {'mean': np.mean(pss), 'sd': np.std(pss)}
                        # head2score['heads'][head_id]['ps'] += float(output.attentions[l][b][h][source_id][target_id])  # l-th layer, h-th head, attention from source to target
    # logit attribution score

    if method in ['all', 'la']:
        print(logit_attrs.shape)  # (batch, layer, head, src_positions)
        logit_attrs = logit_attrs.permute(1, 2, 0, 3).contiguous().view(num_layers, num_heads, -1)  # (layer, head, batch*src_positions)
        for l in range(num_layers):
            for h in range(num_heads):
                head_id = f"{l}-{h}"
                if head_id not in head2score['heads']:
                    head2score['heads'][head_id] = dict()
                head2score['heads'][head_id]['la'] = {'mean': float(logit_attrs[l, h].mean()),
                                            'sd': float(logit_attrs[l, h].std())}
    if method == 'all':
        for l in range(num_layers):
            for h in range(num_heads):
                head_id = f"{l}-{h}"
                ps_values = attentions[l, h, :, src_indices, tgt_indices].flatten().detach().cpu().numpy()
                la_values = logit_attrs[l, h, :].detach().cpu().numpy()
                corr, p_value = stats.pearsonr(ps_values, la_values)
                head2score['heads'][head_id]['ps-la-corr'] = {
                    'r': float(corr),'p': float(p_value)
                    }
        # logit_attrs = logit_attrs.sum(dim=0)  # all batches, (layer, head, src_positions)
        # logit_attrs = logit_attrs.sum(dim=-1)  # all positions, (layer, head)
        # for l in range(num_layers):
        #     for h in range(num_heads):
        #         head_id = f"{l}-{h}"
        #         if head_id not in head2score['heads']:
        #             head2score['heads'][head_id] = {'ps': 0, 'la':0}
        #         head2score['heads'][head_id]['la'] = float(logit_attrs[l, h])
    # associative recall
    if method in ['all', 'ar']:
        target_ids = inputs[:, 1:seq_len+1]  # (batch, seq_len)
        relevant_logits = output.logits[:, seq_len:, :]  # (batch, seq_len, vocab_size)
        target_logits = torch.gather(
            relevant_logits,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        )
        ranks = (relevant_logits > target_logits).sum(dim=-1)  # (batch, seq_len)
        acc = (ranks == 0).float().mean().item()
        mean_rank = ranks.float().mean().item()
        head2score['model']['ar'] = {'acc': float(acc), 'mean_rank': float(mean_rank)}

    # if method in ['all', 'ps']:
    #     for head_id in head2score:
    #         if head_id == 'ar':
    #             continue
    #         head2score['heads'][head_id]['ps'] /= ((seq_len*rep-seq_len)*num_samples)
    # if method in ['all', 'la']:
    #     for head_id in head2score:
    #         if head_id == 'ar':
    #             continue
    #         head2score['heads'][head_id]['la'] /= ((seq_len*rep-seq_len)*num_samples)
    return head2score

def write_heads(model_dir, output_dir, head_type, head2score, revision, threshold, method):
    model_name = '-'.join(model_dir.split(os.path.sep)[-2:])  # model_name/checkpoint
    if revision and 'pythia' in model_name:
        model_name += f'-step{str(revision)}'
    head2score['model']['model_name'] = model_name
    # induction_heads = ['\t'.join(['head_id', 'ps-mean', 'ps-sd', 'la-mean', 'la-sd', 'ps-la-corr', 'ps-la-corr-p'])]
    try:
        info = extract_model_config(model_name)
        toks = human_format(info['toks'])
    except:
        toks = 'unknown'
    config = AutoConfig.from_pretrained(model_dir)
    head2score['model']['config'] = json.loads(config.to_json_string())
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    print(model_name, "  |  Toks: ", toks)
    for l in range(num_layers):
        for h in range(num_heads):
            head_id = f"{l}-{h}"
            # head_id_w = '-'.join([str(num) for num in head_id])
            ps_mean = head2score['heads'][head_id]['ps']['mean']
            ps_sd = head2score['heads'][head_id]['ps']['sd']
            la_mean = head2score['heads'][head_id]['la']['mean']
            la_sd = head2score['heads'][head_id]['la']['sd']
            r = head2score['heads'][head_id]['ps-la-corr']['r']
            p = head2score['heads'][head_id]['ps-la-corr']['p']
            # scores = [head_id_w]
            # if method in ['ps', 'all']:
            #     scores.append(str(ps))
            # if method in ['la', 'all']:
            #     scores.append(str(la))
            print(f"Head {head_id}:  PS: {ps_mean:.4f} ± {ps_sd:.4f}  |  LA: {la_mean:.4f} ± {la_sd:.4f}  |  r: {r:.4f} (p={p:.4f})")
            # induction_heads.append('\t'.join([head_id_w, *scores]))
    print(f"ACC: {head2score['model']['ar']['acc']:.4f}  |  Mean Rank: {head2score['model']['ar']['mean_rank']:.4f}") 
    # print(str(round(head2score['model']['ar']['acc'], 4)), str(round(head2score['model']['ar']['mean_rank'], 4)))

    # with open(os.path.join(output_dir, '-'.join([model_name, f'{head_type}_heads'])) + '.tsv', 'w') as f:
    #     f.write('\n'.join(induction_heads))
    with open(os.path.join(output_dir, '-'.join([model_name, f'{head_type}_heads'])) + '.json', 'w') as f:
        json.dump(head2score, f, indent=4)

def main():
    ROOT_DIR = pathlib.Path(__file__).parent.resolve()
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-o', '--output_dir', default=None,
                        help=f"output directory")
    parser.add_argument('-t', '--head_type', choices=['induction', 'previous_token'], default='induction',
                        help="type of head to detect, default=induction")
    parser.add_argument('-x', '--metrics', choices=['la', 'ps', 'all'], default='all',
                        help="metriX for head detection: [la, ps, all], default=all")
    parser.add_argument('-s', '--seq_len', type=int, default=None,
                        help="length of the random token sequence, default=50")
    parser.add_argument('-p', '--rep', type=int, default=2,
                        help="numbers by which the random sequence is repeated, default=2")
    parser.add_argument('-n', '--num_samples', type=int, default=100,
                        help="how many repeated sequences to test the model on, default=100")
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help="how many repeated sequences to test the model on, default=100")
    parser.add_argument('-r', '--revision', type=str, required=False,
                        help="checkpoint for Pythia")
    parser.add_argument('-th', '--threshold', type=float, default=0.0,
                        help="threshold of prefix matching score,\
                        beyond which a given head is considered a particular type of head, default=0.8")

    args = parser.parse_args()
    model_dir, output_dir, head_type, seq_len, rep, num_samples, revision, threshold, metrics, batch_size = \
        args.model_dir, args.output_dir, args.head_type, args.seq_len,\
        args.rep, args.num_samples, args.revision, args.threshold, args.metrics, args.batch_size
    assert num_samples % batch_size == 0, "Number of samples must be divisible by the batch size"

    # overwrite seq_len if model name has context size in it
    match = re.search(r'c(\d+)', model_dir)
    if match:
        ctx_size = int(match[1])
        print(f"Based on the model name, context size is {str(ctx_size)}")
    else:
        ctx_size = 2048  # default pythia config
    if ctx_size:
        seq_len = min(50, int(ctx_size/2))

    id_range, vocab_size = None, None
    match = re.search(r'v(\d+)', model_dir)
    if match:
        vocab_size = int(match[1])
        print(f"Based on the model name, vocab size is {str(vocab_size)}")
        id_range = (vocab_size, vocab_size+int(vocab_size/10)-1)

    if not output_dir:
        output_dir = os.path.join(DATA_DIR, f'{head_type}_heads')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Get the number of layers and attention heads
    config = AutoConfig.from_pretrained(model_dir)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    if os.path.isdir(model_dir) and 'checkpoint' not in model_dir:  # if multiple custom models
        checkpoints = sorted([d for d in os.listdir(model_dir) if 'checkpoint' in d],
                             key=lambda x: int(x.split('-')[1]))
        for checkpoint in checkpoints:
            # tokenizer = GPT2TokenizerFast.from_pretrained(os.path.join(model_dir, checkpoint))
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            model = HookedGPT2LMHeadModel.from_pretrained(os.path.join(model_dir, checkpoint))
            model = model.to(device)
            scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                                num_layers=num_layers, num_heads=num_heads, rep=rep, num_samples=num_samples,
                                batch_size=batch_size, vocab_size=vocab_size, id_range=id_range)
            write_heads(os.path.join(model_dir, checkpoint), output_dir,
                        head_type, scores, revision, threshold, method=metrics)
            del scores
            del model
            del tokenizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            

    elif os.path.isdir(model_dir):  # if one custom model
        # tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        model = HookedGPT2LMHeadModel.from_pretrained(model_dir)
        model = model.to(device)
        scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                            num_layers=num_layers, num_heads=num_heads, rep=rep, num_samples=num_samples,
                            batch_size=batch_size, vocab_size=vocab_size, id_range=id_range)
        write_heads(model_dir, output_dir, head_type, scores, revision, threshold, method=metrics)

    else:  # if HF model
        if revision or 'gpt' in model_dir:
            if "gpt" in model_dir:
                tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
                model = HookedGPT2LMHeadModel.from_pretrained(model_dir)
            elif "pythia" in model_dir:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = HookedGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f"step{str(revision)}")
            model = model.to(device)
            scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                                num_layers=num_layers, num_heads=num_heads, rep=rep, num_samples=num_samples,
                                batch_size=batch_size, vocab_size=vocab_size, id_range=id_range)
            write_heads(model_dir, output_dir, head_type, scores, revision, threshold, method=metrics)
        elif 'pythia' in model_dir and not revision:
            for revision in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000]:
                print(f"{model_dir} ||| step{str(revision)}")
                tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = HookedGPTNeoXForCausalLM.from_pretrained(model_dir, revision=f"step{str(revision)}")
                model = model.to(device)
                scores = get_scores(model, tokenizer, head_type=head_type, seq_len=seq_len,
                                    num_layers=num_layers, num_heads=num_heads, rep=rep, num_samples=num_samples,
                                    batch_size=batch_size, vocab_size=vocab_size, id_range=id_range)
                write_heads(model_dir, output_dir, head_type, scores, revision, threshold, method=metrics)
                del scores
                del model
                del tokenizer

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()