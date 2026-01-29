import torch.nn.functional
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os, pickle
import torch

"""LOAD DATA AND TOKENIZE THEM USING GPT2 TOKENIZER"""
data = load_dataset('cc100', 'en', streaming=True)
data = data['train']

tokenizer = AutoTokenizer.from_pretrained('gpt2')
toks = []
total = 0
goal = 100_000_000
# goal = 1_000_000
pbar = tqdm(total=goal)
reduced_vocab_size = 10_000
curr_vocab_set = {tokenizer.eos_token_id}
curr_vocab_size = 1

for doc in data:
    if not doc['text'].strip('\n'):
        toks.append(tokenizer.eos_token_id)
    tokenized = tokenizer(doc['text'])['input_ids']
    if reduced_vocab_size:
        new_ids = set(tokenized).difference(curr_vocab_set)
        if curr_vocab_size + len(new_ids) < reduced_vocab_size:
            curr_vocab_set.update(new_ids)
            curr_vocab_size += len(new_ids)
            l = len(tokenized)
            pbar.update(l)
            total += l
            toks.extend(tokenized)
            if total > goal:
                break
    else:
        l = len(tokenized)
        pbar.update(l)
        total += l
        toks.extend(tokenized)
        if total > goal:
            break

"""CONVERT TO 0-9999 IF NECESSARY"""
print('converting reduced vocab...')
if reduced_vocab_size:
    mapping = {i:j for i, j in zip(list(curr_vocab_set), list(range(curr_vocab_size)))}
    toks = [mapping[tok] for tok in toks]
print('done!')

"""COUNT BIGRAMS"""

# list of lists is better, since it will require smoothing anyways
if reduced_vocab_size:
    bigrams = torch.Tensor(curr_vocab_size, curr_vocab_size)
else:
    bigrams = torch.Tensor(len(tokenizer.vocab), len(tokenizer.vocab))
for i in tqdm(range(len(toks)-1), leave=True, position=0):
    cur, nxt = toks[i], toks[i+1]
    # implicitly, EOS -> x is a starting token
    bigrams[cur][nxt] += 1

fn = 'cc100_v10000_100M_bigram_tensor.pkl'
with open(os.path.join('data', fn), 'wb') as f:
    pickle.dump(bigrams, f)
#
# with open(os.path.join('data', fn), 'rb') as f:
#     test = pickle.load(f)
#
# test += 0.01
# test = torch.nn.functional.normalize(test, dim=1)
# sum(test[0])
#
# m = torch.Tensor([
#     [1,2,3],
#     [1,2,3]
# ])
#
# torch.nn.functional.normalize(m, dim=1)
# torch.softmax(m, dim=0)
# m /= m.sum(dim=1, keepdim=True)
