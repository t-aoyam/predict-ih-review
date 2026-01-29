from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import random, os, json, argparse

"""CREATE A 1B DATASET THAT CONTAINS X% OF REPETITIONS"""

def get_num_reps(chunk, ctx_size):
    total = 0
    bigrams = defaultdict(lambda: defaultdict(lambda: 0))
    # unigrams = defaultdict(lambda: 0)
    for i in range(ctx_size-1):
        bigrams[chunk[i]][chunk[i+1]] += 1
        # unigrams[chunk[i]] += 1
    # unigrams[chunk[i+1]] += 1  # last token
    for uni, cont in bigrams.items():
        # print(cont)
        total += sum([count-1 for count in list(cont.values())])
    return total

def sample_by_abab(
        data, tokenizer, ctx_size, rep_p,
        goal_data_size=1_000_000_000, seed=42):
    """
    :param data: DataSet object
    :param tokenizer: Tokenizer object
    :param ctx_size: 4-1024
    :param p: proportion of *no repetition* chunks
    :param goal_data_size: number of tokens
    :return: list of chunks
    """
    goal_norep_size, goal_rep_size = goal_data_size*(1-rep_p), goal_data_size*rep_p
    curr_norep_size, curr_rep_size = 0, 0
    chunks, toks = [], []
    # pbar = tqdm(total=goal_data_size, leave=True, position=0)
    curr_data_size = 0
    buffer_len = 0
    for doc in data:
        if not doc['text'].strip('\n'):
            toks.append(tokenizer.eos_token_id)
            buffer_len += 1
            curr_data_size += 1
        tokenized = tokenizer(doc['text'])['input_ids']
        l = len(tokenized)
        toks.extend(tokenized)
        buffer_len += l
        while buffer_len >= ctx_size and curr_data_size < goal_data_size:
            chunk, toks = toks[:ctx_size], toks[ctx_size:]
            buffer_len -= ctx_size
            num_reps = get_num_reps(chunk, ctx_size)
            if num_reps > 0 and curr_rep_size < goal_rep_size:
                chunks.append(chunk)
                curr_rep_size += ctx_size
                curr_data_size += ctx_size
                norep_progress = round(curr_norep_size / goal_norep_size * 100, 2) if goal_norep_size > 0 else 'NA'
                rep_progress = round(curr_rep_size/goal_rep_size*100, 2) if goal_rep_size > 0 else 'NA'
                total_progress = round(curr_data_size/goal_data_size*100, 2) if goal_data_size > 0 else 'NA'
                print(f"\rNorep:\t{norep_progress}%\t|\tRep:\t{rep_progress}%\t|||\tTotal:\t{total_progress}%", end="")
                # pbar.update(ctx_size)
            elif num_reps == 0 and curr_norep_size < goal_norep_size:
                chunks.append(chunk)
                curr_norep_size += ctx_size
                curr_data_size += ctx_size
                norep_progress = round(curr_norep_size / goal_norep_size * 100, 2) if goal_norep_size > 0 else 'NA'
                rep_progress = round(curr_rep_size/goal_rep_size*100, 2) if goal_rep_size > 0 else 'NA'
                total_progress = round(curr_data_size/goal_data_size*100, 2) if goal_data_size > 0 else 'NA'
                print(f"\rNorep:\t{norep_progress}%\t|\tRep:\t{rep_progress}%\t|||\tTotal:\t{total_progress}%", end="")
                # pbar.update(ctx_size)

        if curr_data_size >= goal_data_size:
            print('Data size reached the goal!')
            break
    random.seed(seed)
    chunks = random.sample(chunks, len(chunks))
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--context_size', type=int,
                        help='context size which tokenized texts will be chunked into')
    parser.add_argument('-p', '--rep_p', type=float,
                        help='proportion of the chunks that have at least one AB...AB')
    args = parser.parse_args()
    ctx_size, rep_p = args.context_size, args.rep_p

    data = load_dataset('cc100', 'en', streaming=True)
    data = data['train']
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    chunks = sample_by_abab(
        data=data, tokenizer=tokenizer,
        ctx_size=ctx_size, rep_p=rep_p,
    )

    fn = f"cc100_1b_c{ctx_size}_rep{int(100*(rep_p))}.jsonl"
    with open(os.path.join('data', fn), 'w') as f:
        for chunk in chunks:
            f.write(json.dumps({'input_ids': chunk})+'\n')

if __name__ == '__main__':
    main()