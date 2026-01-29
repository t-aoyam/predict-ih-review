#!/bin/bash
# run these commands from the root (icml-2026-32900/)

# Experiment 2
# generate a P(AB...A) = P(B|AB...A) = 90% data using natural transition matrix
python -m src.data.generate_data -a 0.9 -b 0.9 -vo 10000 -f data/cc100_v10000_100M_bigram_tensor.pkl
# generate 

python markov_model.py -v 10000 -s 1 -e 6.2 -d uniform -wi 0.1 -ac 0.1 -dw 100 -ew 0.01 -gw 5 -pw 0.1 -m 5000
python generate_data.py -a 0.1 -b 0.3 -tf data/trans_v10000_h0_ld_lowcat_uniform_s1.pkl -s 1

# Experiment 3
# generate |V|x|V| transition matrix (Unif[+D+C]) (Experiment 3)
python -m src.data.markov_model -v 10000 -s 1 -e 6.2 -d uniform -wi 0.4 -ac 0.1 -dw 100 -ew 0.01 -gw 5 -pw 0.1 -m 5000
# using the transition matrix, sample a training data while imposing P(AB...A)=0.1 and P(B|AB...A)=0.3
python -m src.data.generate_data -a 0.1 -b 0.3 -tf data/trans_v10000_h0_ld_highcat_uniform_s1.pkl -s 1
