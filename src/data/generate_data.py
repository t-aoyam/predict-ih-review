from data_generator import DataGenerator
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--p_aba', type=float, required=True,
                        help='P(AB...A)')
    parser.add_argument('-b', '--p_b_given_aba', type=float, required=True,
                        help='P(B|AB...A)')
    parser.add_argument('-t', '--training_size', type=int, default=100_000_000,
                        help='training data size, default=100M')
    parser.add_argument('-va', '--val_size', type=int, default=100_000,
                        help='training data size, default=100K')
    parser.add_argument('-vo', '--vocab_size', type=int, default=10_000,
                        help='vocab size, default=10K')
    parser.add_argument('-ctx', '--ctx_size', type=int, default=64,
                        help='context size, default=64')
    parser.add_argument('-td', '--trans_dist', type=str,
                        help='Type of distribution to sample transition matrix from')
    parser.add_argument('-ed', '--emis_dist', type=str, default=None,
                        help='Type of distribution to sample emission matrix from')
    parser.add_argument('-tf', '--transition_fp', default=None,
                        help='Path to .pkl for transition distribution for sampling')
    parser.add_argument('-ef', '--emission_fp', default=None,
                        help='Path to .pkl for emission distribution for sampling')
    parser.add_argument('-cat', '--cat_num', type=int, default=None,
                        help='number of categories, default=50')
    parser.add_argument('-s', '--seed', type=int,
                        help='random seed')
    args = parser.parse_args()

    data_generator = DataGenerator(
        vocab_size=args.vocab_size,
        ctx_size=args.ctx_size,
        p_aba=args.p_aba,
        p_b_given_aba=args.p_b_given_aba,
        enforce_p_at=int(args.ctx_size//2),
        output_dir='data',
        trans_dist=args.trans_dist,
        emis_dist=args.emis_dist,
        seed=args.seed,
        num_cat=args.cat_num,
        trans_fp=args.transition_fp,
        emis_fp=args.emission_fp,
    )

    data_generator.generate_sequences(split='val', size=100_000)
    data_generator.generate_sequences(split='train', size=100_000_000)


if __name__ == "__main__":
    main()