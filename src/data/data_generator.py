import random, json, pickle, torch, os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DataGenerator:
    def __init__(
            self,
            vocab_size,
            ctx_size,
            p_aba,
            p_b_given_aba,
            enforce_p_at,
            output_dir,
            trans_dist,
            seed,
            num_cat=None,
            trans_fp=None,
            emis_fp=None,
            emis_dist=None
    ):
        self.vocab_size = vocab_size
        self.num_cat = num_cat
        self.ctx_size = ctx_size
        self.p_aba = p_aba
        self.p_b_given_aba = p_b_given_aba
        self.enforce_p_at = enforce_p_at
        if not enforce_p_at:
            self.enforce_p_at = self.ctx_size // 2
        self.trans_dist = trans_dist
        self.seed = seed
        self.emis_dist = emis_dist
        self.output_dir = output_dir
        self.trans_fp = trans_fp
        self.emis_fp = emis_fp
        self.is_hmm = True if self.emis_dist or self.emis_fp else False
        self.mode = 'norep'  # to make p(AB...A) and p(B|AB...A) as close as possible to the theoretical values
        self.trans_mat, self.emis_mat = self._generate_matrices()
        self.eos = self.vocab_size - 1
        if self.is_hmm:
            # self.first_cat_vec = self.trans_mat.sum(dim=0)  # column sum estimate of unigram frequency
            self.first_cat_vec = self._get_left_eigenvector(self.trans_mat)
        else:
            self.first_tok_vec = self._get_left_eigenvector(self.trans_mat)

    @staticmethod
    def _get_left_eigenvector(matrix):
        matrix = matrix.to(DEVICE)
        eigenvalues, eigenvectors = torch.linalg.eig(matrix.T)  # Transpose to get left eigenvectors
        i = torch.argmin(torch.abs(eigenvalues - 1))
        # stationary_dist = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])  # Get dominant eigenvector
        stationary_dist = eigenvectors[:, i].real  # Get eigenvector with eigenval=1
        stationary_dist /= stationary_dist.sum()  # Normalize to probability distribution
        # matrix = matrix.to('cpu')
        stationary_dist = stationary_dist.to('cpu')

        return stationary_dist

    @staticmethod
    def _num_to_alpha(num):
        num_zeros = str(num).count('0')
        num_cubes = num_zeros//3
        alpha = str(num)[:-3*num_cubes] + {0: '', 1: 'K', 2: 'M', 3: 'B', 4: 'T'}[num_cubes] if num_cubes else str(num)
        return alpha

    def _name_data(self, split, size):
        # vocab_ctx_p(ab...a)_p(b|ab...a)_transition_emission_seed_{val|train}_size
        # if emission is none, then it's just token transition
        if self.trans_fp and self.emis_fp:  # HMM: define in terms of LD, CAT, and marginal
            _, v, num_cat, ld, cat, marginal, seed = self.trans_fp.split('.')[0].split('_')
            v, num_cat = v[1:], num_cat[1:]
            fn = '_'.join([
                f'v{str(self.vocab_size)}',
                f'c{str(self.ctx_size)}',
                f'pa{str(int(self.p_aba * 100))}',
                f'pb{str(int(self.p_b_given_aba * 100))}',
                ld,
                cat,
                marginal,
                f's{str(self.seed)}',
                split,
                self._num_to_alpha(size),
            ])
        elif self.trans_fp:  # MM
            info = self.trans_fp.split('.')[0].split('_')
            if len(info) == 5:  # bigram from CC100 e.g. v9999_c64_pa1_pb50_tnatural_enone_s1_train_100M
                fn = '_'.join([
                    f'v{str(self.vocab_size)}',
                    f'c{str(self.ctx_size)}',
                    f'pa{str(int(self.p_aba * 100))}',
                    f'pb{str(int(self.p_b_given_aba * 100))}',
                    f't{self.trans_dist}',
                    f'e{self.emis_dist if self.emis_dist else "none"}',
                    f's{str(self.seed)}',
                    split,
                    self._num_to_alpha(size),
                ])
            else:
                _, v, num_cat, ld, cat, marginal, seed = info
                v, num_cat = v[1:], num_cat[1:]
                fn = '_'.join([
                    f'v{str(self.vocab_size)}',
                    f'c{str(self.ctx_size)}',
                    f'pa{str(int(self.p_aba * 100))}',
                    f'pb{str(int(self.p_b_given_aba * 100))}',
                    ld,
                    cat,
                    marginal,
                    f's{str(self.seed)}',
                    split,
                    self._num_to_alpha(size),
                ])
        elif self.trans_dist and not (self.emis_dist or self.trans_fp or self.emis_fp):  # no ld situation
            fn = '_'.join([
                f'v{str(self.vocab_size)}',
                f'c{str(self.ctx_size)}',
                f'pa{str(int(self.p_aba * 100))}',
                f'pb{str(int(self.p_b_given_aba * 100))}',
                'nold',
                'nocat',
                self.trans_dist,
                f's{str(self.seed)}',
                split,
                self._num_to_alpha(size),
            ])

        else:
            fn = '_'.join([
                f'v{str(self.vocab_size)}',
                f'c{str(self.ctx_size)}',
                f'pa{str(int(self.p_aba * 100))}',
                f'pb{str(int(self.p_b_given_aba * 100))}',
                f't{self.trans_dist}',
                f'e{self.emis_dist if self.emis_dist else "none"}',
                f's{str(self.seed)}',
                split,
                self._num_to_alpha(size),
            ])
        fn += '.jsonl'
        return fn

    def _generate_matrices(self):
        trans_mat, emis_mat = None, None
        if self.trans_fp and self.emis_fp:
            trans_mat = self._read_matrix(self.trans_fp, alpha=0)
            emis_mat = self._read_matrix(self.emis_fp, alpha=1e-6)
            return trans_mat, emis_mat

        elif self.trans_fp:
            alpha = 1e-6 if 'cc100' in self.trans_fp else 0
            trans_mat = self._read_matrix(self.trans_fp, alpha=alpha)
            return trans_mat, emis_mat

        if self.is_hmm:
            if self.trans_dist == 'uniform':
                trans_mat = self._generate_uniform_matrix(
                    nrow=self.num_cat,
                    ncol=self.num_cat
                )
            elif self.trans_dist == 'gaussian':
                trans_mat = self._generate_gaussian_matrix(
                    sds_mean=2,  # number of destination cats vary by 2 on average
                    sds_sd=1,  # how much number of destination cats varies varies by 1 on average
                    min_sd=1,  # number of destination cats varies by 1 at least
                    nrow=self.num_cat,
                    ncol=self.num_cat,
                    max_mean=self.num_cat,  # mean destination category can be the last category
                    min_mean=0,  # or the first category
                    alpha=0
                )

            if self.emis_dist == 'uniform':
                emis_mat = self._generate_uniform_matrix(
                    nrow=self.num_cat,
                    ncol=self.vocab_size,
                )
            elif self.emis_dist == 'gaussian':
                emis_mat = self._generate_gaussian_matrix(
                    sds_mean=3,  # number of emittabel toks vary by 50 on average
                    sds_sd=2,  # how much number of emittabel toks varies varies by 50 on average
                    min_sd=5,  # number of destination cats varies by 5 at least
                    nrow=self.num_cat,
                    ncol=self.vocab_size,
                    max_mean=self.vocab_size,  # mean emission destination can be the last token
                    min_mean=0,  # or the first token
                    alpha=0.0001
                )
        else:  # each row will be identical
            if self.trans_dist == 'uniform':
                weights = torch.ones(self.vocab_size)
            elif self.trans_dist == 'gaussian':
                mean, sd = self.vocab_size / 2, self.vocab_size / 8  # half of the vocab will cover 25% of the data
                weights = torch.arange(self.vocab_size)
                weights = torch.exp(-0.5 * ((weights - mean) / sd) ** 2) / (sd * (2 * torch.pi) ** 0.5)
            elif self.trans_dist == 'zipfian':
                weights = torch.tensor(
                    [1.0 / (i + 1) for i in range(self.vocab_size)]
                )
            else:
                raise IOError('Please choose the target distribution from [zipfian, gaussian, uniform].')

            weights += 1e-6  # smooth
            trans_mat = (weights / torch.sum(weights)).expand(self.vocab_size, self.vocab_size)

        # else:  # regular Markov Model DEPRECATED, MM WILL BE READ THROUGH FILE INPUT
        #     if self.trans_dist == 'uniform':
        #         trans_mat = self._generate_uniform_matrix(
        #             nrow=self.vocab_size,
        #             ncol=self.vocab_size
        #         )
        #
        #     if self.trans_dist == 'zipfian':
        #         # trans_mat = self._generate_zipfian_matrix(  # trivial degenerate zipfian matrix
        #         #     nrow=self.vocab_size,
        #         #     ncol=self.vocab_size
        #         # )
        #
        #         # trans_mat = self._marginally_zipfian()
        #
        #         trans_mat = self._metropolis_hastings()
        #
        #     elif self.trans_dist == 'gaussian':
        #         trans_mat = self._generate_gaussian_matrix(
        #             sds_mean=50,  # number of destination tokens vary by 5 on average
        #             sds_sd=50,  # how much number of destination tokens varies varies by 50 on average
        #             min_sd=5,  # number of destination tokens varies by 5 at least
        #             nrow=self.vocab_size,
        #             ncol=self.vocab_size,
        #             max_mean=self.vocab_size,  # mean destination token can be the last token
        #             min_mean=0,  # or the first token
        #             alpha=0.0001
        #         )
        #
        #     elif self.trans_dist == 'natural':
        #         trans_mat = self._read_matrix(alpha=0.01)

        return trans_mat, emis_mat

    def _is_valid(self, token_id):
        return 0 <= token_id and token_id < self.vocab_size

    @staticmethod
    def _generate_uniform_matrix(nrow, ncol):
        mat = torch.ones((nrow, ncol))
        return mat

    @staticmethod
    def _generate_zipfian_matrix(nrow, ncol):
        zipf = torch.tensor([1/i for i in range(1, ncol+1)])
        mat = zipf.expand(nrow, -1)
        return mat

    @staticmethod
    def _generate_gaussian_matrix(sds_mean, sds_sd, min_sd, nrow, ncol, min_mean, max_mean, alpha=0):
        print('Creating Gaussian transition matrix...')
        # uniformly sample the mean for per-row normal distribution
        means = torch.tensor([random.randint(min_mean, max_mean) for _ in range(nrow)])
        # sample the sd from normal distribution for per-row normal distribution
        sds = torch.tensor([min((max(min_sd, s), int(ncol / 2))) for s in np.random.normal(sds_mean, sds_sd, nrow)])
        x_t = torch.arange(0, ncol)
        mat = torch.exp(-0.5 * ((x_t - means.unsqueeze(1)) / sds.unsqueeze(1)) ** 2) / (
                sds.unsqueeze(1) * (2 * torch.pi) ** 0.5)
        print('Done!')
        if alpha:
            mat += alpha
        return mat

    def _read_matrix(self, fp, alpha):
        with open(fp, 'rb') as f:
            mat = pickle.load(f)
        if alpha:
            mat += alpha

        mat /= mat.sum(dim=1, keepdim=True)

        if 'emis' in fp or 'cc100' in fp:
            old, self.vocab_size = self.vocab_size, mat.shape[1]
            if self.vocab_size != old:
                print(f"Warning: fixing the vocab size from {old} to {self.vocab_size}")

        if ('natural' in self.trans_fp or 'cc100' in self.trans_fp) and self.vocab_size != 50257:
            print(f"Warning: using natural transition probas, but vocab size is {self.vocab_size}")

        return mat

    def _sample(self, prev_tok, prev_cat=None):
        cat = None
        if self.is_hmm:  # Hidden Markov Model
            cat = int(
                torch.multinomial(
                    self.trans_mat[prev_cat], num_samples=1,
                    replacement=True
                )[0]
            )  # transition from the previous cat
            tok = torch.multinomial(
                self.emis_mat[cat], num_samples=1,
                replacement=True
            )  # emission from the current cat
        else:  # regular Markov Model
            tok = torch.multinomial(
                self.trans_mat[prev_tok], num_samples=1,
                replacement=True
            )[0]  # transition from the previous token

        return int(tok), cat

    def _sample_from_cands(self, cands, prev_tok, prev_cat, mode='include'):
        cat = None
        if mode == 'include':  # only consider the cands
            if self.is_hmm:
                cat = int(
                    torch.multinomial(
                        self.trans_mat[prev_cat], num_samples=1,
                        replacement=True
                    )[0]
                )  # transition from the previous cat
                probas = self.emis_mat[cat][torch.tensor(cands, dtype=torch.int64)]
            else:
                probas = self.trans_mat[prev_tok][torch.tensor(cands, dtype=torch.int64)]
            w_i = torch.multinomial(
                probas, num_samples=1,
                replacement=True
            )[0]  # transition from the previous tok
            w_i = cands[int(w_i)]
        elif mode == 'exclude':  # only consider the items NOT in the cands
            if self.is_hmm:
                cat = int(
                    torch.multinomial(
                        self.trans_mat[prev_cat], num_samples=1,
                        replacement=True
                    )[0]
                )  # transition from the previous cat
                probas = self.emis_mat[cat].clone()
            else:
                probas = self.trans_mat[prev_tok].clone()
            probas[torch.tensor(cands, dtype=torch.int64)] = 0
            w_i = torch.multinomial(
                probas, num_samples=1,
                replacement=True
            )[0]
        return int(w_i), cat

    def generate_sequences(self, split, size):
        # set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        num_seq = int(size//self.ctx_size)
        # seqs = []
        pbar = tqdm(total=num_seq)

        output_fn = self._name_data(split, size)
        out = open(os.path.join(self.output_dir, output_fn), 'w')

        for _ in range(num_seq):
            seq = []
            unigrams = set()  # keep track of unigrams at and including i - 2 for word_i to choose from
            bigrams = defaultdict(lambda: set())

            # sample the first 2
            first_cat = None
            if self.is_hmm:
                first_cat = torch.multinomial(self.first_cat_vec, num_samples=1, replacement=True)[0]
                first = int(torch.multinomial(self.emis_mat[first_cat], num_samples=1, replacement=True)[0])
                while not self._is_valid(first):
                    first = int(torch.multinomial(self.emis_mat[first_cat], num_samples=1, replacement=True)[0])
            else:
                first = int(
                    torch.multinomial(
                        self.first_tok_vec, num_samples=1, replacement=True
                    )[0]
                )  # not a transition yet, so sample from a uniform dist
                while not self._is_valid(first):
                    first = int(
                        torch.multinomial(
                            self.trans_mat[self.eos], num_samples=1, replacement=True
                        )[0]
                    )  # not a transition yet, so sample from a uniform dist
            second, second_cat = self._sample(first, first_cat)
            while any([
                first == second,
                not self._is_valid(second)
            ]):  # make sure to make AB and not AA
                second, second_cat = self._sample(first, first_cat)
            seq.extend([first, second])
            bigrams[first].add(second)  # binary variable
            prev2_tok, prev2_cat, prev_tok, prev_cat = first, first_cat, second, second_cat

            w_i, w_i_cat = None, None  # in case of reference problem
            # sample the first half
            for i in range(2, self.enforce_p_at):
                unigrams.add(prev2_tok)
                if self.mode == 'random':
                    w_i, w_i_cat = self._sample(prev_tok, prev_cat)
                    while any([
                        not self._is_valid(w_i)
                    ]):
                        w_i, w_i_cat = self._sample(prev_tok, prev_cat)
                elif self.mode == 'norep':
                    w_i, w_i_cat = self._sample_from_cands(list(unigrams), prev_tok, prev_cat, mode='exclude')
                    while any([
                        w_i in unigrams,
                        not self._is_valid(w_i)
                    ]):
                        print([
                            self.vocab_size,
                            w_i in unigrams,
                            self._is_valid(w_i) == False,
                            w_i,
                            len(seq),
                            unigrams,
                        ])
                        w_i, w_i_cat = self._sample_from_cands(list(unigrams), prev_tok, prev_cat, mode='exclude')

                bigrams[prev_tok].add(w_i)
                prev2_tok = prev_tok
                prev_tok = w_i
                seq.append(w_i)

            # sample the second half
            effective_p_aba = self.p_aba / (1 + self.p_b_given_aba)
            for i in range(self.enforce_p_at, self.ctx_size):  # time step i
                unigrams.add(prev2_tok)
                is_ABA = bigrams[prev_tok].difference({prev_tok})
                make_ABA = random.random() < effective_p_aba  # to enforce p(AB...A) constraint
                make_ABAB = random.random() < self.p_b_given_aba  # to enforce p(AB...AB) constraint
                if is_ABA:  # if the previous token is A of AB...A
                    if make_ABAB:
                        # if make_ABA:  # given 12345, AnyAny...56 is ABAB, and AnyAny...6 is ABA
                        cands = list(bigrams[prev_tok].difference({prev_tok}))
                        w_i, w_i_cat = self._sample_from_cands(cands, prev_tok, prev_cat)
                        # elif not make_ABA:  # not possible
                    else:  # not ABAB
                        if make_ABA:  # given 12345, AnyAny...56 is NOT ABAB, and AnyAny...6 is ABA
                            # i.e. pick a token that's in unigram but not in bigram continuation
                            cands = list(unigrams.difference(bigrams[prev_tok]).difference({prev_tok}))
                            w_i, w_i_cat = self._sample_from_cands(cands, prev_tok, prev_cat)
                        elif not make_ABA:
                            # w_i, w_i_cat = self._sample(prev_tok, prev_cat)
                            w_i, w_i_cat = self._sample_from_cands(
                                list(unigrams.union(bigrams[prev_tok])),
                                prev_tok,
                                prev_cat,
                                mode='exclude'
                            )
                            while any([
                                w_i in bigrams[prev_tok],
                                w_i in unigrams,
                                not self._is_valid(w_i)
                            ]):
                                w_i, w_i_cat = self._sample(prev_tok, prev_cat)
                else:  # if the previous token is NOT A of AB...A
                    if make_ABA:  # given 12345, AnyAny...6 is ABA
                        cands = list(unigrams.difference({prev_tok}))
                        w_i, w_i_cat = self._sample_from_cands(cands, prev_tok, prev_cat)
                    else:  # given 12345, AnyAny...6 is NOT ABA
                        # w_i, w_i_cat = self._sample(prev_tok, prev_cat)
                        w_i, w_i_cat = self._sample_from_cands(
                            list(unigrams),
                            prev_tok,
                            prev_cat,
                            mode='exclude'
                        )
                        while w_i in unigrams or not self._is_valid(w_i):
                            w_i, w_i_cat = self._sample(prev_tok, prev_cat)
                w_i = int(w_i)
                seq.append(w_i)
                bigrams[prev_tok].add(w_i)
                prev2_tok, prev2_cat, prev_tok, prev_cat = prev_tok, prev_cat, w_i, w_i_cat
            # if val:
            #     seq = [i + train_vocab_size for i in seq]
            # if self.output_dir:
            out.write(json.dumps(dict({'input_ids': seq}))+'\n')
            # else:
            #     seqs.append(seq[:ctx_size])
            pbar.update()

    def _marginally_zipfian(self):
        n = 1000
        # permute 1-10000 and index is the ranking
        # this is the order of frequency (first element in this list is most frequent)
        means = torch.arange(1, n+1)
        # means = means[torch.randperm(means.size()[0])]
        means = means.flip(0)
        sds = torch.arange(1, n+1)/2  # scale sds with frequency!
        x_t = torch.arange(0, n)
        mat = torch.exp(-0.5 * ((x_t - means.unsqueeze(1)) / sds.unsqueeze(1)) ** 2) / (
                sds.unsqueeze(1) * (2 * torch.pi) ** 0.5)
        # if alpha:
        #     mat += alpha
        return mat
    def _metropolis_hastings(self):
        n = self.vocab_size
        # n = 1000

        # Zipfian distribution
        ranks = torch.arange(1, n + 1)
        pi = 1 / ranks
        pi /= pi.sum()
        # pi = pi[torch.randperm(pi.size()[0])]  # avoid self loop

        # Uniform proposal matrix (zero diagonal)
        # Q = np.ones((n, n)) / (n - 1)
        # np.fill_diagonal(Q, 0)

        # Gaussian proposal matrix
        means = torch.arange(0, n)
        # sample the sd from normal distribution for per-row normal distribution
        sds = torch.tensor([100]*n)
        x_t = torch.arange(0, n)
        Q = torch.exp(-0.5 * ((x_t - means.unsqueeze(1)) / sds.unsqueeze(1)) ** 2) / (
                sds.unsqueeze(1) * (2 * torch.pi) ** 0.5)
        # Q = Q.numpy()

        # Broadcasted min(pi[j]/pi[i], 1)
        ratio = pi[np.newaxis, :] / pi[:, np.newaxis]
        accept_prob = np.minimum(1, ratio)

        # Metropolis-Hastings matrix
        P = Q * accept_prob
        P = P.numpy()
        # Set diagonals to ensure rows sum to 1
        np.fill_diagonal(P, 0)  # Re-zero diagonal in case of float errors
        row_sums = P.sum(axis=1)
        # np.fill_diagonal(P, 1 - row_sums)
        P /= row_sums

        return torch.tensor(P)

    def get_theoretical_entropy_rate(self):
        return

    def get_empirical_entropy_rate(self):
        effective_p_aba = self.p_aba / (1 + self.p_b_given_aba)

        if self.is_hmm:  # first combine transition and emission into 1
            trans = self.emis_mat.T @ self.trans_mat @ self.emis_mat
        else:
            trans = self.trans_mat  # avoid cloning for memory footprint

        trans = trans / trans.sum(dim=1, keepdims=True)
        # trans = trans.numpy()

        stationary_dist = self._get_left_eigenvector(trans)
        base_entropy = np.sum(-stationary_dist * np.sum(trans * np.log2(trans, where=(trans > 0)), axis=1))

        # when ABA, weighted sum of P(B|ABA) and base entropy
        copy_entropy = self.p_b_given_aba * np.log2(self.p_b_given_aba) + (1 - self.p_b_given_aba) * base_entropy
        # when not ABA, just base entropy
        # weighted sum of ABA entropy and non-ABA entropy is the entropy rate
        H_copy = effective_p_aba * copy_entropy + (1 - effective_p_aba) * base_entropy
        H_base = base_entropy
        # dip = base_entropy - copy_entropy
