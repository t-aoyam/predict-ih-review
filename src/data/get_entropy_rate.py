import numpy as np
import pickle
import torch, random, os
from scipy.linalg import eig

def get_entropy_rate(p_aba, p_b_given_aba, dist, v, sd=50, fp=None, alpha=0.01):
    p_aba /= 2
    if dist == 'uniform':
        base_entropy = np.log2(v)
    elif dist == 'normal':
        means = [random.randint(0, v) for i in range(v)]
        sds = [min((max(5, s), v / 2)) for s in np.random.normal(50, 50, v)]
        print('Creating Gaussian transition matrix...')
        means = torch.tensor(means)
        sds = torch.tensor(sds)
        # x_t = torch.arange(0, v).unsqueeze(0)
        # x_t = torch.tensor(torch.arange(0, v).expand(v, v))
        x_t = torch.arange(0, v)
        trans = torch.exp(-0.5 * ((x_t - means.unsqueeze(1)) / sds.unsqueeze(1)) ** 2) / (sds.unsqueeze(1) * (2 * torch.pi) ** 0.5)
        trans += alpha  # smoothing
        print('Done!')
        means, sds, x_t = None, None, None
        trans = trans / trans.sum(dim=1, keepdim=True)
        unigram = trans.sum(dim=0)/trans.sum()
        # trans = torch.where(trans == 0, torch.tensor(float('nan')), trans)
        log_trans = torch.log2(1/trans)
        log_trans[log_trans==torch.inf] = 0
        trans *= log_trans
        # entropies = torch.nan_to_num(entropies, nan=0.0)  # convert to NaN to 0 so that it's removed in the next ops
        # entropies *= unigram
        # base_entropy = entropies.sum(dim=1)
        base_entropy = sum(unigram * trans.sum(dim=1))  # weight by unigram probas

        # base_entropy = np.log2(2*np.pi*np.e*sd**2)/2
    elif dist == 'natural':
        # conditional entropy for all toks?
        with open(fp, 'rb') as f:
            trans = pickle.load(f)
        trans += alpha  # smoothing
        # row margin can be thought of as unigram probas (row total)
        trans = trans / trans.sum(dim=1, keepdim=True)
        # trans = torch.where(trans == 0, torch.tensor(float('nan')), trans)
        unigram = trans.sum(dim=0)/trans.sum()
        # entropies = trans * torch.log2(1/trans)  # P(x) * log2 P(x)
        trans *= torch.log2(1/trans)
        # entropies = torch.nan_to_num(entropies, nan=0.0)  # convert to NaN to 0 so that it's removed in the next ops
        trans = torch.nan_to_num(trans, nan=0.0)
        # entropies[entropies == torch.inf] = 0
        # trans[trans == torch.inf] = 0
        # base_entropy = sum(unigram * entropies.sum(dim=1))  # weight by unigram probas
        base_entropy = sum(unigram * trans.sum(dim=1))  # weight by unigram probas


    # when ABA, weighted sum of P(B|ABA) and base entropy
    copy_entropy = p_b_given_aba*np.log2(p_b_given_aba) + (1-p_b_given_aba)*base_entropy
    # when not ABA, just base entropy
    # weighted sum of ABA entropy and non-ABA entropy is the entropy rate
    H_copy = p_aba * copy_entropy + (1-p_aba) * base_entropy
    H_base = base_entropy
    # dip = base_entropy - copy_entropy
    return round(float(H_base), 3), round(float(H_copy), 3), round(float(base_entropy), 3), round(float(copy_entropy), 3)

def get_entropy_rate_eigen(p_aba, p_b_given_aba, dist, v, sd=50, fp=None, alpha=0.01):
    p_aba /= 2
    if dist == 'uniform':
        pass
    elif dist == 'normal':
        means = [random.randint(0, v) for i in range(v)]
        sds = [min((max(5, s), v / 2)) for s in np.random.normal(50, 50, v)]
        print('Creating Gaussian transition matrix...')
        means = torch.tensor(means)
        sds = torch.tensor(sds)
        # x_t = torch.arange(0, v).unsqueeze(0)
        # x_t = torch.tensor(torch.arange(0, v).expand(v, v))
        x_t = torch.arange(0, v)
        trans = torch.exp(-0.5 * ((x_t - means.unsqueeze(1)) / sds.unsqueeze(1)) ** 2) / (sds.unsqueeze(1) * (2 * torch.pi) ** 0.5)
        print('Done!')
    elif dist == 'natural':
        with open(fp, 'rb') as f:
            trans = pickle.load(f)
    trans += alpha  # smoothing
    trans = trans / trans.sum(dim=1, keepdims=True)
    trans = trans.numpy()

    eigenvalues, eigenvectors = eig(trans.T)  # Transpose to get left eigenvectors
    stationary_dist = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])  # Get dominant eigenvector
    stationary_dist /= stationary_dist.sum()  # Normalize to probability distribution
    base_entropy = np.sum(-stationary_dist * np.sum(trans * np.log2(trans, where=(trans > 0)), axis=1))

    # when ABA, weighted sum of P(B|ABA) and base entropy
    copy_entropy = p_b_given_aba*np.log2(p_b_given_aba) + (1-p_b_given_aba)*base_entropy
    # when not ABA, just base entropy
    # weighted sum of ABA entropy and non-ABA entropy is the entropy rate
    H_copy = p_aba * copy_entropy + (1-p_aba) * base_entropy
    H_base = base_entropy
    # dip = base_entropy - copy_entropy

    return round(float(H_base), 3), round(float(H_copy), 3), round(float(base_entropy), 3), round(float(copy_entropy), 3)

# print(get_entropy_rate(
#     p_aba=0.9,
#     p_b_given_aba=0.9,
#     dist='normal',
#     v=10000,
#     fp = os.path.join('data', 'cc100_v10000_100M_bigram_tensor.pkl')
# ))

# print(get_entropy_rate(
#     p_aba=0.9,
#     p_b_given_aba=0.9,
#     dist='natural',
#     v=9999,
#     fp = os.path.join('data', 'cc100_v10000_100M_bigram_tensor.pkl')
# ))


print(get_entropy_rate_eigen(
    p_aba=0.9,
    p_b_given_aba=0.9,
    dist='natural',
    v=9999,
    fp = os.path.join('data', 'cc100_v10000_100M_bigram_tensor.pkl')
))

for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for b in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"P(AB...A): {a}, P(B|AB...A): {b}")
        print(get_entropy_rate(
            p_aba=a,
            p_b_given_aba=b,
            dist='natural',
            v=9999,
            fp=os.path.join('data', 'cc100_v10000_100M_bigram_tensor.pkl')
        ))