import argparse
import random
import torch, os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.groupiness_loss import GroupinessLoss
from scipy.stats import kstest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MarkovModel:
    def __init__(self, vocab_size, target_dist_shape, seed, zipf_lambda=1.0):
        """
        Initialize a Hidden Markov Model with:
        - num_categories (c): Number of hidden states
        - vocab_size (v): Size of vocabulary
        - alpha: Parameter for Zipf distribution
        - target_dist_shape: shape of the marginal token distribution
        """
        self.vocab_size = vocab_size
        self.zipf_lambda = zipf_lambda
        self.target_dist_shape = target_dist_shape
        self.seed = seed
        self.groupiness = None  # added with generate()

        # Model parameters

        self.token_transition_matrix = None
        self.token_distribution = None
        self.target_token_distribution = self._generate_target_distribution()
        self.final_dist_loss = None
        self.final_similarity_loss = None

        self.diversity_metrics = None
        self.observed_tokens = None
        self.sequence_token_distribution = None

    def load(self, token_trans_fp):
        with open(token_trans_fp, 'rb') as f:
            self.token_transition_matrix = pickle.load(f)
        self.token_transition_matrix = (self.token_transition_matrix) / self.token_transition_matrix.sum(dim=1, keepdim=True)
        self.token_distribution = self._calculate_stationary_distribution(self.token_transition_matrix)


    def generate(
            self,
            distribution_weight, entropy_weight, pointy_weight, groupy_weight, #similarity_weight,
            target_entropy=None, target_within_group_sim=None, target_across_group_sim=None,
            max_iterations=1000, learning_rate=0.1, device=device
    ):
        """
        Generate HMM parameters with specific properties:
        1. Marginal token distribution follows desired distribution (e.g., Zipf)
        2. Token-to-token transitions have diverse rows

        Uses GPU if available for faster computation.
        """
        # Set device to GPU if available, otherwise CPU
        print(f"Using device: {device}")

        # Set all the seeds here
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        # np.random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        if target_within_group_sim:
            if target_within_group_sim <= 0.1:
                self.groupiness = 'low'
            elif target_within_group_sim <= 0.3:
                self.groupiness = 'mid'
            else:
                self.groupiness = 'high'

        # Initialize parameters with PyTorch tensors on the appropriate device
        token_transition_matrix = torch.rand(self.vocab_size, self.vocab_size, requires_grad=True, device=device)

        # Setup optimizer
        optimizer = torch.optim.Adam([token_transition_matrix], lr=learning_rate)

        # initialize group loss
        groupiness_optimizer = GroupinessLoss(num_groups=50, group_size=200, v=self.vocab_size)

        # Optimization loop
        for iteration in range(max_iterations):
            # Normalize matrices to be valid probability distributions
            t_matrix = torch.nn.functional.softmax(token_transition_matrix, dim=1)

            # Calculate token distribution
            token_dist = self._calculate_stationary_distribution(t_matrix)

            # Distribution loss: how well we match the target token distribution
            """ORIGINAL APPROACH"""
            # dist_loss = torch.sum((token_dist - self.target_token_distribution) ** 2)
            """SORTING APPROACH"""
            # dist_loss = torch.sum(  # sort to match the shape
            #     (torch.sort(token_dist).values - torch.sort(self.target_token_distribution).values) ** 2
            # )
            """KL DIVERGENCE APPROACH"""
            # Sort both distributions first
            sorted_pred = torch.sort(token_dist).values
            sorted_target = torch.sort(self.target_token_distribution).values
            # Small epsilon to avoid log(0)
            epsilon = 1e-8
            # KL divergence on sorted distributions
            dist_loss = torch.sum(sorted_target * torch.log((sorted_target + epsilon) / (sorted_pred + epsilon)))

            # Similarity Loss: penalize similarity between rows of token transition matrix
            # Calculate all pairwise similarities in one vectorized operation
            # normalized_rows = t_matrix / (torch.norm(t_matrix, dim=1, keepdim=True) + 1e-10)
            # similarity_matrix = torch.matmul(normalized_rows, normalized_rows.t())

            # We want off-diagonal elements to be small (dissimilar rows)
            # Create a mask to select only off-diagonal elements
            # mask = torch.ones_like(similarity_matrix, device=device) - torch.eye(self.vocab_size, device=device)
            # similarity_loss = torch.sum(similarity_matrix * mask) / (self.vocab_size * (self.vocab_size - 1))

            # If target entropy is specified, add the difference as loss
            delta_entropy_loss = 0
            if target_entropy:
                delta_entropy_loss = (self._get_entropy(t_matrix, token_dist) - target_entropy) ** 2

            # pointy transition distribution loss
            max_transition_probs = torch.max(t_matrix, dim=1).values
            # Penalize these maximum probabilities (encourage diversity)
            # We can use various penalties - this one penalizes the square of probabilities
            pointy_loss = torch.mean(max_transition_probs)

            # local, global approach
            # current_groupiness = 0
            # if target_groupiness:
            #     current_groupiness, global_sim, local_sim = self._local_vs_global_groupiness_loss(t_matrix)
            #     groupy_loss = (target_groupiness - current_groupiness)**2
            if target_within_group_sim and target_across_group_sim:
                within, across = groupiness_optimizer.get_groupiness_loss(transition_matrix=t_matrix)
                within_loss = (within - target_within_group_sim)**2
                across_loss = (across - target_across_group_sim)**2

            # Total loss combines distribution matching and diversity
            total_loss = distribution_weight * dist_loss +\
                         entropy_weight * delta_entropy_loss +\
                         pointy_weight * pointy_loss +\
                         groupy_weight * within_loss +\
                         groupy_weight * across_loss
                         # groupy_weight * groupy_loss
                        # similarity_weight * similarity_loss +\

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Print progress occasionally
            if (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}: Dist. Loss = {dist_loss.item():.6f}, "
                      # f"Sim. Loss = {similarity_loss.item():.6f}, "
                      f"Ent. Loss = {delta_entropy_loss.item():.6f}, "
                      f"Pnt. Loss = {pointy_loss.item():.6f}, "
                      f"Within. = {within.item():.6f}, "
                      f"Across. = {across.item():.6f} "
                      )

        # Store final parameters (on CPU for easier handling if needed)
        self.token_transition_matrix = torch.nn.functional.softmax(token_transition_matrix, dim=1).detach()
        self.token_distribution = self._calculate_stationary_distribution(self.token_transition_matrix)
        self.final_dist_loss = dist_loss
        # self.final_similarity_loss = similarity_loss

    def save(self):
        self.token_transition_matrix = self.token_transition_matrix.detach().cpu()

        trans_fn = self._name_matrix('trans')
        with open(os.path.join('data', trans_fn), 'wb') as f:
            pickle.dump(self.token_transition_matrix, f)

    def _local_vs_global_groupiness_loss(self, matrix, batch_size=1000, k=10):
        """
        P: [n_tokens, n_tokens] transition matrix (row-stochastic)
        batch_size: number of rows to sample at once
        k: number of neighbors for "local similarity" part
        """

        # 1. Randomly sample batch of rows
        n = matrix.size(0)
        idx = torch.randperm(n, device=matrix.device)[:batch_size]
        batch = torch.nn.functional.normalize(matrix[idx], p=2, dim=1)  # [batch_size, d]

        # 2. Compute cosine similarity within the batch
        sim = batch @ batch.T  # [batch_size, batch_size]
        sim.fill_diagonal_(-1.0)  # exclude self-similarity

        # 3. Local similarity: encourage k-nearest neighbors to be similar
        topk_vals, _ = torch.topk(sim, k=k, dim=1)  # [batch_size, k]
        local_similarity = topk_vals.mean()

        # 4. Global similarity: discourage average similarity across all pairs
        global_similarity = sim.mean()

        # 5. Final loss: maximize local similarity, minimize global similarity
        loss = global_similarity - local_similarity  # want global low, local high

        return loss, global_similarity, local_similarity

    @staticmethod
    def _get_entropy(transition=None, stationary=None):
        if transition is None:
            return torch.sum(
                torch.log2(stationary + 1e-6) * stationary
            )
        return -torch.dot(
            # per-row conditional entropy for each token; 1d vector of V
            torch.sum(
                torch.log2(transition + 1e-6) * transition, dim=1
            ),
            # weight by the stationary token distribution
            stationary  # 1d vector of V
        )

    def _name_matrix(self, mat):
        fn = '_'.join([
            mat,
            f'v{self.vocab_size}',
            'h0',  # no category
            'ld',
            f'{self.groupiness}cat',
            self.target_dist_shape,
            f's{self.seed}'
        ]) + '.pkl'

        return fn

    def _generate_target_distribution(self):
        if self.target_dist_shape == 'zipfian':
            weights = torch.tensor(
                [1.0 / ((i + 1) ** self.zipf_lambda) for i in range(self.vocab_size)],
                device=device
            )
        elif self.target_dist_shape == 'gaussian':
            mean, sd = self.vocab_size/2, self.vocab_size/8  # half of the vocab will cover 25% of the data
            weights = torch.arange(self.vocab_size, device=device)
            weights = torch.exp(-0.5 * ((weights - mean) / sd) ** 2) / (sd * (2 * torch.pi) ** 0.5)
        elif self.target_dist_shape == 'uniform':
            weights = torch.ones(self.vocab_size, device=device)
        else:
            raise IOError('Please choose the target distribution from [zipfian, gaussian, uniform].')

        weights += 1e-6  # smooth

        return weights / torch.sum(weights)


    def _calculate_stationary_distribution(self, transition_matrix, num_iterations=50, eigenvector=False):
        if eigenvector:
            eigenvalues, eigenvectors = torch.linalg.eig(transition_matrix.T)

            # Find the index of eigenvalue closest to 1
            idx = torch.argmin(torch.abs(eigenvalues - 1))

            # Get the corresponding eigenvector and normalize
            stationary = eigenvectors[:, idx].real
            return stationary / torch.sum(stationary)
        """
        Calculate the stationary distribution of a Markov chain using power iteration.
        This is a vectorized, stable approach.
        """
        # Start with uniform distribution
        state_dist = torch.ones(self.vocab_size, device=device) / self.vocab_size

        # Power iteration to converge to stationary distribution
        for _ in range(num_iterations):
            state_dist = torch.matmul(state_dist, transition_matrix)

        return state_dist

    def generate_sequence(self, length):
        """
        Generate a sequence of tokens from the HMM.
        """
        if self.token_transition_matrix is None:
            raise ValueError("Model parameters not generated yet. Call generate() first.")

        # Initialize with a random state
        # current_state = torch.multinomial(torch.ones(self.num_categories) / self.num_categories, 1).item()
        prev_tok = torch.multinomial(self.token_distribution, 1).item()

        observed_tokens = [prev_tok]

        for _ in range(length):
            # Generate token based on previous token
            curr_tok = torch.multinomial(self.token_transition_matrix[prev_tok], 1).item()
            observed_tokens.append(curr_tok)
            prev_tok = curr_tok

        return observed_tokens

    def get_stats(self):
        # entropy
        token_dist = self._calculate_stationary_distribution(transition_matrix=self.token_transition_matrix)
        entropy = self._get_entropy(self.token_transition_matrix, token_dist)
        entropy_target = self._get_entropy(stationary=self.target_token_distribution)

        # KL divergence
        sorted_pred = torch.sort(token_dist).values
        sorted_target = torch.sort(self.target_token_distribution).values
        epsilon = 1e-8
        kl = torch.sum(sorted_target * torch.log((sorted_target + epsilon) / (sorted_pred + epsilon)))
        ks = kstest(sorted_target, sorted_pred)

        # similarity metrics
        groupiness_optimizer = GroupinessLoss(num_groups=50, group_size=200, v=self.vocab_size)
        within, across = groupiness_optimizer.get_groupiness_loss(transition_matrix=self.token_transition_matrix)

        return {'entropy': entropy, 'entropy_target': entropy_target, 'kl': kl, 'within': within, 'across': across, 'ks':ks}

    def evaluate(self, plot=False):
        """
        Evaluate the HMM properties and visualize results.
        """
        if self.token_transition_matrix is None:
            raise ValueError("Model parameters not generated yet. Call generate() first.")

        # Analyze token transition matrix diversity
        diversity_metrics = self._analyze_token_transition_diversity()

        # Generate sample sequence
        sequence_length = 1000
        observed_tokens = self.generate_sequence(sequence_length)

        # Calculate token distribution in generated sequence
        token_counts = torch.bincount(torch.tensor(observed_tokens), minlength=self.vocab_size).float()
        token_freq = token_counts / torch.sum(token_counts)

        if plot:
            # Plot target vs actual token distribution
            plt.figure(figsize=(10, 6))
            plt.plot(self.target_token_distribution.numpy(), label='Target (Zipf)')
            plt.plot(sorted(self.token_distribution.numpy(), reverse=True), label='Model Distribution')
            plt.plot(sorted(token_freq.numpy(), reverse=True), label='Generated Sequence')
            plt.title('Token Distribution: Target vs Actual vs Generated')
            plt.xlabel('Token ID')
            plt.ylabel('Probability')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot token transition matrix
            # plt.figure(figsize=(8, 8))
            # sns.heatmap(self.token_transition_matrix.numpy(), cmap='viridis')
            # plt.title('Token Transition Matrix')
            # plt.xlabel('Next Token')
            # plt.ylabel('Previous Token')
            # plt.show()

            # Visualize row diversity with similarity matrix
            # similarity_matrix = diversity_metrics['similarity_matrix']
            # plt.figure(figsize=(8, 8))
            # sns.heatmap(similarity_matrix.numpy(), cmap='coolwarm', vmin=0, vmax=1)
            # plt.title('Cosine Similarity Between Token Transition Rows')
            # plt.xlabel('Token ID')
            # plt.ylabel('Token ID')
            # plt.show()

        self.diversity_metrics = diversity_metrics
        self.observed_tokens = observed_tokens
        self.sequence_token_distribution = token_freq

    def _analyze_token_transition_diversity(self):
        """
        Analyze the diversity of rows in the token transition matrix.
        """
        # Calculate pairwise cosine similarity between rows
        normalized_rows = self.token_transition_matrix / (
                    torch.norm(self.token_transition_matrix, dim=1, keepdim=True) + 1e-10)
        similarity_matrix = torch.matmul(normalized_rows, normalized_rows.t())

        # Calculate statistics
        # Exclude diagonal (self-similarity is always 1)
        mask = torch.ones_like(similarity_matrix, device=device) - torch.eye(self.vocab_size, device=device)
        off_diag_similarities = similarity_matrix * mask

        mean_similarity = torch.sum(off_diag_similarities) / (self.vocab_size * (self.vocab_size - 1))
        max_similarity = torch.max(off_diag_similarities)

        # The lower the mean similarity, the more diverse the rows
        print(f"Mean cosine similarity between rows: {mean_similarity.item():.4f}")
        print(f"Max cosine similarity between rows: {max_similarity.item():.4f}")

        return {
            'similarity_matrix': similarity_matrix,
            'mean_similarity': mean_similarity.item(),
            'max_similarity': max_similarity.item()
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab_size', default=10_000, type=int,
                        help='vocabulary size')
    parser.add_argument('-d', '--target_distribution', required=True, type=str,
                        help='target marginal token distribution: [uniform, gaussian, zipfian]')
    parser.add_argument('-s', '--seed', required=True, type=int,
                        help='random seed')
    parser.add_argument('-e', '--target_entropy', default=None, type=float,
                        help='target entropy')
    parser.add_argument('-wi', '--target_within_group_sim', default=None, type=float,
                        help='target_within_group_sim: high (~0.5) means strong group structure')
    parser.add_argument('-ac', '--target_across_group_sim', default=None, type=float,
                        help='target_across_group_sim: set to 0.1')
    parser.add_argument('-dw', '--distribution_weight', default=100, type=float,
                        help='how much to penalize the divergence from specified shape of marginal distribution')
    # parser.add_argument('-sw', '--similarity_weight', default=2, type=float,
    #                     help='how much to penalize similar token transitions')
    parser.add_argument('-ew', '--entropy_weight', default=0.001, type=float,
                        help='how much to penalize the divergence from specified entropy')
    parser.add_argument('-pw', '--pointy_weight', default=0.1, type=float,
                        help='how much to penalize pointy token transition')
    parser.add_argument('-gw', '--groupy_weight', default=0.1, type=float,
                        help='how much to penalize pointy token transition')
    parser.add_argument('-m', '--max_iter', default=1000, type=int,
                        help='maximum iteration for distribution optimization')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='learning_rate')
    args = parser.parse_args()

    # Create and generate HMM
    mm = MarkovModel(
        vocab_size=args.vocab_size,
        target_dist_shape=args.target_distribution,
        seed=args.seed
    )
    mm.generate(
        distribution_weight=args.distribution_weight,
        # similarity_weight=args.similarity_weight,
        target_entropy=args.target_entropy,
        target_within_group_sim=args.target_within_group_sim,
        target_across_group_sim=args.target_across_group_sim,
        entropy_weight=args.entropy_weight,
        pointy_weight=args.pointy_weight,
        groupy_weight=args.groupy_weight,
        max_iterations=args.max_iter,
        learning_rate=args.learning_rate
    )

    print("\nFinal losses:")
    print(f"Distribution Loss: {mm.final_dist_loss:.6f}")
    # print(f"Similarity Loss: {mm.final_similarity_loss:.6f}")

    # Evaluate
    mm.evaluate()

    # Check if rows are different (sample a few)
    print("\nToken transition matrix row differences (should be non-zero):")
    for i in range(min(3, args.vocab_size)):
        for j in range(i + 1, min(4, args.vocab_size)):
            diff = torch.norm(mm.token_transition_matrix[i] - mm.token_transition_matrix[j])
            print(f"Difference between rows {i} and {j}: {diff.item():.4f}")

    mm.save()

if __name__ == "__main__":
    main()

# mm = MarkovModel(vocab_size=9999, seed=1, target_dist_shape='zipfian')
# mm.load('data\\cc100_v10000_100M_bigram_tensor.pkl')
# -torch.sum(torch.log2(mm.token_transition_matrix+1e-6) * mm.token_transition_matrix, axis=1) @ mm.token_distribution

