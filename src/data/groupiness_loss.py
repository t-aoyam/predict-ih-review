import torch

class GroupinessLoss():
    def __init__(self, num_groups, group_size, v):
        self.num_groups = num_groups
        self.group_size = group_size
        self.v = v
        self.group_indices = self._create_groups_with_overlap()

    def _create_groups_with_overlap(self):
        """
        Create overlapping groups from a transition matrix. Each group will contain `group_size`
        rows, and rows will overlap with each other based on `overlap_factor`.
        """

        # Calculate the step size for each group
        # if num_groups = 50 and group size = 300, and v = 10000... -> 10000//300 = 33
        step_size = self.v // self.num_groups
        start_indices = torch.arange(0, self.v, step_size)

        # Generate the group assignments using vectorization
        start_indices = start_indices.unsqueeze(1)  # Shape (num_groups, 1)

        # Add overlap to subsequent groupsd
        end_indices = start_indices + self.group_size

        # Ensure indices stay within the matrix size
        start_indices = torch.clamp(start_indices, 0, self.v - self.group_size)
        end_indices = torch.clamp(end_indices, self.group_size, self.v)

        # Generate slices for each group in a vectorized manner
        indices = torch.stack(
            [torch.arange(start, end) for start, end in zip(start_indices.flatten(), end_indices.flatten())])

        return indices

    def get_groupiness_loss(self, transition_matrix):
        device = transition_matrix.device
        group_vectors = transition_matrix[self.group_indices]
        group_vectors = group_vectors / (group_vectors.norm(dim=-1, keepdim=True) + 1e-8)

        # -------- Intra-group similarity --------
        # Compute cosine similarities inside each group
        intra_sims = torch.einsum('gic,gjc->gij', group_vectors, group_vectors)  # (num_groups, group_size, group_size)
        # Mask out self-similarity
        mask = ~torch.eye(self.group_size, device=device).bool()
        intra_group_mean = intra_sims[:, mask].mean()

        # -------- Inter-group similarity --------
        # Flatten groups: (num_groups * group_size, num_cols)
        flat_vectors = group_vectors.view(-1, group_vectors.size(-1))  # (num_groups * group_size, num_cols)
        flat_vectors = flat_vectors / (flat_vectors.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute pairwise similarities between all groups
        sims = flat_vectors @ flat_vectors.T  # (total_size, total_size)

        # Build group ids for each row
        group_ids = torch.arange(self.num_groups, device=device).repeat_interleave(self.group_size)  # (total_size,)

        # Create inter-group mask: True where different groups
        inter_mask = group_ids.unsqueeze(0) != group_ids.unsqueeze(1)  # (total_size, total_size)

        # Exclude self similarities too
        diag_mask = ~torch.eye(len(flat_vectors), device=device).bool()
        valid_inter_mask = inter_mask & diag_mask

        inter_group_mean = sims[valid_inter_mask].mean()

        # -------- Overlap penalty (neighboring groups only) --------
        # Neighboring groups: group i vs group i+1
        # if self.num_groups > 1:
        #     neighbor_sims = []
        #     for i in range(self.num_groups - 1):
        #         g1 = group_vectors[i]  # (group_size, dim)
        #         g2 = group_vectors[i + 1]  # (group_size, dim)
        #         sim = (g1 @ g2.T).mean()
        #         neighbor_sims.append(sim)
        #     overlap_penalty = torch.stack(neighbor_sims).mean()
        # else:
        #     overlap_penalty = torch.tensor(0.0, device=device)

        # -------- Final Loss --------
        # loss = (
        #         -self.intra_weight * intra_group_mean +  # maximize intra similarity
        #         self.inter_weight * inter_group_mean +  # minimize inter similarity
        #         self.overlap_weight * overlap_penalty  # minimize overlap
        # )

        return intra_group_mean, inter_group_mean
