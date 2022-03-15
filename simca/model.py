import torch

from simca.layers import ScaledEmbedding


class SiMCa(torch.nn.Module):
    def __init__(self, capacities, n_users, alpha,
                 n_features, user_embeddings=None,
                 train_user_embeddings=False):
        super(SiMCa, self).__init__()

        self.capacities = torch.FloatTensor(capacities)
        self.alpha = alpha

        self.item_embeddings = ScaledEmbedding(
            num_embeddings=capacities.shape[1],
            embedding_dim=n_features,
        )

        self.user_embeddings = ScaledEmbedding(
            num_embeddings=n_users,
            embedding_dim=n_features,
            _weight=user_embeddings
        )
        self.user_embeddings.weight.requires_grad = train_user_embeddings

    def forward(self, users_tensor, items_tensor, D_tensor):
        """Run the forward pass

        Args:
            users_tensor (torch.LongTensor): User ids
            items_tensor (torch.LongTensor): Item ids
            D_tensor (torch.FloatTensor): Travel distance matrix

        Returns:
            torch.FloatTensor: Probability allocation matrix
        """

        # Size variables
        batch_size = users_tensor.shape[0]
        n_candidates = items_tensor.shape[1]

        # Embedding lookup
        item_embeddings = self.item_embeddings(items_tensor)
        user_embeddings = self.user_embeddings(users_tensor)

        # Dot products
        dot_products = torch.bmm(
            item_embeddings,
            user_embeddings.view(batch_size, user_embeddings.shape[1], 1)
        )
        dot_products = dot_products.view(batch_size, n_candidates)

        # Affinity matrix
        if D_tensor is not None:
            affinity_matrix = (1-self.alpha) * dot_products \
                               - self.alpha * (D_tensor / torch.mean(D_tensor))
        else:
            affinity_matrix = dot_products

        return affinity_matrix
