import torch

from src.ml.layers import ScaledEmbedding


class SiMaC(torch.nn.Module):
    def __init__(self, capacities, n_users, epsilon, alpha,
                 n_features, n_iter=10, user_embeddings=None,
                 train_user_embeddings=False):
        super(SiMaC, self).__init__()
        self.capacities = torch.FloatTensor(capacities)
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_iter = n_iter

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
        n_users = self.user_embeddings.weight.shape[0]

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
        affinity_matrix = (1-self.alpha) * dot_products \
                           -self.alpha * (D_tensor / torch.mean(D_tensor))

        self.affinity_matrix = affinity_matrix

        # Sinkhorn algorithm: declare variables
        K = torch.exp(affinity_matrix / self.epsilon)  # divide affinity matrix by epsilon
                                                       # epsilon: temperature for entropic regularization
                                                       # K is Gibbs kernel: exp(-cost/epsilon)

        # Scaling factor
        scaling_factor = batch_size / n_users

        a = torch.ones(batch_size)                      # users
        b = self.capacities.squeeze() * scaling_factor  # capacities
        v = torch.ones(n_candidates)

        # Sinkhorn algorithm: run iterations
        for _ in range(self.n_iter):
            # find vectors u & vec
            # solve for conservation of mass
            u = a / torch.matmul(K, v)
            v = b / torch.matmul(torch.transpose(K, 1, 0), u)

        uv = torch.outer(u, v)
        P = K * uv

        return P


class SiMaCImplicit(torch.nn.Module):
    def __init__(self, capacities, n_users, epsilon, alpha,
                 n_features, user_embeddings=None,
                 train_user_embeddings=False):
        super(SiMaCImplicit, self).__init__()
        self.capacities = torch.FloatTensor(capacities)
        self.epsilon = epsilon
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
        n_users = self.user_embeddings.weight.shape[0]

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
        affinity_matrix = (1-self.alpha) * dot_products \
                           -self.alpha * (D_tensor / torch.mean(D_tensor))

        return affinity_matrix
