import torch
from sklearn.datasets import make_blobs, make_moons, make_circles
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, n_features, n_clusters, n_samples):
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_samples = n_samples

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_true[idx]

    def plot(self):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(5, 5)
        )

        sns.scatterplot(  # plot first 2 components
            x=self.X[:, 0],
            y=self.X[:, 1],
            hue=map(str, self.y_true),
            ax=ax,
            legend=False
        )

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Clusters visualization")

        return fig


class BlobsDataset(ToyDataset):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
    """
    def __init__(self, n_features, n_clusters, n_samples):
        super().__init__(n_features, n_clusters, n_samples)

        X, y_true = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=.8,
            random_state=0
        )

        self.X = torch.FloatTensor(X)
        self.y_true = torch.LongTensor(y_true)


class MoonsDataset(ToyDataset):
    def __init__(self, n_samples):
        super().__init__(n_samples, n_features=2, n_clusters=2)

        X, y_true = make_moons(
            n_samples=n_samples,
            random_state=0,
            noise=.05
        )

        self.X = torch.FloatTensor(X)
        self.y_true = torch.LongTensor(y_true)


class CirclesDataset(ToyDataset):
    def __init__(self, n_samples):
        super().__init__(n_samples, n_features=2, n_clusters=2)

        X, y_true = make_circles(
            n_samples=n_samples,
            random_state=0,
            noise=.05
        )

        self.X = torch.FloatTensor(X)
        self.y_true = torch.LongTensor(y_true)
