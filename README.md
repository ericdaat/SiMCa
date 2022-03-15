# SiMCa: Sinkhorn Matrix Factorization with Capacity Constraints

[Eric Daoud](https://edaoud.com/)<sup>1,2</sup>,
[Luca Ganassali](https://lganassali.github.io/)<sup>1</sup>,
[Antoine Baker](https://prairie-institute.fr/chairs/antoine-baker/)<sup>1</sup> and
[Marc Lelarge](https://www.di.ens.fr/~lelarge/)<sup>1</sup>

1. INRIA, DI/ENS, Paris, France
2. Institut Curie, Paris, France

## Abstract

For a very broad range of problems, recommendation algorithms have been increasingly used over the past decade. In most of these algorithms, the predictions are built upon user-item affinity scores which are obtained from  high-dimensional embeddings of items and users. In more complex scenarios, with geometrical or capacity constraints, prediction based on embeddings may not be sufficient and some additional features should be considered in the design of the algorithm.

In this work, we study the recommendation problem in the setting where affinities between users and items are based both on their embeddings in a latent space and on their geographical distance in their underlying euclidean space (e.g., $\mathbb{R}^2$), together with item capacity constraints. This framework is motivated by some real-world applications, for instance in healthcare: the task is to recommend hospitals to patients based on their location, pathology, and hospital capacities. In these applications, there is somewhat of an asymmetry between users and items: items are viewed as static points, their embeddings, capacities and locations constraining the allocation. Upon the observation of an optimal allocation, user embeddings, items capacities, and their positions in their underlying euclidean space, our aim is to recover item embeddings in the latent space; doing so, we are then able to use this estimate e.g. in order to predict future allocations.

We propose an algorithm (SiMCa) based on matrix factorization enhanced with optimal transport steps to model user-item affinities and learn item embeddings from observed data. We then illustrate and discuss the results of such an approach for hospital recommendation on synthetic data.
