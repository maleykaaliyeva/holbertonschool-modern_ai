#!/usr/bin/env python3
"""Module for K-Means clustering."""

from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """Create and fit a K-Means clustering model."""
    model = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    return model.fit(X)
