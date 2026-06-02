#!/usr/bin/env python3
"""Module for applying PCA to tabular data."""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Apply PCA to X and return transformed data and fitted PCA object."""
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)
    return X_pca, pca
