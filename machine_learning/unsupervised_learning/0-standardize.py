#!/usr/bin/env python3
"""Standardize tabular data using Scikit-learn."""

from sklearn import preprocessing


def Standardize(X):
    """Return the standardized version of X."""
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
