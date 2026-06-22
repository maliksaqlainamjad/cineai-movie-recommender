"""
K-Means Clustering Module
Groups movies into clusters based on features: year, duration, rating.
Used to find which cluster a user's preferences map to, then recommend from that cluster.
"""

import numpy as np
import pandas as pd


class KMeansCustom:
    """K-Means implemented from scratch using NumPy."""

    def __init__(self, k=4, max_iter=100, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # Initialize centroids by picking k random points
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy().astype(float)

        for _ in range(self.max_iter):
            # Assign each point to nearest centroid
            distances = np.array([
                np.linalg.norm(X - c, axis=1) for c in self.centroids
            ])  # shape: (k, n)
            self.labels = np.argmin(distances, axis=0)

            # Recompute centroids
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) if np.any(self.labels == i) else self.centroids[i]
                for i in range(self.k)
            ])

            # Stop if centroids didn't move
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return self

    def predict(self, X):
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        return np.argmin(distances, axis=0)


def cluster_movies(movies_df, k=4):
    """
    Cluster movies by year, duration, rating.
    Returns movies_df with a 'cluster' column, the model, and scaler params.
    """
    features = movies_df[["year", "duration", "rating"]].values.astype(float)

    # Manual min-max normalization
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    X_scaled = (features - min_vals) / (max_vals - min_vals + 1e-9)

    model = KMeansCustom(k=k)
    model.fit(X_scaled)

    df = movies_df.copy()
    df["cluster"] = model.labels

    return df, model, min_vals, max_vals


def get_cluster_for_preferences(preferences, model, min_vals, max_vals, movies_df):
    """
    Map user preferences to a cluster.
    Finds which cluster center is closest to the user's feature vector.
    """
    # Build a representative feature vector from preferences
    year = (int(preferences.get("min_year", 2000)) + int(preferences.get("max_year", 2023))) / 2
    duration = float(preferences.get("max_duration", 150))
    rating = float(preferences.get("min_rating", 7.0))

    point = np.array([[year, duration, rating]], dtype=float)
    point_scaled = (point - min_vals) / (max_vals - min_vals + 1e-9)

    cluster_id = model.predict(point_scaled)[0]
    return int(cluster_id)


def get_cluster_summary(movies_df_clustered, k=4):
    """Return a summary description for each cluster."""
    summaries = {}
    for i in range(k):
        cluster = movies_df_clustered[movies_df_clustered["cluster"] == i]
        if len(cluster) == 0:
            continue
        avg_rating = cluster["rating"].mean()
        avg_year = int(cluster["year"].mean())
        genres = cluster["genre"].value_counts().index[0] if len(cluster) > 0 else "Mixed"
        summaries[i] = {
            "count": len(cluster),
            "avg_rating": round(avg_rating, 2),
            "avg_year": avg_year,
            "dominant_genre": genres,
        }
    return summaries
