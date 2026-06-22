"""
ANN Module — Artificial Neural Network (from scratch, NumPy only)
Predicts a match score for each movie given user preferences.
Architecture: Input(5) -> Hidden(8) -> Hidden(4) -> Output(1)
"""

import numpy as np


class ANN:
    """
    Feedforward Neural Network with 2 hidden layers.
    Trained using backpropagation + gradient descent.
    """

    def __init__(self, layer_sizes=[5, 8, 4, 1], lr=0.01, epochs=500):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        np.random.seed(0)
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.5
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_deriv(self, a):
        return a * (1 - a)

    def _forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self._sigmoid(z)
            activations.append(a)
        return activations

    def _backward(self, activations, y):
        m = y.shape[0]
        deltas = [None] * len(self.weights)

        # Output layer error
        error = activations[-1] - y
        deltas[-1] = error * self._sigmoid_deriv(activations[-1])

        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i + 1] @ self.weights[i + 1].T
            deltas[i] = error * self._sigmoid_deriv(activations[i + 1])

        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * (activations[i].T @ deltas[i]) / m
            self.biases[i] -= self.lr * deltas[i].mean(axis=0, keepdims=True)

    def train(self, X, y):
        for _ in range(self.epochs):
            activations = self._forward(X)
            self._backward(activations, y)

    def predict(self, X):
        activations = self._forward(X)
        return activations[-1]


def prepare_features(movies_df, preferences):
    """
    Build feature matrix for ANN from movie data + user preferences.
    Features per movie: [year_norm, duration_norm, rating_norm, genre_match, language_match]
    """
    df = movies_df.copy()

    # Normalize continuous features
    df["year_norm"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min() + 1e-9)
    df["dur_norm"] = (df["duration"] - df["duration"].min()) / (df["duration"].max() - df["duration"].min() + 1e-9)
    df["rat_norm"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min() + 1e-9)

    # Genre match feature
    pref_genre = preferences.get("genre", "Any")
    df["genre_match"] = (df["genre"] == pref_genre).astype(float)

    # Language match feature
    pref_lang = preferences.get("language", "Any")
    df["lang_match"] = (df["language"] == pref_lang).astype(float) if pref_lang != "Any" else 0.5

    features = df[["year_norm", "dur_norm", "rat_norm", "genre_match", "lang_match"]].values.astype(float)
    return features


def generate_synthetic_labels(movies_df, preferences):
    """
    Generate synthetic training labels based on how well each movie matches preferences.
    Label = normalized composite score (0 to 1).
    """
    scores = []
    pref_genre = preferences.get("genre", "Any")
    pref_lang = preferences.get("language", "Any")
    min_rating = float(preferences.get("min_rating", 7.0))

    for _, row in movies_df.iterrows():
        s = row["rating"] / 10.0  # base score from rating

        if pref_genre != "Any" and row["genre"] == pref_genre:
            s += 0.2
        if pref_lang != "Any" and row["language"] == pref_lang:
            s += 0.1
        if row["rating"] >= min_rating:
            s += 0.1

        scores.append(min(s, 1.0))

    return np.array(scores).reshape(-1, 1)


def predict_scores_ann(movies_df, preferences):
    """
    Train ANN on synthetic labels and predict match scores for all movies.
    Returns movies_df with a new 'ann_score' column.
    """
    X = prepare_features(movies_df, preferences)
    y = generate_synthetic_labels(movies_df, preferences)

    ann = ANN(layer_sizes=[5, 8, 4, 1], lr=0.05, epochs=800)
    ann.train(X, y)

    preds = ann.predict(X).flatten()
    df = movies_df.copy()
    df["ann_score"] = np.round(preds * 10, 2)  # scale to 0-10

    return df, ann
