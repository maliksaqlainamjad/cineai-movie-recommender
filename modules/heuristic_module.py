"""
Heuristic Module — Final Ranking
Combines CSP filtering, search results, cluster membership, and ANN scores
into a single ranked list with explainability.
"""


def compute_heuristic_score(movie_row, preferences, cluster_id=None, movie_cluster=None):
    """
    Compute a composite heuristic score for a movie given user preferences.
    Returns float score and a list of explanation strings.
    """
    score = 0.0
    explanation = []

    # Base: IMDb rating (weight: 40%)
    rating_score = movie_row["rating"] * 4.0
    score += rating_score
    explanation.append(f"IMDb rating {movie_row['rating']}/10 → +{round(rating_score, 1)} pts")

    # Genre match (weight: 20%)
    if preferences.get("genre") and preferences["genre"] != "Any":
        if movie_row["genre"] == preferences["genre"]:
            score += 20.0
            explanation.append(f"Genre match ({movie_row['genre']}) → +20 pts")
        else:
            explanation.append(f"Genre mismatch ({movie_row['genre']} ≠ {preferences['genre']}) → +0 pts")

    # Year range preference (weight: 10%)
    if preferences.get("min_year") and preferences.get("max_year"):
        mid = (int(preferences["min_year"]) + int(preferences["max_year"])) / 2
        proximity = max(0, 10 - abs(movie_row["year"] - mid) / 5)
        score += proximity
        explanation.append(f"Year {movie_row['year']} (target ~{int(mid)}) → +{round(proximity, 1)} pts")

    # Duration preference (weight: 10%)
    if preferences.get("max_duration"):
        if movie_row["duration"] <= int(preferences["max_duration"]):
            score += 10.0
            explanation.append(f"Duration {movie_row['duration']} min fits your limit → +10 pts")

    # Language match (weight: 10%)
    if preferences.get("language") and preferences["language"] != "Any":
        if movie_row["language"] == preferences["language"]:
            score += 10.0
            explanation.append(f"Language match ({movie_row['language']}) → +10 pts")

    # Cluster bonus (weight: 10%)
    if cluster_id is not None and movie_cluster is not None:
        if movie_cluster == cluster_id:
            score += 10.0
            explanation.append(f"In your preferred movie cluster → +10 pts")

    return round(score, 2), explanation


def rank_movies(movies_df, preferences, cluster_id=None):
    """
    Rank all movies by heuristic score and return sorted DataFrame with explanations.
    """
    scores = []
    explanations = []

    cluster_col = movies_df["cluster"].values if "cluster" in movies_df.columns else [None] * len(movies_df)

    for i, (_, row) in enumerate(movies_df.iterrows()):
        mc = cluster_col[i] if cluster_col[i] is not None else None
        s, exp = compute_heuristic_score(row, preferences, cluster_id, mc)

        # Boost by ANN score if available
        if "ann_score" in movies_df.columns:
            ann_boost = row["ann_score"] * 0.5
            s += ann_boost
            exp.append(f"ANN predicted score {row['ann_score']}/10 → +{round(ann_boost, 1)} pts")

        scores.append(s)
        explanations.append(exp)

    movies_df = movies_df.copy()
    movies_df["heuristic_score"] = scores
    movies_df["explanation"] = explanations
    return movies_df.sort_values("heuristic_score", ascending=False).reset_index(drop=True)
