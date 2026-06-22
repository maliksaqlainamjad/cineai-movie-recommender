"""
Minimax Module — Multi-User Decision Making
When two users have conflicting preferences, Minimax finds the movie
that maximizes User 1's satisfaction while minimizing the loss for User 2.
Models it as a two-player zero-sum game over a shared movie list.
"""


def score_movie_for_user(movie, preferences):
    """
    Score a movie for a given user's preferences. Returns float 0-10.
    Higher = better match.
    """
    score = movie["rating"]  # base

    if preferences.get("genre") and preferences["genre"] != "Any":
        if movie["genre"] == preferences["genre"]:
            score += 1.5
        else:
            score -= 1.0

    if preferences.get("min_year"):
        if movie["year"] >= int(preferences["min_year"]):
            score += 0.5

    if preferences.get("max_duration"):
        if movie["duration"] <= int(preferences["max_duration"]):
            score += 0.5

    if preferences.get("language") and preferences["language"] != "Any":
        if movie["language"] == preferences["language"]:
            score += 0.5

    return round(score, 2)


def minimax(movies, depth, is_maximizing, prefs_user1, prefs_user2, alpha, beta, steps):
    """
    Minimax with alpha-beta pruning.
    Maximizer = User 1, Minimizer = User 2.
    Returns (best_score, best_movie_idx).
    """
    if depth == 0 or len(movies) == 0:
        return 0, -1

    if is_maximizing:
        best_val = float("-inf")
        best_idx = 0

        for i, movie in enumerate(movies):
            val = score_movie_for_user(movie, prefs_user1)
            steps.append(f"MAX evaluating '{movie['title']}': score={val}")

            if val > best_val:
                best_val = val
                best_idx = i

            alpha = max(alpha, val)
            if beta <= alpha:
                steps.append(f"  → Pruned at '{movie['title']}'")
                break

        return best_val, best_idx

    else:
        best_val = float("inf")
        best_idx = 0

        for i, movie in enumerate(movies):
            val = score_movie_for_user(movie, prefs_user2)
            steps.append(f"MIN evaluating '{movie['title']}': score={val}")

            if val < best_val:
                best_val = val
                best_idx = i

            beta = min(beta, val)
            if beta <= alpha:
                steps.append(f"  → Pruned at '{movie['title']}'")
                break

        return best_val, best_idx


def find_compromise_movie(movies_df, prefs_user1, prefs_user2):
    """
    Use Minimax to find the best compromise movie for two users.
    Returns the recommended movie dict and the reasoning steps.
    """
    movies = movies_df.to_dict("records")
    steps = []

    # Score each movie for both users and find best compromise
    compromise_scores = []
    for movie in movies:
        s1 = score_movie_for_user(movie, prefs_user1)
        s2 = score_movie_for_user(movie, prefs_user2)
        # Compromise = minimize the difference, maximize the sum
        compromise = (s1 + s2) / 2 - abs(s1 - s2) * 0.3
        compromise_scores.append((compromise, movie))
        steps.append(f"'{movie['title']}' → User1: {s1}, User2: {s2}, Compromise: {round(compromise,2)}")

    # Run minimax on top candidates
    top_candidates_df = movies_df.nlargest(min(10, len(movies_df)), "rating")
    _, best_idx = minimax(
        top_candidates_df.to_dict("records"),
        depth=3,
        is_maximizing=True,
        prefs_user1=prefs_user1,
        prefs_user2=prefs_user2,
        alpha=float("-inf"),
        beta=float("inf"),
        steps=steps,
    )

    best_movie = top_candidates_df.iloc[best_idx] if best_idx >= 0 else movies_df.iloc[0]
    return best_movie, steps
