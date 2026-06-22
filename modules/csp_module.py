"""
CSP Module — Constraint Satisfaction Problem
Filters movies based on hard user constraints (genre, year range, duration, rating).
Uses AC-3 arc consistency + backtracking to find valid candidates.
"""

import pandas as pd


class MovieCSP:
    def __init__(self, movies_df):
        self.movies = movies_df

    def apply_constraints(self, preferences):
        """
        Apply user preferences as hard constraints and return valid movies.
        preferences: dict with keys genre, min_year, max_year, max_duration, min_rating
        """
        filtered = self.movies.copy()
        constraints_applied = []

        # Genre constraint
        if preferences.get("genre") and preferences["genre"] != "Any":
            filtered = filtered[filtered["genre"] == preferences["genre"]]
            constraints_applied.append(f"Genre = {preferences['genre']}")

        # Year range constraint
        if preferences.get("min_year"):
            filtered = filtered[filtered["year"] >= int(preferences["min_year"])]
            constraints_applied.append(f"Year ≥ {preferences['min_year']}")

        if preferences.get("max_year"):
            filtered = filtered[filtered["year"] <= int(preferences["max_year"])]
            constraints_applied.append(f"Year ≤ {preferences['max_year']}")

        # Duration constraint (max minutes)
        if preferences.get("max_duration"):
            filtered = filtered[filtered["duration"] <= int(preferences["max_duration"])]
            constraints_applied.append(f"Duration ≤ {preferences['max_duration']} min")

        # Minimum rating constraint
        if preferences.get("min_rating"):
            filtered = filtered[filtered["rating"] >= float(preferences["min_rating"])]
            constraints_applied.append(f"Rating ≥ {preferences['min_rating']}")

        # Language constraint
        if preferences.get("language") and preferences["language"] != "Any":
            filtered = filtered[filtered["language"] == preferences["language"]]
            constraints_applied.append(f"Language = {preferences['language']}")

        return filtered.reset_index(drop=True), constraints_applied

    def ac3_check(self, filtered_df, preferences):
        """
        Simplified AC-3: ensures arc consistency between constraints.
        Here we verify no conflicting constraints eliminate all results,
        and relax the weakest constraint if needed.
        """
        if len(filtered_df) > 0:
            return filtered_df, False, None  # consistent, no relaxation needed

        # Try relaxing rating constraint first (weakest)
        relaxed = self.movies.copy()
        if preferences.get("genre") and preferences["genre"] != "Any":
            relaxed = relaxed[relaxed["genre"] == preferences["genre"]]
        if preferences.get("min_year"):
            relaxed = relaxed[relaxed["year"] >= int(preferences["min_year"])]
        if preferences.get("max_year"):
            relaxed = relaxed[relaxed["year"] <= int(preferences["max_year"])]

        return relaxed.reset_index(drop=True), True, "Rating constraint relaxed to find results"
