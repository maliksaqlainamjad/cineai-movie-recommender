"""
Search Module — BFS, DFS, A* Search
Treats movies as nodes in a graph connected by shared attributes (genre, director, year proximity).
Explores this graph to find the best matching movies from a seed movie.
"""

import heapq
from collections import deque


def build_movie_graph(movies_df):
    """
    Build adjacency list: two movies are connected if they share genre or director.
    Returns dict: movie_id -> list of neighbor movie_ids
    """
    graph = {row["id"]: [] for _, row in movies_df.iterrows()}
    rows = movies_df.to_dict("records")

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            # Connect if same genre or same director
            if a["genre"] == b["genre"] or a["director"] == b["director"]:
                graph[a["id"]].append(b["id"])
                graph[b["id"]].append(a["id"])

    return graph


def bfs_search(graph, start_id, movies_df, max_nodes=10):
    """BFS — explores movie graph level by level from a seed movie."""
    visited = set()
    queue = deque([start_id])
    result_ids = []
    steps = []

    while queue and len(result_ids) < max_nodes:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        result_ids.append(node)
        steps.append(f"Visited movie ID {node}")

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)

    result_df = movies_df[movies_df["id"].isin(result_ids)].copy()
    return result_df, steps


def dfs_search(graph, start_id, movies_df, max_nodes=10):
    """DFS — explores movie graph depth-first from a seed movie."""
    visited = set()
    stack = [start_id]
    result_ids = []
    steps = []

    while stack and len(result_ids) < max_nodes:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        result_ids.append(node)
        steps.append(f"Visited movie ID {node}")

        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)

    result_df = movies_df[movies_df["id"].isin(result_ids)].copy()
    return result_df, steps


def heuristic(movie_row, preferences):
    """
    Heuristic for A*: estimates how well a movie matches preferences.
    Lower score = better match (A* minimizes cost).
    """
    score = 0.0

    # Penalize rating distance from max (10)
    score += (10.0 - movie_row["rating"]) * 1.5

    # Penalize year distance from preferred decade center
    if preferences.get("min_year") and preferences.get("max_year"):
        mid_year = (int(preferences["min_year"]) + int(preferences["max_year"])) / 2
        score += abs(movie_row["year"] - mid_year) * 0.05

    # Prefer shorter movies slightly (configurable)
    score += movie_row["duration"] * 0.01

    return score


def astar_search(graph, start_id, movies_df, preferences, max_nodes=10):
    """
    A* Search — uses heuristic to explore the most promising movies first.
    Priority queue ordered by heuristic score.
    """
    movie_map = {row["id"]: row for _, row in movies_df.iterrows()}

    visited = set()
    # (heuristic_score, movie_id)
    heap = [(heuristic(movie_map[start_id], preferences), start_id)]
    result_ids = []
    steps = []

    while heap and len(result_ids) < max_nodes:
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        result_ids.append(node)
        steps.append(f"Expanded movie ID {node} | h={cost:.2f}")

        for neighbor in graph.get(node, []):
            if neighbor not in visited and neighbor in movie_map:
                h = heuristic(movie_map[neighbor], preferences)
                heapq.heappush(heap, (h, neighbor))

    result_df = movies_df[movies_df["id"].isin(result_ids)].copy()
    return result_df, steps
