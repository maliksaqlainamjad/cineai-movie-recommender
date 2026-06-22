"""
Hybrid AI Movie Recommendation System
FAST-NUCES | AI Lab Semester Project | AI 2002
Integrates: CSP, BFS/DFS/A*, K-Means, ANN, Minimax
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np

from modules.csp_module import MovieCSP
from modules.search_module import build_movie_graph, bfs_search, dfs_search, astar_search
from modules.kmeans_module import cluster_movies, get_cluster_for_preferences, get_cluster_summary
from modules.ann_module import predict_scores_ann
from modules.heuristic_module import rank_movies
from modules.minimax_module import find_compromise_movie

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI — Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d0f14; }

    .hero {
        background: linear-gradient(135deg, #1a1d26 0%, #0d0f14 100%);
        border: 1px solid #2a2d3e;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 2.8rem; font-weight: 700; color: #e8eaf6; margin: 0; }
    .hero h1 span { color: #7c6af7; }
    .hero p  { color: #9e9eb8; font-size: 1.05rem; margin-top: 0.5rem; }

    .movie-card {
        background: #1a1d26;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .movie-card:hover { border-color: #7c6af7; }
    .movie-title { font-size: 1.15rem; font-weight: 600; color: #e8eaf6; }
    .movie-meta  { color: #9e9eb8; font-size: 0.85rem; margin-top: 0.3rem; }
    .score-badge {
        display: inline-block;
        background: #7c6af7;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .rating-star { color: #f5c518; font-weight: 600; }

    .explain-item {
        background: #12141c;
        border-left: 3px solid #7c6af7;
        padding: 0.4rem 0.8rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #c5c7d8;
    }

    .step-log {
        background: #12141c;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.8rem;
        color: #7fba84;
        max-height: 200px;
        overflow-y: auto;
    }

    .tag {
        display: inline-block;
        background: #2a2d3e;
        color: #9e9eb8;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.78rem;
        margin-right: 0.3rem;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7c6af7;
        border-bottom: 1px solid #2a2d3e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    div[data-testid="stExpander"] { border: 1px solid #2a2d3e !important; border-radius: 10px; }
    .stButton > button {
        background: #7c6af7 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
        padding: 0.6rem 1rem !important;
    }
    .stButton > button:hover { background: #6857e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "data", "movies.csv")
    return pd.read_csv(path)

movies_df = load_data()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 Cine<span>AI</span></h1>
  <p>Hybrid AI Movie Recommendation System — CSP · BFS/DFS/A* · K-Means · ANN · Minimax</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — User Preferences ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Your Preferences")

    genres = ["Any"] + sorted(movies_df["genre"].unique().tolist())
    languages = ["Any"] + sorted(movies_df["language"].unique().tolist())

    genre = st.selectbox("Favorite Genre", genres)
    language = st.selectbox("Language", languages)

    col1, col2 = st.columns(2)
    with col1:
        min_year = st.number_input("Year From", min_value=1970, max_value=2024, value=1990)
    with col2:
        max_year = st.number_input("Year To", min_value=1970, max_value=2024, value=2024)

    max_duration = st.slider("Max Duration (min)", 60, 240, 180)
    min_rating = st.slider("Min IMDb Rating", 5.0, 10.0, 7.0, 0.1)

    st.markdown("---")
    st.markdown("### 🔍 Search Algorithm")
    algo = st.radio("Algorithm for graph search", ["A* (Optimal)", "BFS", "DFS"])

    st.markdown("---")
    st.markdown("### 👥 Multi-User Mode")
    multi_user = st.checkbox("Enable Minimax (2 users)")

    prefs_user2 = {}
    if multi_user:
        st.markdown("**User 2 Preferences**")
        genre2 = st.selectbox("User 2 Genre", genres, key="g2")
        lang2 = st.selectbox("User 2 Language", languages, key="l2")
        rating2 = st.slider("User 2 Min Rating", 5.0, 10.0, 7.5, 0.1, key="r2")
        prefs_user2 = {"genre": genre2, "language": lang2, "min_rating": rating2}

    st.markdown("---")
    top_n = st.slider("Top N recommendations", 3, 10, 5)
    run_btn = st.button("🚀 Get Recommendations")

# ── Main Logic ────────────────────────────────────────────────────────────────
preferences = {
    "genre": genre,
    "language": language,
    "min_year": min_year,
    "max_year": max_year,
    "max_duration": max_duration,
    "min_rating": min_rating,
}

if run_btn:
    with st.spinner("Running AI pipeline..."):

        # ── Step 1: CSP ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🧩 Step 1 — Constraint Satisfaction (CSP)</div>', unsafe_allow_html=True)
        csp = MovieCSP(movies_df)
        filtered_df, constraints = csp.apply_constraints(preferences)
        filtered_df, relaxed, relax_msg = csp.ac3_check(filtered_df, preferences)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.success(f"✅ {len(filtered_df)} movies passed constraints")
            for c in constraints:
                st.markdown(f'<span class="tag">{c}</span>', unsafe_allow_html=True)
            if relaxed:
                st.warning(f"⚠️ AC-3 relaxation: {relax_msg}")
        with col_b:
            st.metric("Movies Before Filter", len(movies_df))
            st.metric("Movies After Filter", len(filtered_df))

        if len(filtered_df) == 0:
            st.error("No movies found. Try relaxing your constraints.")
            st.stop()

        # ── Step 2: K-Means Clustering ────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Step 2 — K-Means Clustering</div>', unsafe_allow_html=True)
        clustered_all, km_model, min_vals, max_vals = cluster_movies(movies_df, k=4)
        user_cluster = get_cluster_for_preferences(preferences, km_model, min_vals, max_vals, movies_df)
        summaries = get_cluster_summary(clustered_all)

        # Attach cluster to filtered_df
        cluster_map = dict(zip(clustered_all["id"], clustered_all["cluster"]))
        filtered_df["cluster"] = filtered_df["id"].map(cluster_map)

        cols = st.columns(4)
        for i, (cid, info) in enumerate(summaries.items()):
            with cols[i % 4]:
                border = "border: 2px solid #7c6af7;" if cid == user_cluster else ""
                st.markdown(f"""
                <div style="background:#1a1d26;{border}border-radius:10px;padding:0.8rem;text-align:center;">
                  <div style="font-size:0.75rem;color:#9e9eb8;">Cluster {cid}</div>
                  <div style="font-size:1.2rem;font-weight:700;color:#e8eaf6;">{info['dominant_genre']}</div>
                  <div style="font-size:0.75rem;color:#9e9eb8;">⭐ {info['avg_rating']} · {info['count']} movies</div>
                  {'<div style="font-size:0.7rem;color:#7c6af7;margin-top:4px;">← Your cluster</div>' if cid == user_cluster else ''}
                </div>
                """, unsafe_allow_html=True)

        # ── Step 3: Graph Search ──────────────────────────────────────────────
        st.markdown(f'<div class="section-header">🔍 Step 3 — Graph Search ({algo})</div>', unsafe_allow_html=True)
        graph = build_movie_graph(filtered_df)
        seed_id = filtered_df.iloc[0]["id"] if len(filtered_df) > 0 else movies_df.iloc[0]["id"]

        if algo == "BFS":
            searched_df, search_steps = bfs_search(graph, seed_id, filtered_df)
        elif algo == "DFS":
            searched_df, search_steps = dfs_search(graph, seed_id, filtered_df)
        else:
            searched_df, search_steps = astar_search(graph, seed_id, filtered_df, preferences)

        with st.expander(f"📋 {algo} Traversal Log ({len(search_steps)} steps)"):
            st.markdown('<div class="step-log">' + "<br>".join(search_steps) + '</div>', unsafe_allow_html=True)

        if len(searched_df) == 0:
            searched_df = filtered_df.copy()

        # ── Step 4: ANN Scoring ───────────────────────────────────────────────
        st.markdown('<div class="section-header">🧠 Step 4 — ANN Rating Prediction</div>', unsafe_allow_html=True)
        ann_df, ann_model = predict_scores_ann(searched_df, preferences)
        st.info(f"ANN (5→8→4→1) trained on {len(ann_df)} candidate movies. Predicted match scores loaded.")

        ann_preview = ann_df[["title", "rating", "ann_score"]].head(5).copy()
        ann_preview.columns = ["Movie", "IMDb Rating", "ANN Match Score (0-10)"]
        st.dataframe(ann_preview, use_container_width=True, hide_index=True)

        # ── Step 5: Heuristic Ranking ─────────────────────────────────────────
        st.markdown('<div class="section-header">🏆 Step 5 — Heuristic Ranking & Final Recommendations</div>', unsafe_allow_html=True)
        ranked_df = rank_movies(ann_df, preferences, cluster_id=user_cluster)
        top_movies = ranked_df.head(top_n)

        for rank, (_, row) in enumerate(top_movies.iterrows(), 1):
            with st.container():
                st.markdown(f"""
                <div class="movie-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                      <div class="movie-title">#{rank} {row['title']}</div>
                      <div class="movie-meta">
                        {row['genre']} · {row['year']} · {row['duration']} min · {row['language']}
                      </div>
                      <div style="margin-top:0.5rem;">
                        <span class="rating-star">★ {row['rating']}</span>
                        &nbsp;&nbsp;
                        <span class="score-badge">AI Score: {row['heuristic_score']}</span>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🔎 Why was this recommended?"):
                    for reason in row["explanation"]:
                        st.markdown(f'<div class="explain-item">→ {reason}</div>', unsafe_allow_html=True)

        # ── Step 6: Minimax (Multi-User) ──────────────────────────────────────
        if multi_user and prefs_user2:
            st.markdown('<div class="section-header">🎭 Step 6 — Minimax: Multi-User Compromise</div>', unsafe_allow_html=True)
            compromise_movie, mm_steps = find_compromise_movie(ranked_df.head(15), preferences, prefs_user2)

            st.success(f"🤝 Compromise Pick: **{compromise_movie['title']}** ({compromise_movie['genre']}, {compromise_movie['year']}) — ⭐ {compromise_movie['rating']}")

            with st.expander("📋 Minimax Evaluation Log"):
                st.markdown('<div class="step-log">' + "<br>".join(mm_steps[:30]) + '</div>', unsafe_allow_html=True)

        # ── Summary ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;color:#9e9eb8;font-size:0.85rem;padding:1rem 0;">
          FAST-NUCES Faisalabad · AI 2002 Lab Project · Saqlain Amjed (23F-0537)
        </div>
        """, unsafe_allow_html=True)

else:
    # Default landing state
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#9e9eb8;">
      <div style="font-size:3rem;">🎬</div>
      <div style="font-size:1.2rem;font-weight:600;color:#e8eaf6;margin-top:1rem;">Set your preferences and click <span style="color:#7c6af7;">Get Recommendations</span></div>
      <div style="margin-top:0.5rem;font-size:0.9rem;">
        The system will run CSP → Graph Search → K-Means → ANN → Heuristic Ranking
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🛠️ AI Techniques Used</div>', unsafe_allow_html=True)
    techniques = [
        ("🧩", "CSP (Constraint Satisfaction)", "Filters movies using hard constraints: genre, year, duration, rating, language. AC-3 arc consistency applied."),
        ("🔍", "BFS / DFS / A* Search", "Explores a movie similarity graph. A* uses a heuristic to find optimal candidates first."),
        ("📊", "K-Means Clustering", "Groups 50 movies into 4 clusters by year/duration/rating. Maps your preferences to the closest cluster."),
        ("🧠", "ANN (Neural Network)", "5→8→4→1 network trained from scratch with NumPy. Predicts match score for each candidate movie."),
        ("🏆", "Heuristic Ranking", "Combines rating, genre match, year proximity, duration, language, cluster, and ANN score into a final ranked list."),
        ("🎭", "Minimax + Alpha-Beta", "Optional multi-user mode: finds the best compromise movie when two users have conflicting preferences."),
    ]
    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate(techniques):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="movie-card">
              <div style="font-size:1.8rem;">{icon}</div>
              <div style="font-weight:600;color:#e8eaf6;margin-top:0.5rem;">{name}</div>
              <div style="font-size:0.83rem;color:#9e9eb8;margin-top:0.4rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
