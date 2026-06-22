# 🎬 CineAI — Hybrid AI Movie Recommendation System

> **FAST-NUCES Faisalabad | AI 2002 Lab Semester Project**  
> Built by **Saqlain Amjed** (23F-0537) · BSCS 6th Semester

A fully functional Hybrid AI system that recommends movies using six distinct AI techniques — all implemented from scratch. The system takes user preferences as input and runs them through a complete AI pipeline: constraint filtering, graph search, clustering, neural network scoring, heuristic ranking, and optional multi-user decision-making.

---

## 🚀 Live Demo

```bash
git clone https://github.com/your-username/cineai-movie-recommender
cd cineai-movie-recommender
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 AI Techniques Implemented

| Technique | Purpose | Implementation |
|-----------|---------|----------------|
| **CSP** (Constraint Satisfaction) | Filter movies by hard constraints (genre, year, duration, rating, language) with AC-3 arc consistency | `modules/csp_module.py` |
| **BFS / DFS** | Explore movie similarity graph breadth-first and depth-first | `modules/search_module.py` |
| **A\* Search** | Optimal graph search using a heuristic function to find the best candidates first | `modules/search_module.py` |
| **K-Means Clustering** | Group movies into clusters by year, duration, and rating; map user preferences to closest cluster | `modules/kmeans_module.py` |
| **ANN** (Neural Network) | 5→8→4→1 feedforward network trained from scratch (NumPy only) to predict match scores | `modules/ann_module.py` |
| **Heuristic Ranking** | Composite scoring function combining all module outputs into a final ranked list with explanations | `modules/heuristic_module.py` |
| **Minimax + Alpha-Beta** | Multi-user compromise: finds the best movie when two users have conflicting preferences | `modules/minimax_module.py` |

---

## 🏗️ System Architecture

```
User Preferences (genre, year, duration, rating, language)
         │
         ▼
┌─────────────────┐
│   CSP Module    │  ← Filters invalid movies using hard constraints + AC-3
└────────┬────────┘
         │ Candidate movies
         ▼
┌─────────────────┐
│  K-Means Module │  ← Clusters all movies; identifies user's preferred cluster
└────────┬────────┘
         │ Cluster ID
         ▼
┌─────────────────┐
│  Search Module  │  ← BFS / DFS / A* over movie similarity graph
└────────┬────────┘
         │ Explored movies
         ▼
┌─────────────────┐
│   ANN Module    │  ← Predicts match score (0–10) for each candidate
└────────┬────────┘
         │ Scored movies
         ▼
┌──────────────────────┐
│  Heuristic Ranking   │  ← Combines all signals → ranked list with explanations
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
Top-N Results  Minimax (optional)
               ← Multi-user compromise
```

---

## 📁 Project Structure

```
cineai-movie-recommender/
│
├── app.py                    # Streamlit UI — main entry point
├── requirements.txt
│
├── data/
│   └── movies.csv            # 50-movie dataset (IMDb-based)
│
└── modules/
    ├── csp_module.py         # Constraint Satisfaction + AC-3
    ├── search_module.py      # BFS, DFS, A* on movie graph
    ├── kmeans_module.py      # K-Means from scratch (NumPy)
    ├── ann_module.py         # ANN from scratch (NumPy, backprop)
    ├── heuristic_module.py   # Composite scoring + explainability
    └── minimax_module.py     # Minimax + Alpha-Beta pruning
```

---

## 🔍 How It Works — Step by Step

### 1. CSP — Constraint Satisfaction
The user's preferences (genre, year range, max duration, min IMDb rating, language) are modeled as **hard constraints**. The CSP module filters the movie database to only keep movies that satisfy all constraints. AC-3 (Arc Consistency Algorithm 3) checks for conflicting constraints and relaxes the weakest one (rating) if no movies pass.

### 2. K-Means Clustering
All 50 movies are clustered into **4 groups** based on year, duration, and rating using a from-scratch K-Means implementation. The user's preferences are also mapped to the nearest cluster center. Movies in the user's cluster receive a bonus score in the final ranking.

### 3. Graph Search (BFS / DFS / A\*)
Movies are treated as **nodes in a graph**. Two movies are connected if they share a genre or director. Starting from the highest-rated filtered movie as a seed, the search algorithm explores this graph to find the most relevant candidates.

- **BFS** explores level by level (broadest exploration)
- **DFS** goes deep on one path first
- **A\*** uses a heuristic (rating quality + year proximity + duration) to expand the most promising movies first

### 4. ANN — Artificial Neural Network
A **5→8→4→1 feedforward neural network** is trained from scratch using NumPy (no TensorFlow or PyTorch). Features per movie: `[year_norm, duration_norm, rating_norm, genre_match, language_match]`. The network is trained using backpropagation + gradient descent on synthetic labels derived from the user's preferences. It outputs a match score from 0–10.

### 5. Heuristic Ranking
A composite heuristic function combines:
- IMDb rating (40% weight)
- Genre match (+20 pts)
- Year proximity to preference range (up to +10 pts)
- Duration fit (+10 pts)
- Language match (+10 pts)
- Cluster membership bonus (+10 pts)
- ANN predicted score (0.5× weight)

Each recommendation is shown with a **full explanation** of why it was chosen.

### 6. Minimax — Multi-User Mode (Optional)
When two users have different preferences, **Minimax with Alpha-Beta pruning** models the decision as a two-player game. User 1 is the maximizer (wants the best match for themselves), User 2 is the minimizer. The algorithm finds the movie that produces the best compromise outcome, pruning branches that cannot improve the result.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **Pandas** — data handling
- **NumPy** — all ML algorithms implemented from scratch
- **No external ML libraries used for core algorithms** (ANN and K-Means are pure NumPy)

---

## 📊 Dataset

A curated 50-movie CSV dataset covering diverse genres, years (1972–2021), languages, and IMDb ratings (7.3–9.3). Sourced from IMDb data.

Fields: `id, title, genre, year, duration, rating, director, language`

---

## 💡 Key Learning Outcomes

- Modeled real-world filtering as a CSP and applied AC-3 consistency
- Implemented graph-based search (BFS, DFS, A*) on a domain-specific problem
- Built K-Means clustering from scratch and used it for preference mapping
- Implemented backpropagation and gradient descent without any ML framework
- Combined multiple AI techniques into a single explainable pipeline
- Applied adversarial search (Minimax + Alpha-Beta) to a multi-agent decision problem

---

## 👤 Author

**Saqlain Amjed**  
BS Computer Science — FAST-NUCES Faisalabad (Batch 2023)  
Roll No: 23F-0537  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)

---

## 📄 License

MIT License — free to use, modify, and distribute.
