import scipy.sparse
import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
import re

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies  = pd.read_pickle('src/movies.pkl')
    tfidf_matrix = scipy.sparse.load_npz('src/tfidf_matrix.npz')
    ratings = pd.read_csv('data/ratings.csv')
    return movies, tfidf_matrix, ratings

movies, tfidf_matrix, ratings = load_data()

# ── Build User-Movie Rating Matrix (cached) ───────────────────────────────────
@st.cache_data
def build_rating_matrix(_ratings, _movies):
    """
    Builds a user-movie matrix where:
      rows    = real users
      columns = movies (indexed by movieId)
      values  = ratings (0 where unrated)

    Also returns a stats table (avg rating + count per movie)
    for the content-based fallback.
    """
    matrix = _ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

    stats = _ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    return matrix, stats

rating_matrix, movie_stats = build_rating_matrix(ratings, movies)

# ── Clean Display Titles ───────────────────────────────────────────────────────
def clean_title(title):
    match = re.match(r'^(.*),\s*(The|A|An)\s*(\(\d{4}\))$', title, re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)} {match.group(3)}"
    return title

movies['title_display'] = movies['title'].apply(clean_title)

# ── Fuzzy Matching ────────────────────────────────────────────────────────────
def find_best_match(user_input):
    if not user_input.strip():
        return None
    query        = user_input.strip().lower()
    titles       = movies['title'].tolist()
    titles_lower = [t.lower() for t in titles]

    # Stage 1: all query words appear in title
    substring_matches = [
        i for i, t in enumerate(titles_lower)
        if all(word in t for word in query.split())
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        return min(substring_matches, key=lambda i: len(titles[i]))

    # Stage 2: fuzzy fallback
    match = process.extractOne(query, titles_lower, scorer=fuzz.token_set_ratio)
    if match and match[1] >= 85:
        return match[2]
    return None

# ── User-Based Collaborative Filtering ───────────────────────────────────────
def get_user_cf_recommendations(user_ratings_dict, n=10, k=20):
    """
    user_ratings_dict: {movieId: star_rating} for the new user's inputs
    n: number of recommendations to return
    k: number of similar real users to consult

    Steps:
    1. Build a sparse rating vector for the new user
    2. Find K real users whose ratings on the SAME movies are most similar
    3. Collect movies those neighbors rated highly (>=4 stars)
    4. Score each candidate movie by how many neighbors liked it,
       weighted by how similar each neighbor is to the new user
    5. Fall back to content-based for movies with no neighbor signal
    """
    input_movie_ids = set(user_ratings_dict.keys())

    # ── Step 1: Align new user vector with rating matrix columns ──────────────
    # Only keep columns (movieIds) that the new user has rated
    common_movies = [mid for mid in user_ratings_dict if mid in rating_matrix.columns]

    if not common_movies:
        # None of the input movies have been rated by anyone in the dataset
        # Fall back to content-based entirely
        return None

    # Subset the rating matrix to only the movies our new user rated
    sub_matrix = rating_matrix[common_movies]

    # Build new user's vector for those same movies
    new_user_vector = np.array([[user_ratings_dict[mid] for mid in common_movies]])

    # Only consider real users who have rated AT LEAST ONE of the input movies
    # (users with all zeros are useless neighbors)
    rated_mask = (sub_matrix > 0).any(axis=1)
    sub_matrix_filtered = sub_matrix[rated_mask]

    if sub_matrix_filtered.empty:
        return None

    # ── Step 2: Cosine similarity between new user and all real users ─────────
    sim_scores = cosine_similarity(new_user_vector, sub_matrix_filtered.values)[0]

    # Get top-K neighbor indices
    top_k_local = np.argsort(sim_scores)[::-1][:k]
    neighbor_ids = sub_matrix_filtered.index[top_k_local]
    neighbor_sims = sim_scores[top_k_local]

    # ── Step 3: Collect candidate movies from neighbors ───────────────────────
    # Get full rating rows for all neighbors
    neighbor_ratings = rating_matrix.loc[neighbor_ids]

    # Score each movie: sum of (similarity * rating) across neighbors
    # Only count movies the neighbor actually rated (>0) and new user hasn't seen
    movie_scores = {}

    for sim, (_, neighbor_row) in zip(neighbor_sims, neighbor_ratings.iterrows()):
        for movie_id, rating_val in neighbor_row.items():
            if rating_val >= 4.0 and movie_id not in input_movie_ids:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = 0
                movie_scores[movie_id] += sim * rating_val

    return movie_scores

def get_content_fallback(input_indices, exclude_ids, n=10):
    """
    TF-IDF genre similarity fallback for movies with no collaborative signal.
    Enriched with avg rating so popular quality films rank above obscure ones.
    """
    from scipy.sparse import hstack, csr_matrix

    stats = movie_stats.copy()
    movies_merged = movies.merge(stats, on='movieId', how='left')
    movies_merged['avg_rating'].fillna(movies_merged['avg_rating'].median(), inplace=True)
    movies_merged['rating_count'].fillna(0, inplace=True)

    avg_norm   = (movies_merged['avg_rating']   - movies_merged['avg_rating'].min())   / \
                 (movies_merged['avg_rating'].max()   - movies_merged['avg_rating'].min())
    count_norm = (movies_merged['rating_count'] - movies_merged['rating_count'].min()) / \
                 (movies_merged['rating_count'].max() - movies_merged['rating_count'].min())

    rating_features = csr_matrix(np.column_stack([avg_norm * 0.1, count_norm * 0.1]))
    combined        = hstack([tfidf_matrix * 0.8, rating_features])

    query_vecs = combined[input_indices]
    sim_scores = cosine_similarity(query_vecs, combined).mean(axis=0)

    sim_series = pd.Series(sim_scores, index=movies_merged['movieId'].values)
    sim_series = sim_series.drop(index=list(exclude_ids), errors='ignore')

    return sim_series.sort_values(ascending=False).head(n)

def get_final_recommendations(input_indices, user_ratings_dict, n=10):
    """
    Combines collaborative filtering + content-based:
    - CF provides the primary ranking signal
    - Content-based fills gaps where CF has no signal
    - Final list is ranked by CF score where available,
      content score otherwise (clearly labeled)
    """
    input_movie_ids = set(user_ratings_dict.keys())

    # ── Collaborative filtering ───────────────────────────────────────────────
    cf_scores = get_user_cf_recommendations(user_ratings_dict, n=n*3, k=20)

    results = []

    if cf_scores:
        # Sort CF results and take top N
        sorted_cf = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)

        for movie_id, score in sorted_cf[:n]:
            movie_row = movies[movies['movieId'] == movie_id]
            if movie_row.empty:
                continue
            results.append({
                'title':         movie_row.iloc[0]['title'],
                'title_display': movie_row.iloc[0]['title_display'],
                'genres':        movie_row.iloc[0]['genres'],
                'score':         round(score, 2),
                'source':        'Collaborative'
            })

    # ── Fill remaining slots with content-based ───────────────────────────────
    if len(results) < n:
        cf_movie_ids    = {r['title'] for r in results}
        fallback_scores = get_content_fallback(input_indices, input_movie_ids, n=n)

        for movie_id, score in fallback_scores.items():
            if len(results) >= n:
                break
            movie_row = movies[movies['movieId'] == movie_id]
            if movie_row.empty:
                continue
            if movie_row.iloc[0]['title'] in cf_movie_ids:
                continue
            results.append({
                'title':         movie_row.iloc[0]['title'],
                'title_display': movie_row.iloc[0]['title_display'],
                'genres':        movie_row.iloc[0]['genres'],
                'score':         round(score, 4),
                'source':        'Genre Match'
            })

    return pd.DataFrame(results[:n])

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender")
st.markdown(
    "Rate up to 5 movies you love. The recommender finds real users with "
    "similar taste and recommends what they loved that you haven't seen yet."
)

with st.sidebar:
    st.header("🎥 Rate Your Favorites")
    st.caption("Leave a field blank to use fewer than 5 movies.")

    inputs = []
    for i in range(1, 6):
        col1, col2 = st.columns([2, 1])
        with col1:
            title = st.text_input(f"Movie {i}", key=f"title_{i}",
                                  placeholder=f"Movie title {i}")
        with col2:
            stars = st.selectbox("Stars", [5, 4, 3, 2, 1],
                                 key=f"stars_{i}", index=0)
        inputs.append((title, stars))

    run_btn = st.button("🔍 Get Recommendations", type="primary",
                        use_container_width=True)

# ── On Button Click ───────────────────────────────────────────────────────────
if run_btn:
    input_indices      = []
    matched_titles     = []
    user_ratings_dict  = {}   # {movieId: star_rating}
    unmatched          = []

    for title_input, star_rating in inputs:
        if not title_input.strip():
            continue
        idx = find_best_match(title_input)
        if idx is not None:
            movie_row = movies.iloc[idx]
            input_indices.append(idx)
            matched_titles.append(
                f"{movie_row['title_display']} ({star_rating}⭐)"
            )
            user_ratings_dict[movie_row['movieId']] = float(star_rating)
        else:
            unmatched.append(title_input)

    if unmatched:
        st.warning(
            f"⚠️ No confident match for: **{', '.join(unmatched)}**. "
            "Try adding the year, e.g. 'Inception (2010)'."
        )

    if not input_indices:
        st.error("No valid movies found. Please check your inputs.")
        st.stop()

    st.success(f"Finding recommendations based on: **{', '.join(matched_titles)}**")

    with st.spinner("Finding your taste neighbors..."):
        recs = get_final_recommendations(input_indices, user_ratings_dict, n=10)

    if recs.empty:
        st.error("Could not generate recommendations. Try different movies.")
        st.stop()

    st.subheader("🍿 Recommended for You")

    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        genres_display = row['genres'].replace('|', ' · ')
        badge = "🤝 Collaborative" if row['source'] == 'Collaborative' else "🏷️ Genre Match"
        st.markdown(f"**{i}. {row['title_display']}**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(genres_display)
        with col2:
            st.caption(badge)
        st.divider()