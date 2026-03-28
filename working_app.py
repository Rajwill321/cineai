import streamlit as st
import pandas as pd
import numpy as np
import pickle
from llm_client import get_llm_recommendations

# ---------------- LOAD FILES ---------------- #

model          = pickle.load(open('xgboost_model.pkl', 'rb'))
movies         = pd.read_csv('movies.csv')
user_features  = pd.read_csv('features_user.csv')
movie_features = pd.read_csv('features_movie.csv')
user_svd       = pd.read_csv('features_user_svd.csv')
movie_svd      = pd.read_csv('features_movie_svd.csv')

# Feature columns — order confirmed from model.feature_names_in_
USER_BASE_COLS  = ['user_total_ratings', 'user_avg_rating', 'user_rating_std']
MOVIE_BASE_COLS = ['movie_total_ratings', 'movie_avg_rating', 'movie_rating_std']
BIAS_COLS       = ['user_bias', 'movie_bias']
USER_SVD_COLS   = [f'user_svd_{i}'  for i in range(50)]
MOVIE_SVD_COLS  = [f'movie_svd_{i}' for i in range(50)]
FEATURE_COLS    = USER_BASE_COLS + MOVIE_BASE_COLS + BIAS_COLS + USER_SVD_COLS + MOVIE_SVD_COLS

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(page_title="CineAI", layout="wide")

# ---------------- STYLES ---------------- #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .stApp { background-color: #0f0f0f; color: #e0e0e0; }

.app-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 52px; letter-spacing: 3px;
    color: #E50914; margin-bottom: 0; line-height: 1;
}
.app-sub {
    font-family: 'DM Sans', sans-serif; font-size: 13px; color: #666;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 28px;
}
.section-header {
    font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 500;
    letter-spacing: 3px; text-transform: uppercase; color: #555; margin: 32px 0 4px 0;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 28px; letter-spacing: 2px;
    color: #e0e0e0; margin-bottom: 6px; line-height: 1;
}
.section-subtitle {
    font-family: 'DM Sans', sans-serif; font-size: 13px; color: #555;
    margin-bottom: 14px; line-height: 1.5;
}
.section-divider {
    height: 1px;
    background: linear-gradient(to right, #E50914 0%, #333 60%, transparent 100%);
    margin-bottom: 16px;
}
.movie-row {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 14px; background: #181818;
    border-radius: 8px; margin-bottom: 6px; border: 1px solid #222;
}
.movie-row:hover { border-color: #E50914; background: #1e1e1e; }
.movie-thumb { width: 48px; height: 68px; border-radius: 5px; object-fit: cover; flex-shrink: 0; }
.movie-info  { flex: 1; min-width: 0; }
.movie-name  {
    font-family: 'DM Sans', sans-serif; font-size: 14px; font-weight: 500; color: #f0f0f0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 3px;
}
.movie-meta   { font-family: 'DM Sans', sans-serif; font-size: 11px; color: #555; margin-bottom: 3px; }
.movie-reason { font-family: 'DM Sans', sans-serif; font-size: 12px; color: #888;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.score-raw  { flex-shrink:0; font-family:'DM Sans',sans-serif; font-size:12px;
              font-weight:500; color:#f5c518; min-width:52px; text-align:right; }
.score-final{ flex-shrink:0; font-family:'DM Sans',sans-serif; font-size:12px;
              font-weight:500; color:#46d369; min-width:52px; text-align:right; }
.boost-pill {
    flex-shrink: 0; font-family: 'DM Sans', sans-serif; font-size: 10px;
    color: #f5a623; background: #2a1d00; border: 1px solid #3a2a00;
    padding: 2px 7px; border-radius: 20px; white-space: nowrap;
}
.badge-ml {
    flex-shrink: 0; font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 500;
    letter-spacing: 1px; color: #46d369; background: #0d2b1a;
    border: 1px solid #1a4a2a; padding: 3px 9px; border-radius: 20px;
}
.badge-ai {
    flex-shrink: 0; font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 500;
    letter-spacing: 1px; color: #ff6b6b; background: #2b0d0d;
    border: 1px solid #4a1a1a; padding: 3px 9px; border-radius: 20px;
}
.insight-box {
    background: #111; border: 1px solid #2a2a2a; border-left: 3px solid #E50914;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;
    font-family: 'DM Sans', sans-serif; font-size: 13px; color: #888; line-height: 1.7;
}
.insight-box b { color: #ccc; }
.warn-box {
    background: #1a1500; border: 1px solid #3a3000; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px;
    font-family: 'DM Sans', sans-serif; font-size: 13px; color: #aaa;
}
.empty-state { font-family: 'DM Sans', sans-serif; font-size: 13px; color: #444;
               padding: 20px 0; text-align: center; }
section[data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #1e1e1e; }
section[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }
.stButton > button {
    background: #E50914; color: white; font-family: 'DM Sans', sans-serif;
    font-weight: 500; letter-spacing: 1px; border: none;
    border-radius: 6px; padding: 10px 20px; width: 100%;
}
.stButton > button:hover { background: #b0070f; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown('<div class="app-title">CineAI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.markdown("### Step 1 — User")
user_id = st.sidebar.number_input("User ID", min_value=1, step=1, value=1)

st.sidebar.markdown("### Step 2 — Preferences")
all_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Horror", "Animation"]
selected_genre  = st.sidebar.selectbox("Favourite Genre", all_genres)
movie_list      = movies['title'].dropna().unique()
selected_recent = st.sidebar.selectbox("Recently Watched", movie_list)

st.sidebar.markdown("### Step 3 — AI Query")
user_query = st.sidebar.text_input("What are you in the mood for?",
                                   placeholder="e.g. latest sci-fi 2025")
st.sidebar.markdown("---")

# Three separate buttons — one per section
btn_raw    = st.sidebar.button("1 — Show Raw Model Scores",      use_container_width=True)
btn_boost  = st.sidebar.button("2 — Apply Personalization Boost", use_container_width=True)
btn_ai     = st.sidebar.button("3 — Get AI Recommendations",     use_container_width=True)

# ---------------- SESSION STATE ---------------- #
# Keep results alive between button clicks so sections stack up

if 'raw_results' not in st.session_state:
    st.session_state.raw_results    = None
if 'boost_results' not in st.session_state:
    st.session_state.boost_results  = None
if 'llm_results' not in st.session_state:
    st.session_state.llm_results    = None
if 'user_not_found' not in st.session_state:
    st.session_state.user_not_found = False

# ---------------- HELPERS ---------------- #

def get_poster(seed):
    return f"https://picsum.photos/seed/{seed}/96/136"

def render_raw_rows(df, max_rows=10):
    """Raw model scores — no boosts applied."""
    left, right = st.columns(2)
    for i, row in enumerate(df.head(max_rows).to_dict("records")):
        title  = str(row.get("title", "Unknown"))
        score  = round(float(row.get("pred_rating", 0)), 3)
        genres = str(row.get("genres", ""))
        seed   = int(row.get("movieId", i + 1))
        col    = left if i % 2 == 0 else right
        with col:
            st.markdown(f"""
            <div class="movie-row">
                <img class="movie-thumb" src="{get_poster(seed)}" />
                <div class="movie-info">
                    <div class="movie-name">{title}</div>
                    <div class="movie-meta">{genres[:50] if genres else "—"}</div>
                </div>
                <div class="score-raw">&#11088; {score}</div>
                <span class="badge-ml">ML</span>
            </div>
            """, unsafe_allow_html=True)

def render_boost_rows(df, max_rows=10):
    """Shows raw score, what boost was added, and the final score."""
    left, right = st.columns(2)
    for i, row in enumerate(df.head(max_rows).to_dict("records")):
        title       = str(row.get("title", "Unknown"))
        pred        = round(float(row.get("pred_rating", 0)), 3)
        genre_b     = float(row.get("genre_boost", 0))
        recent_b    = float(row.get("recent_boost", 0))
        final       = round(float(row.get("final_score", 0)), 3)
        genres      = str(row.get("genres", ""))
        seed        = int(row.get("movieId", i + 1))
        col         = left if i % 2 == 0 else right

        # Build boost label — only show boosts that were applied
        boost_parts = []
        if genre_b > 0:
            boost_parts.append(f"genre +{genre_b}")
        if recent_b > 0:
            boost_parts.append(f"recent +{recent_b}")
        boost_label = "  ".join(boost_parts) if boost_parts else "no boost"

        with col:
            st.markdown(f"""
            <div class="movie-row">
                <img class="movie-thumb" src="{get_poster(seed)}" />
                <div class="movie-info">
                    <div class="movie-name">{title}</div>
                    <div class="movie-meta">{genres[:40] if genres else "—"}</div>
                </div>
                <span class="boost-pill">{boost_label}</span>
                <div class="score-raw" style="text-decoration:line-through;opacity:0.4;min-width:44px">{pred}</div>
                <div class="score-final">&#11088; {final}</div>
            </div>
            """, unsafe_allow_html=True)

def render_llm_rows(movies_list, max_rows=10):
    left, right = st.columns(2)
    for i, movie in enumerate(movies_list[:max_rows]):
        title  = str(movie.get("title", "Unknown"))
        reason = str(movie.get("reason", ""))
        seed   = (abs(hash(title)) % 900) + 100
        col    = left if i % 2 == 0 else right
        with col:
            st.markdown(f"""
            <div class="movie-row">
                <img class="movie-thumb" src="{get_poster(seed)}" />
                <div class="movie-info">
                    <div class="movie-name">{title}</div>
                    <div class="movie-reason">{reason[:80]}{'...' if len(reason) > 80 else ''}</div>
                </div>
                <span class="badge-ai">AI</span>
            </div>
            """, unsafe_allow_html=True)

# ---------------- CORE MODEL FUNCTION ---------------- #

def run_model(user_id):
    """
    Returns full dataframe with pred_rating, genre_boost, recent_boost,
    final_score for every movie. Returns None if user not found.
    """
    user_base_row = user_features[user_features['userId'] == user_id]
    user_svd_row  = user_svd[user_svd['userId'] == user_id]

    if user_base_row.empty:
        return None

    u_base = user_base_row.iloc[0]
    u_svd  = (user_svd_row.iloc[0]
              if not user_svd_row.empty
              else pd.Series({c: 0.0 for c in USER_SVD_COLS}))

    movies_merged = (movie_features
                     .merge(movie_svd, on='movieId', how='left')
                     .merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
                     .fillna(0))

    for col in USER_BASE_COLS + BIAS_COLS:
        movies_merged[col] = float(u_base.get(col, 0.0))
    for col in USER_SVD_COLS:
        movies_merged[col] = float(u_svd.get(col, 0.0))

    X = movies_merged[FEATURE_COLS].values
    movies_merged['pred_rating'] = model.predict(X)

    movies_merged['genre_boost'] = movies_merged['genres'].apply(
        lambda x: 0.5 if selected_genre.lower() in str(x).lower() else 0
    )
    movies_merged['recent_boost'] = movies_merged['title'].apply(
        lambda x: 0.3 if str(selected_recent).split()[0].lower() in str(x).lower() else 0
    )
    movies_merged['final_score'] = (
        movies_merged['pred_rating'] +
        movies_merged['genre_boost'] +
        movies_merged['recent_boost']
    )
    return movies_merged

# ---------------- BUTTON ACTIONS ---------------- #

if btn_raw:
    with st.spinner("Running model..."):
        df = run_model(user_id)
    if df is None:
        st.session_state.user_not_found = True
        st.session_state.raw_results    = None
    else:
        st.session_state.user_not_found = False
        # Sort by raw pred_rating only — no boosts
        st.session_state.raw_results = df.sort_values('pred_rating', ascending=False).head(10)
    # Reset downstream sections when re-running step 1
    st.session_state.boost_results = None
    st.session_state.llm_results   = None

if btn_boost:
    if st.session_state.raw_results is None and not st.session_state.user_not_found:
        st.warning("Run Step 1 first.")
    else:
        with st.spinner("Applying boosts..."):
            df = run_model(user_id)
        if df is not None:
            st.session_state.boost_results = df.sort_values('final_score', ascending=False).head(10)

if btn_ai:
    llm_query = user_query.strip() if user_query.strip() else "best trending movies right now"
    with st.spinner("Fetching AI picks..."):
        st.session_state.llm_results = get_llm_recommendations(
            query=llm_query,
            genre=selected_genre,
            recent_movie=str(selected_recent),
        )

# ---------------- RENDER SECTIONS ---------------- #

# ── Section 1: Raw model scores ───────────────────────────────────────────────
if st.session_state.raw_results is not None or st.session_state.user_not_found:

    st.markdown('<div class="section-header">Step 1 — XGBoost output</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Raw Model Scores</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Pure model prediction — 108 features in, predicted rating out. No genre or preference applied yet.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.session_state.user_not_found:
        st.markdown('<div class="warn-box">User ID not found in training data.</div>',
                    unsafe_allow_html=True)
    else:
        # Show what the model knows about this user
        user_row = user_features[user_features['userId'] == user_id].iloc[0]
        st.markdown(f"""
        <div class="insight-box">
            <b>User {user_id} profile used by model</b><br>
            Total ratings given: <b>{int(user_row['user_total_ratings'])}</b> &nbsp;|&nbsp;
            Avg rating: <b>{round(user_row['user_avg_rating'], 2)}</b> &nbsp;|&nbsp;
            Std dev: <b>{round(user_row['user_rating_std'], 2)}</b> &nbsp;|&nbsp;
            User bias: <b>{round(user_row['user_bias'], 3)}</b>
            <br><span style="color:#444; font-size:12px;">
            + 50 SVD taste factors looked up automatically
            </span>
        </div>
        """, unsafe_allow_html=True)
        render_raw_rows(st.session_state.raw_results)

# ── Section 2: After personalization boost ────────────────────────────────────
if st.session_state.boost_results is not None:

    st.markdown('<div class="section-header">Step 2 — Re-ranking layer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">After Personalization Boost</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtitle">Genre match <b style="color:#f5a623">+0.5</b> for <b style="color:#f5a623">{selected_genre}</b> &nbsp;·&nbsp; Recent watch keyword match <b style="color:#f5a623">+0.3</b> for <b style="color:#f5a623">{str(selected_recent)[:30]}</b><br>Strikethrough = raw score &nbsp;·&nbsp; Green = final score after boost</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_boost_rows(st.session_state.boost_results)

# ── Section 3: AI recommendations ─────────────────────────────────────────────
if st.session_state.llm_results is not None:

    llm_label = f'Based on &ldquo;{user_query}&rdquo;' if user_query.strip() else "Trending Now"
    st.markdown('<div class="section-header">Step 3 — LLM layer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{llm_label}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Live recommendations from the LLM — internet-aware, query-driven, not limited to the MovieLens dataset.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.session_state.llm_results:
        render_llm_rows(st.session_state.llm_results)
    else:
        st.markdown('<div class="empty-state">Could not fetch AI recommendations — try again.</div>',
                    unsafe_allow_html=True)
