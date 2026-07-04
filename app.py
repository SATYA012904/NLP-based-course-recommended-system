import streamlit as st
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title="NLP Based Course Recommendation System",

    page_icon="🎓",

    layout="wide"
)

# ==========================================
# MODERN CSS
# ==========================================

st.markdown(
    """
    <style>

    .stApp {

        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #1e293b
        );

        color: white;
    }

    .main-title {

        text-align: center;

        font-size: 55px;

        font-weight: bold;

        color: #38bdf8;

        margin-top: 20px;

        margin-bottom: 10px;
    }

    .sub-title {

        text-align: center;

        font-size: 22px;

        color: #cbd5e1;

        margin-bottom: 40px;
    }

    .course-card {

        background: rgba(255,255,255,0.08);

        backdrop-filter: blur(12px);

        border-radius: 20px;

        padding: 25px;

        margin-bottom: 25px;

        border: 1px solid rgba(255,255,255,0.1);

        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .course-title {

        font-size: 28px;

        font-weight: bold;

        color: #38bdf8;

        margin-bottom: 15px;
    }

    .course-text {

        font-size: 18px;

        color: #e2e8f0;

        line-height: 1.8;
    }

    div.stButton > button {

        background: linear-gradient(
            90deg,
            #38bdf8,
            #0ea5e9
        );

        color: white;

        border-radius: 12px;

        height: 55px;

        width: 100%;

        font-size: 20px;

        font-weight: bold;

        border: none;
    }

    input {

        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("final_courses.csv")

# ==========================================
# LOAD EMBEDDINGS
# ==========================================

with open("course_embeddings.pkl", "rb") as f:

    course_embeddings = pickle.load(f)

# ==========================================
# LOAD MODEL
# ==========================================

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend_courses_(user_input):

    # Query embedding
    query_embedding = model.encode([user_input])

    # Similarity scores
    similarity_scores = cosine_similarity(

        query_embedding,

        course_embeddings
    )

    # Flatten scores
    similarity_scores = similarity_scores.flatten()

    # Add similarity column
    df["similarity"] = similarity_scores

    # Convert rating to numeric
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    ).fillna(0)

    # ==========================================
    # FILTER ENGLISH COURSES
    # ==========================================

    filtered_df = df[
        df["title"].str.contains(
            r'^[A-Za-z0-9\s:,&()\-]+$',
            regex=True,
            na=False
        )
    ]

    # ==========================================
    # SIMILARITY FILTER
    # ==========================================

    filtered_df = filtered_df[
        filtered_df["similarity"] > 0.35
    ]

    # ==========================================
    # REMOVE DUPLICATES
    # ==========================================

    filtered_df = filtered_df.drop_duplicates(
        subset=["title"]
    )

    # ==========================================
    # FINAL SCORE
    # ==========================================

    filtered_df["normalized_rating"] = (
        filtered_df["rating"] / 5.0
    )

    filtered_df["final_score"] = (

        0.7 * filtered_df["similarity"]

        +

        0.3 * filtered_df["normalized_rating"]
    )

    # ==========================================
    # TOP COURSES
    # ==========================================

    top_rated_courses = filtered_df.sort_values(

        by="final_score",

        ascending=False
    ).head(5)

    # ==========================================
    # REMOVE TOP COURSES
    # ==========================================

    remaining_courses = filtered_df[
        ~filtered_df["title"].isin(
            top_rated_courses["title"]
        )
    ]

    # ==========================================
    # RELATED COURSES
    # ==========================================

    all_related_courses = remaining_courses.sort_values(

        by="similarity",

        ascending=False
    ).head(20)

    return top_rated_courses, all_related_courses

# ==========================================
# TITLE
# ==========================================

st.markdown(
    """
    <div class="main-title">
        🎓 NLP Based Course Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     """
#     <div class="sub-title">
#         Smart Semantic Recommendations using NLP & Sentence Transformers
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# ==========================================
# SEARCH INPUT
# ==========================================

user_input = st.text_input(
    "",
    placeholder="🔍 Search courses like Machine Learning, Python, AI..."
)

# ==========================================
# BUTTON
# ==========================================

recommend = st.button(
    "🚀 Recommend Courses",
    use_container_width=True
)

# ==========================================
# RECOMMENDATION SECTION
# ==========================================

if recommend:

    if user_input:

        with st.spinner("Finding best courses for you..."):

            top_rated, all_related = recommend_courses_(
                user_input
            )

        # ==========================================
        # TOP COURSES
        # ==========================================

    

        st.subheader("🏆 Best Recommendations")

        for _, course in top_rated.iterrows():

            platform = str(course["platform"]).replace(
                "[", ""
            ).replace(
                "]", ""
            ).replace(
                "'", ""
            )

            instructor = str(course["instructor"]).replace(
                "[", ""
            ).replace(
                "]", ""
            ).replace(
                "'", ""
            )

            with st.container(border=True):

                col1, col2 = st.columns([4,1])

                with col1:

                    st.markdown(
                        f"## 🎓 {course['title']}"
                    )

                    st.write(
                        f"⭐ Rating: {course['rating']}"
                    )

                    st.write(
                        f"👨‍🏫 Instructor: {instructor}"
                    )

                    st.write(
                        f"🏫 Platform: {platform}"
                    )

                with col2:

                    st.link_button(
                        "Open Course",
                        course["url"]
                    )

                st.divider()

        # ==========================================
        # RELATED COURSES
        # ==========================================

     

        st.subheader("📚 Explore More Courses")

        for _, course in all_related.iterrows():

            with st.container(border=True):

                col1, col2 = st.columns([4,1])

                with col1:

                    st.markdown(
                        f"### 🎓 {course['title']}"
                    )

                    st.write(
                        f"⭐ Rating: {course['rating']}"
                    )

                with col2:

                    st.link_button(
                        "Open",
                        course["url"]
                    )