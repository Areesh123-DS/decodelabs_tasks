import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="AI Job Recommender", page_icon="", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f0f2f7;
    }
    .header-banner {
        background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 60%, #3d4a8a 100%);
        border-radius: 20px;
        padding: 48px 40px 44px 40px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(26,31,60,0.18);
    }
    .header-banner h1 {
        font-family: 'Sora', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 14px 0;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        font-size: 1.05rem;
        color: #b0b8d8;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.7;
        font-weight: 400;
    }
    .header-accent {
        display: inline-block;
        background: linear-gradient(90deg, #5b73e8, #8b9cf4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #5b73e8;
        margin-bottom: 6px;
    }
    .section-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1f3c;
        margin-bottom: 20px;
    }

    .input-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 28px 32px 24px 32px;
        box-shadow: 0 2px 16px rgba(26,31,60,0.07);
        margin-bottom: 24px;
    }

    .stTextInput > label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #3d4a8a;
        letter-spacing: 0.3px;
    }
    .stTextInput > div > div > input {
        border-radius: 10px !important; # for overriding streamlit internal css
        border: 1.5px solid #dde1f0 !important;
        background: #f7f8fc !important;
        font-size: 0.95rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #5b73e8 !important;
        background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(91,115,232,0.12) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2d3561, #5b73e8) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 12px 28px !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 16px rgba(91,115,232,0.30) !important;
        transition: opacity 0.2s, transform 0.15s !important;
    }
    .stButton > button:hover {
        opacity: 0.92 !important;
        transform: translateY(-1px) !important;
    }

    hr {
        border: none;
        border-top: 1.5px solid #e3e6f0;
        margin: 28px 0;
    }

    .results-heading {
        font-family: 'Sora', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1f3c;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e3e6f0;
    }
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 2px 16px rgba(26,31,60,0.07);
        border-left: 5px solid #5b73e8;
        transition: box-shadow 0.2s;
    }
    .result-card:hover {
        box-shadow: 0 6px 28px rgba(26,31,60,0.12);
    }
    .result-role {
        font-family: 'Sora', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1f3c;
        margin: 0;
    }
    .result-score {
        font-size: 1.05rem;
        font-weight: 700;
        color: #5b73e8;
        background: #eef0fc;
        padding: 5px 14px;
        border-radius: 20px;
    }
    .skills-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8b95b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 14px 0 8px 0;
    }
    .skill-badge {
        background-color: #eef0fc;
        color: #3d4a8a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        letter-spacing: 0.2px;
    }

    .rank-pill {
        display: inline-block;
        background: #2d3561;
        color: #ffffff;
        font-size: 0.7rem;
        font-weight: 700;
        border-radius: 6px;
        padding: 2px 9px;
        margin-right: 10px;
        letter-spacing: 1px;
        vertical-align: middle;
    }

    .stAlert {
        border-radius: 12px !important;
    }

    .footer {
        text-align: center;
        color: #9ba3c0;
        font-size: 0.78rem;
        margin-top: 48px;
        padding-top: 20px;
        border-top: 1.5px solid #e3e6f0;
        letter-spacing: 0.3px;
    }
    </style>
""", unsafe_allow_html=True)


# ── HERO BANNER ──────────────────────────────────────────────────────────────
st.markdown("""
    <div class="header-banner">
        <h1>AI <span class="header-accent">Job</span> Recommender</h1>
        <p>Enter three skills and discover the career paths that align with your strengths.
        Powered by TF-IDF matrix and cosine similarity engine.</p>
    </div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("raw_skills.csv", on_bad_lines='skip', engine='python')
    return df

try:
    df = load_data()
    Role_col = 'role'
    Skills_col = 'skills'
except FileNotFoundError:
    st.error("Could not find 'raw_skills.csv'.")
    st.stop()


st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Enter Your Top 3 Skills</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    skill1 = st.text_input("Primary Skill", placeholder="e.g., Machine Learning").strip()
with col2:
    skill2 = st.text_input("Secondary Skill", placeholder="e.g., Python").strip()
with col3:
    skill3 = st.text_input("Third Skill", placeholder="e.g., SQL").strip()

st.markdown("<br>", unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
with btn_col2:
    submit_clicked = st.button("Find Matching Roles", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)



if submit_clicked:
    if skill1 and skill2 and skill3:
        user_doc = [skill1.lower(), skill2.lower(), skill3.lower()]
        documents = []
        for text in df[Skills_col].tolist():
            text_str = str(text).lower()
            tokens = [t.strip() for t in text_str.split(',') if t.strip()]
            documents.append(tokens)

        vocab = set()
        for doc in documents:
            vocab.update(doc)
        vocab.update(user_doc)
        vocab = sorted(list(vocab))
        vocab_index = {word: i for i, word in enumerate(vocab)}
        vocab_size = len(vocab)
        num_docs = len(documents)

        idf_dict = {}
        for word in vocab:
            doc_count = sum(1 for doc in documents if word in doc)
            idf_dict[word] = math.log(num_docs / (1 + doc_count))

        def compute_tfidf(tokens):
            vector = np.zeros(vocab_size)
            if not tokens:
                return vector
            word_counts = {}
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            for token, count in word_counts.items():
                if token in vocab_index:
                    tf = count / len(tokens)
                    vector[vocab_index[token]] = tf * idf_dict[token]
            return vector

        job_vectors = np.array([compute_tfidf(doc) for doc in documents])
        user_vector = compute_tfidf(user_doc)
        user_magnitude = np.sqrt(np.sum(user_vector ** 2))

        similarity_scores = []
        for job_vec in job_vectors:
            dot_product = np.sum(user_vector * job_vec)
            job_magnitude = np.sqrt(np.sum(job_vec ** 2))
            if user_magnitude == 0 or job_magnitude == 0:
                score = 0.0
            else:
                score = dot_product / (user_magnitude * job_magnitude)
            similarity_scores.append(score)

        df['Similarity Score'] = similarity_scores
        top_recommendations = df.sort_values(by='Similarity Score', ascending=False).head(3)
        valid_recommendations = top_recommendations[top_recommendations['Similarity Score'] > 0]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="results-heading">Top Matching Career Paths</div>', unsafe_allow_html=True)

        if not valid_recommendations.empty:
            rank_labels = ["BEST MATCH", "2ND MATCH", "3RD MATCH"]
            for i, (index, row) in enumerate(valid_recommendations.iterrows()):
                percentage_score = round(row['Similarity Score'] * 100, 2)
                rank = rank_labels[i] if i < len(rank_labels) else f"#{i+1}"
                badges = "".join([
                    f'<span class="skill-badge">{s.strip()}</span>'
                    for s in str(row[Skills_col]).split(',')
                ])
                st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <span class="rank-pill">{rank}</span>
                                <span class="result-role">{row[Role_col]}</span>
                            </div>
                            <span class="result-score">{percentage_score}% Match</span>
                        </div>
                        <div class="skills-label">Skill Alignment</div>
                        <div>{badges}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matches found with a similarity score above 0%. Try different or broader skill keywords.")
    else:
        st.error("Please fill in all three skill fields to get recommendations.")

st.markdown("""
    <div class="footer">
        AI Job Recommender &nbsp;&bull;&nbsp; TF-IDF + Cosine Similarity &nbsp;&bull;&nbsp; Built with Streamlit
    </div>
""", unsafe_allow_html=True)
