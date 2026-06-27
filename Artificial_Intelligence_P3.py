import streamlit as st
import pandas as pd
import numpy as np
import math

@st.cache_data # for fast loading from the cache every time
def load_data():
    df = pd.read_csv("raw_skills.csv")
    return df

try:
    df = load_data()
    Role_col = 'role' 
    Skills_col = 'skills'
except FileNotFoundError:
    st.error("Could not find 'raw_skills.csv'.")
    st.stop()


st.title("🎯 AI Tech Stack Recommender")
st.write("Calculates TF-IDF Matrix and Cosine Similarity.")

col1, col2, col3 = st.columns(3)
with col1:
    skill1 = st.text_input("Skill 1", placeholder="e.g., Machine Learning").strip()
with col2:
    skill2 = st.text_input("Skill 2", placeholder="e.g., Python").strip()
with col3:
    skill3 = st.text_input("Skill 3", placeholder="e.g., SQL").strip()


if st.button("Recommend My Career Path"):
    if skill1 and skill2 and skill3:
        
        # Keep multi-word phrases intact as individual skill tokens
        user_doc = [skill1.lower(), skill2.lower(), skill3.lower()]
        
        # Parse skills by comma
        documents = []
        for text in df[Skills_col].tolist():
            text_str = str(text).lower()
            tokens = [t.strip() for t in text_str.split(',') if t.strip()]
            documents.append(tokens)
        
        # vocabulary of unique words
        vocabulary = set()
        for doc in documents:
            vocabulary.update(doc)
        vocabulary.update(user_doc) 
        
        vocabulary = sorted(list(vocabulary))
        vocab_index = {word: i for i, word in enumerate(vocabulary)}
        vocab_size = len(vocabulary)
        
        # Calculate IDF for every phrase from scratch
        num_docs = len(documents)
        idf_dict = {}
        for word in vocabulary:
            doc_count = sum(1 for doc in documents if word in doc)
            idf_dict[word] = math.log(num_docs / (1 + doc_count))
            
      #for calculating tf-idf vectors
        def compute_tfidf_vector(tokens):
            vector = np.zeros(vocab_size)
            if not tokens:
                return vector
            
            word_counts = {}
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
                
            for token, count in word_counts.items():
                if token in vocab_index:
                    tf = count / len(tokens)
                    idf = idf_dict[token]
                    vector[vocab_index[token]] = tf * idf
            return vector

        job_vectors = np.array([compute_tfidf_vector(doc) for doc in documents])
        user_vector = compute_tfidf_vector(user_doc)
        
        # for calculating cosine similarity
        similarity_scores = []
        user_magnitude = np.sqrt(np.sum(user_vector ** 2))
        
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
        
        if not valid_recommendations.empty:
            st.success("### 🚀 Your Top Recommended Career Paths:")
            for index, row in valid_recommendations.iterrows():
                with st.container():
                    st.markdown(f"#### **{row[Role_Col]}**")
                    st.caption(f"Cosine Match Score: {round(row['Similarity Score'] * 100, 2)}%")
                    st.write(f"**Dataset Core Skills:** {row[Skills_col]}")
                    st.markdown("---")
        else:
            st.warning("No matches found with a similarity score greater than 0. Try entering different keywords!")
            
    else:
        st.warning("Please fill out all 3 fields.")
