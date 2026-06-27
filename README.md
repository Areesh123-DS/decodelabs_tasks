# AI Intern — DecodeLabs Assignments

Three Tasks built with Python, covering rule-based NLP, machine learning classification, and intelligent career recommendation.

---

## Table of Contents

- [Task 1 — Rule-Based Chatbot](#task-1--rule-based-chatbot)
- [Task 2 — Iris Species Classifier (KNN)](#task-2--iris-species-classifier-knn)
- [Task 3 — AI Job Recommender](#task-3--ai-job-recommender)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Task 1 — Rule-Based Chatbot

A lightweight conversational chatbot built with Streamlit that responds to user messages using keyword-based intent matching. If a keyword is found in the user's message, it returns the mapped response otherwise it falls back to a default reply.

### Supported Intents

| Keyword   | Response                 |
|-----------|--------------------------|
| `hi`      | Greeting response        |
| `name`    | Returns the bot's name   |
| `morning` | Replies to good morning  |
| `eid`     | Eid Mubarak response     |

---

## Task 2 — Iris Species Classifier (KNN)

A complete ML pipeline that classifies Iris flower species using K-Nearest Neighbors, with the optimal K selected automatically via the elbow method.

### Pipeline

Load CSV → Encode labels → Visualize → Scale features → Find best K → Train KNN → Evaluate

### Evaluation Metrics
- Accuracy and F1 Score
- Confusion Matrix
- Full Classification Report

---

## Task 3 — AI Job Recommender

A Streamlit app that recommends career paths by matching a user's skills against job roles using a TF-IDF matrix and cosine similarity pipeline, built from scratch without any ML libraries.

### How It Works

The user enters 3 skills. Each job's skill list is converted into a TF-IDF vector, as is the user's input. Cosine similarity then ranks jobs by how closely they match, and the top 3 results are displayed with a percentage match score.

---

## Requirements

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

**Task 1**
```bash
streamlit run Artificial_Intelligence_P1.py
```

**Task 2** — requires `Iris.csv` in the same directory
```bash
python Artificial Intelligence P2.py
```

**Task 3** — requires `raw_skills.csv` in the same directory
```bash
streamlit run Artificial_Intelligence_P3.py
```

---

> Built as part of the DecodeLabs AI Internship Program.
