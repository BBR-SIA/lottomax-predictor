# Lotto Max Predictive Engine with Streamlit Dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(layout="wide")
st.title("🎯 Lotto Max Predictive Engine Dashboard")

# Load historical data
uploaded_file = st.file_uploader("Upload your Lotto Max CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Extract only winning number columns
    number_columns = data.columns[2:9]  # Adjust if needed
    all_draws = data[number_columns].apply(pd.to_numeric, errors='coerce').dropna().astype(int)

    # ----- 1. Temporal Pattern Module -----
    def get_temporal_scores(draws):
        decay_rate = 0.97
        scores = np.zeros(51)
        for i, draw in enumerate(reversed(draws.values)):
            for num in draw:
                scores[num] += decay_rate ** i
        return scores / scores.sum()

    temporal_scores = get_temporal_scores(all_draws)

    # ----- 2. Co-Occurrence Matrix -----
    def build_cooccurrence_matrix(draws):
        matrix = np.zeros((50, 50))
        for draw in draws.values:
            for i in draw:
                for j in draw:
                    if i != j:
                        matrix[i - 1][j - 1] += 1
        return matrix / matrix.sum()

    co_occurrence = build_cooccurrence_matrix(all_draws)

    # ----- 3. Multivariate Score -----
    def get_composite_score(temporal, co_matrix):
        freq = np.array([Counter(all_draws.values.flatten())[i] for i in range(1, 51)])
        recency = np.zeros(50)
        for i in range(1, 51):
            indices = np.where(all_draws.values == i)
            if len(indices[0]) > 0:
                last_appearance = np.max(indices[0])
                recency[i - 1] = 1 / (len(all_draws) - last_appearance + 1)
        co_score = co_matrix.sum(axis=1)
        temporal_used = temporal[1:] if len(temporal) == 51 else temporal
        total = (
            0.4 * temporal_used +
            0.3 * (freq / freq.sum()) +
            0.2 * recency +
            0.1 * co_score
        )
        return total / total.sum()

    composite_score = get_composite_score(temporal_scores, co_occurrence)

    # ----- 4. Machine Learning Core (Random Forest) -----
    mlb = MultiLabelBinarizer(classes=range(1, 51))
    y = mlb.fit_transform([set(draw) for draw in all_draws.values])
    X = []
    for i in range(len(y) - 1):
        X.append(y[i])
        y_target = y[i + 1]

    X = np.array(X)
    y_target = y[1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ----- Prediction -----
    def generate_prediction(score, co_matrix):
        top = np.argsort(score)[-10:][::-1]
        draws = []
        for _ in range(10):
            draw = set()
            for i in top:
                if len(draw) < 3:
                    draw.add(i + 1)
            co_related = co_matrix[list(draw)].sum(axis=0)
            best_partners = np.argsort(co_related)[-10:][::-1]
            for i in best_partners:
                if len(draw) < 7 and (i + 1) not in draw:
                    draw.add(i + 1)
            draws.append(sorted(draw))
        return draws

    predicted_draws = generate_prediction(composite_score, co_occurrence)

    # ----- UI: Charts and Predictions -----
    st.header("🔢 Predicted Draws (Top 10)")
    for i, d in enumerate(predicted_draws):
        st.write(f"Draw {i+1}: {d}")

    st.header("📊 Heatmap of Number Co-Occurrence")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(co_occurrence, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.header("📈 Temporal Scores for Numbers")
    fig2, ax2 = plt.subplots()
    ax2.bar(range(1, 51), temporal_scores[1:], color="teal")
    ax2.set_title("Temporal Score (Recency-weighted)")
    ax2.set_xlabel("Number")
    ax2.set_ylabel("Score")
    st.pyplot(fig2)

else:
    st.warning("Please upload a Lotto Max CSV file to begin analysis.")
