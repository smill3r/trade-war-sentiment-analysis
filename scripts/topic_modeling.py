import os
import warnings
import pandas as pd
import numpy as np
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

# Load data
df = pd.read_csv("../data/comments_analizados_con_sentimiento_cardiffnlp.csv")

# Enhanced text filtering with proper null handling
df_filtered = df[
    df["comment_clean"].apply(
        lambda x: isinstance(x, str) and 
        len(str(x).strip()) > 20 and  # Convert to string first
        not any(marker in str(x) for marker in ["[deleted]", "[removed]"])
    )
].copy()

# Custom vectorizer to exclude short words
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),  # Keep your bigrams
    stop_words="english",  # Remove common stopwords
    token_pattern=r'(?u)\b[A-Za-z]{4,}\b'  # Only words with 4+ letters
)

print(f"📊 Working with {len(df_filtered)} valid comments")

# Initialize model with optimized parameters
topic_model = BERTopic(
    language="english",
    n_gram_range=(1, 2),
    vectorizer_model=vectorizer_model,
    low_memory=True,
    calculate_probabilities=True,
    verbose=True
)

# Fit and transform
topics, probs = topic_model.fit_transform(df_filtered["comment_clean"])

# Generate visualizations
fig = topic_model.visualize_barchart(top_n_topics=20)
fig.write_html("../visualizations/topic_barchart.html")

# Assign topics
df_filtered["topic"] = topics
df_filtered["topic_prob"] = [np.max(p) if p is not None else None for p in probs]

# Generate improved labels
topic_info = topic_model.get_topic_info()
topic_labels = {
    row["Topic"]: " | ".join([word for word, _ in topic_model.get_topic(row["Topic"])[:3]])
    for _, row in topic_info.iterrows()
}

# Map labels
df_filtered["topic_label"] = df_filtered["topic"].map(topic_labels).fillna("Outlier")

# Merge back with original data
df = df.merge(
    df_filtered[["topic", "topic_prob", "topic_label"]],
    how="left",
    left_index=True,
    right_index=True
).fillna({
    "topic": -1,
    "topic_prob": 0,
    "topic_label": "Sin tema"
})

# Save results
df.to_csv("../data/comments_analizados_con_sentimiento_y_temas.csv", index=False)

print("\n🔍 Final Topic Distribution:")
print(topic_model.get_topic_info()[["Topic", "Name", "Count"]].sort_values("Count", ascending=False))
print("\n✅ Analysis complete!")
