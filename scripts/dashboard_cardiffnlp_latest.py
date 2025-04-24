import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load data
df = pd.read_csv("../data/comments_analizados_con_sentimiento_latest.csv")

# Streamlit settings
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")
keywords = df["keyword"].dropna().unique().tolist()
selected_keywords = st.sidebar.multiselect("Select keywords", keywords, default=keywords)

sentiments = df["sentiment"].dropna().unique().tolist()
selected_sentiments = st.sidebar.multiselect("Select sentiments", sentiments, default=sentiments)

# Filter data based on selections
df_filtered = df[df["keyword"].isin(selected_keywords) & df["sentiment"].isin(selected_sentiments)]

# Layout with columns
col1, col2 = st.columns(2)

# Sentiment Distribution Plot
with col1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df_filtered["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Word Cloud Plot
with col2:
    st.subheader("Word Cloud of Comments")
    text = " ".join(df_filtered["comment_clean"].dropna())
    wordcloud = WordCloud(width=600, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Sentiment by Keyword
st.subheader("Average Sentiment by Keyword")
keyword_sentiment = df_filtered.groupby("keyword")["sentiment_score"].mean().sort_values()
fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x=keyword_sentiment.index, y=keyword_sentiment.values, palette="coolwarm", ax=ax)
ax.set_ylabel("Sentiment Score")
ax.set_xlabel("Keyword")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Sentiment Over Time (Time series plot)
st.subheader("Sentiment Over Time")
df_filtered['created_utc'] = pd.to_datetime(df_filtered['created_utc'], errors='coerce')
df_filtered['date'] = df_filtered['created_utc'].dt.date
sentiment_time = df_filtered.groupby('date')['sentiment_score'].mean()

fig, ax = plt.subplots(figsize=(12,6))
sentiment_time.plot(kind='line', ax=ax, color='b', marker='o', linestyle='-', linewidth=2)
ax.set_title("Average Sentiment Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Average Sentiment Score")
plt.xticks(rotation=45)
st.pyplot(fig)

# Boxplot of Sentiment by Keyword
st.subheader("Sentiment Distribution by Keyword (Boxplot)")
fig, ax = plt.subplots(figsize=(12,6))
sns.boxplot(x="keyword", y="sentiment_score", data=df_filtered, palette="vlag", ax=ax)
ax.set_title("Sentiment Score Distribution by Keyword")
ax.set_xlabel("Keyword")
ax.set_ylabel("Sentiment Score")
plt.xticks(rotation=45)
st.pyplot(fig)

# Correlation Heatmap (Sentiment Score and other numeric columns)
st.subheader("Correlation Heatmap")
corr = df_filtered[['sentiment_score', 'comment_score', 'num_comments']].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt='.2f')
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

# Display filtered comments in table format
st.subheader("Filtered Comments")
st.dataframe(df_filtered[["keyword", "sentiment", "comment"]].reset_index(drop=True))
