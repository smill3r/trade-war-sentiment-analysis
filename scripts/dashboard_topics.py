import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Análisis de Sentimiento y Temas en Reddit con BERTopic", layout="wide")

# Load data
df = pd.read_csv("data/comments_analizados_con_sentimiento_y_temas.csv")
df["topic_label"] = df["topic_label"].fillna("Sin tema")
df["sentiment"] = df["sentiment"].fillna("Sin sentimiento")

st.title("Guerra Arancelaria USA - China: Modelado de Temas de Análisis de Sentimiento con BERTopic")

# Sidebar filters
st.sidebar.header("Filtros")
keywords = df["keyword"].dropna().unique().tolist()
selected_keywords = st.sidebar.multiselect("Palabras clave", keywords, default=keywords)

sentiments = df["sentiment"].unique().tolist()
selected_sentiments = st.sidebar.multiselect("Sentimientos", sentiments, default=sentiments)

# Sort topics by frequency and unselect all by default
topic_counts = df["topic_label"].value_counts()
topics = topic_counts.index.tolist()
selected_topics = st.sidebar.multiselect(
    "Temas (BERTopic)", options=topics, default=[],
    help="Selecciona los temas que te interesen explorar."
)

# Apply filters
if selected_topics:
    df_filtrado = df[
        df["keyword"].isin(selected_keywords) &
        df["sentiment"].isin(selected_sentiments) &
        df["topic_label"].isin(selected_topics)
    ]
else:
    st.warning("Selecciona al menos un tema para visualizar resultados.")
    st.stop()

# Visualizations
# Distribución de sentimientos
st.subheader("Distribución de Sentimiento")
fig1, ax1 = plt.subplots()
sns.countplot(data=df_filtrado, x="sentiment", palette="Set2", ax=ax1)
st.pyplot(fig1)

# Sentimiento promedio por palabra clave
st.subheader("Sentimiento promedio por palabra clave")
fig2, ax2 = plt.subplots()
mean_sent = df_filtrado.groupby("keyword")["sentiment_score"].mean().sort_values()
sns.barplot(x=mean_sent.index, y=mean_sent.values, palette="coolwarm", ax=ax2)
ax2.set_ylabel("Sentiment Score")
ax2.set_xlabel("Palabra clave")
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# Sentimiento a lo largo del tiempo
st.subheader("Sentimiento a lo largo del tiempo")
df_filtrado["created_utc"] = pd.to_datetime(df_filtrado["created_utc"], errors='coerce')
df_filtrado["fecha"] = df_filtrado["created_utc"].dt.date
sent_time = df_filtrado.groupby("fecha")["sentiment_score"].mean()

fig3, ax3 = plt.subplots()
sent_time.plot(ax=ax3, marker='o', linestyle='-')
ax3.set_title("Sentimiento promedio diario")
ax3.set_ylabel("Score")
ax3.set_xlabel("Fecha")
st.pyplot(fig3)

# Wordcloud
st.subheader("Nube de Palabras")
text = " ".join(df_filtrado["comment_clean"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
fig4, ax4 = plt.subplots()
ax4.imshow(wordcloud, interpolation="bilinear")
ax4.axis("off")
st.pyplot(fig4)

# Tópicos más frecuentes
st.subheader("Distribución de Comentarios por Tema")
fig5, ax5 = plt.subplots()
df_filtrado["topic_label"].value_counts().plot(kind="bar", color="orchid", ax=ax5)
ax5.set_xlabel("Tema")
ax5.set_ylabel("Número de comentarios")
st.pyplot(fig5)

# Sentimiento por tema
st.subheader("Distribución de Sentimiento por Tema")
fig6, ax6 = plt.subplots(figsize=(12, 5))
sns.boxplot(x="topic_label", y="sentiment_score", data=df_filtrado, palette="vlag", ax=ax6)
ax6.set_xlabel("Tema")
ax6.set_ylabel("Sentiment Score")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
st.pyplot(fig6)

# Tabla de comentarios
st.subheader("Comentarios filtrados")
st.dataframe(df_filtrado[["keyword", "sentiment", "topic_label", "comment"]].reset_index(drop=True))
