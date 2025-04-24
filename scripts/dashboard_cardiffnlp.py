import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configuraciones de estilo
sns.set(style="whitegrid")
st.set_page_config(page_title="Análisis de Sentimiento en Reddit con RoBERTa", layout="wide")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("data/comments_analizados_con_sentimiento_cardiffnlp.csv")

df = cargar_datos()

st.title("Guerra Arancelaria USA - China: Análisis de Sentimiento de Comentarios de Reddit con RoBERTa")

# Sidebar de filtros
st.sidebar.header("Filtros")
keywords = df["keyword"].dropna().unique().tolist()
selected_keywords = st.sidebar.multiselect("Selecciona palabras clave:", keywords, default=keywords)

sentimientos = df["sentiment"].dropna().unique().tolist()
selected_sentiments = st.sidebar.multiselect("Selecciona sentimientos:", sentimientos, default=sentimientos)

# Filtrado
df_filtrado = df[df["keyword"].isin(selected_keywords) & df["sentiment"].isin(selected_sentiments)]

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución de sentimientos")
    counts = df_filtrado["sentiment"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, palette="Set2", ax=ax)
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)

with col2:
    st.subheader("Nube de palabras")
    texto = " ".join(df_filtrado["comment_clean"].dropna())
    nube = WordCloud(width=600, height=400, background_color="white").generate(texto)
    fig, ax = plt.subplots()
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

st.subheader("Sentimiento promedio por palabra clave")
graf_data = df_filtrado.groupby("keyword")["sentiment_score"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x=graf_data.index, y=graf_data.values, palette="coolwarm", ax=ax)
ax.set_ylabel("Score promedio (-1 negativo, +1 positivo)")
ax.set_xlabel("Palabra clave")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Sentiment Over Time (Time series plot)
st.subheader("Sentimiento a lo largo del Tiempo")
df_filtrado['created_utc'] = pd.to_datetime(df_filtrado['created_utc'], errors='coerce')
df_filtrado = df_filtrado[df_filtrado['created_utc'].dt.year == 2025]
df_filtrado['date'] = df_filtrado['created_utc'].dt.date
sentiment_time = df_filtrado.groupby('date')['sentiment_score'].mean()

fig, ax = plt.subplots(figsize=(12,6))
sentiment_time.plot(kind='line', ax=ax, color='b', marker='o', linestyle='-', linewidth=2)
ax.set_title("Sentimiento Promedio a lo largo del Tiempo")
ax.set_xlabel("Fecha")
ax.set_ylabel("Sentimiento Promedio")
plt.xticks(rotation=45)
st.pyplot(fig)

# Boxplot of Sentiment by Keyword
st.subheader("Distribucion del Sentimiento por Palabra Clave (Boxplot)")
fig, ax = plt.subplots(figsize=(12,6))
sns.boxplot(x="keyword", y="sentiment_score", data=df_filtrado, palette="vlag", ax=ax)
ax.set_title("Distribucion de Puntuacion del Sentimiento por Palabra Clave")
ax.set_xlabel("Palabra Clave")
ax.set_ylabel("Puntuacion del Sentimiento")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Tabla de comentarios")
st.dataframe(df_filtrado[["keyword", "sentiment", "comment"]].reset_index(drop=True))
