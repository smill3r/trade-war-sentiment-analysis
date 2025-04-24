import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configuraciones de estilo
sns.set(style="whitegrid")
st.set_page_config(page_title="Análisis de Sentimiento en Reddit", layout="wide")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("data/comments_analizados_con_sentimiento_cardiffnlp.csv")

df = cargar_datos()

st.title("🗳️ Análisis de Sentimiento de Comentarios de Reddit")

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

st.subheader("Tabla de comentarios")
st.dataframe(df_filtrado[["keyword", "sentiment", "comment"]].reset_index(drop=True))

