import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos con sentimiento y tópico
df = pd.read_csv("../data/posts_con_topics.csv")

# Calcular sentimiento promedio por tema
df_grouped = df.groupby("topic").agg({
    "sentiment_score": "mean",
    "text_clean": "count"
}).rename(columns={"text_clean": "num_posts"}).reset_index()

# Eliminar tópicos con pocos posts
df_grouped = df_grouped[df_grouped["num_posts"] > 5]

# Visualización
plt.figure(figsize=(12, 6))
sns.barplot(x="topic", y="sentiment_score", data=df_grouped, palette="coolwarm")
plt.title("Sentimiento promedio por tópico")
plt.xlabel("Tópico")
plt.ylabel("Sentimiento promedio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("../visualizaciones/sentimiento_por_topic.png")
plt.show()
