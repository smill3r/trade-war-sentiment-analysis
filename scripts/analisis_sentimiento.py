import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Cargar los datos
df = pd.read_csv("../data/comments_reddit.csv")

# --- Paso 1: Reemplazar NaN en la columna de texto con una cadena vacía ---
df["comment"] = df["comment"].fillna("")  # Reemplazar NaN por ""

# --- Paso 2: Limpiar el texto ---
def limpiar_texto(texto):
    texto = str(texto).strip()  # Asegurarse de que no tenga espacios antes y después
    if len(texto) > 0:
        texto = re.sub(r"http\S+|www.\S+", "", texto)  # quitar URLs
        texto = re.sub(r"[^a-zA-Z0-9\s.,!?;]", "", texto)   # Permitir puntuación básica
        texto = texto.lower()
    return texto if len(texto.strip()) > 1 else None  # Dejar que textos pequeños pero significativos no se eliminen

# Aplicar limpieza al comentario
df["comment_clean"] = df["comment"].apply(limpiar_texto)

# --- Paso 3: Análisis de sentimiento ---
analyzer = SentimentIntensityAnalyzer()

def obtener_sentimiento(texto):
    if texto:  # Solo procesar si el texto no es vacío
        scores = analyzer.polarity_scores(texto)
        return scores["compound"]
    return None  # Devolver None si no hay texto

df["sentiment_score"] = df["comment_clean"].apply(obtener_sentimiento)

# --- Paso 4: Clasificación de sentimiento ---
def clasificar_sentimiento(score):
    if score is None:
        return "sin sentimiento"
    elif score >= 0.05:
        return "positivo"
    elif score <= -0.05:
        return "negativo"
    else:
        return "neutral"

df["sentiment"] = df["sentiment_score"].apply(clasificar_sentimiento)

# --- Paso 5: Verificación de la cantidad de comentarios procesados ---
print(f"Total de comentarios procesados: {len(df)}")
print(f"Comentarios no vacíos: {df['comment_clean'].dropna().shape[0]}")

# Guardar los resultados en un nuevo archivo
df.to_csv("../data/comments_analizados_con_sentimiento.csv", index=False)
print("Análisis completado. Resultados guardados en data/comments_analizados_con_sentimiento.csv")
