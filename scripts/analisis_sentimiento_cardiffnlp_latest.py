import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Configurar el dispositivo: usar MPS si está disponible
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Usar el modelo más actualizado de CardiffNLP
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Cargar los comentarios
df = pd.read_csv("../data/comments_reddit.csv")
df["comment"] = df["comment"].fillna("")
df["comment_clean"] = df["comment"].apply(lambda x: re.sub(r"http\S+|www\.\S+", "", str(x).strip()))

# Análisis de sentimiento con clasificación basada en el score
def analizar_sentimiento(texto):
    try:
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        scores = softmax(logits.cpu().numpy()[0])
        labels = ["negative", "neutral", "positive"]
        score_dict = dict(zip(labels, scores))

        # Clasificación basada en el score más alto
        max_score = max(scores)
        if max_score == scores[0]:  # Negative
            sentiment = "negative"
        elif max_score == scores[1]:  # Neutral
            sentiment = "neutral"
        else:  # Positive
            sentiment = "positive"

        return max_score, sentiment
    except Exception as e:
        return None, "sin sentimiento"

# Aplicar a cada comentario
df_valid = df["comment_clean"].dropna()
df_resultados = df_valid.apply(lambda x: pd.Series(analizar_sentimiento(x)))
df["sentiment_score"] = df_resultados[0]
df["sentiment"] = df_resultados[1]

# Guardar los resultados
df.to_csv("../data/comments_analizados_con_sentimiento_latest.csv", index=False)
print(f"✅ Análisis completado. Comentarios procesados: {len(df)}")
