import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor

# Configurar MPS para GPU (si está disponible)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Cargar el modelo y el tokenizador
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Cargar los comentarios
df = pd.read_csv("../data/comments_reddit.csv")
df["comment"] = df["comment"].fillna("")
df["comment_clean"] = df["comment"].apply(lambda x: re.sub(r"http\S+|www.\S+", "", str(x).strip().lower()))

# Función de análisis de sentimiento
def analizar_sentimiento(texto):
    try:
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        scores = softmax(logits.cpu().numpy()[0])
        labels = ["negative", "neutral", "positive"]
        score_dict = dict(zip(labels, scores))
        return score_dict["positive"] - score_dict["negative"], labels[scores.argmax()]
    except Exception as e:
        print(f"Error en el comentario: {texto}. Error: {str(e)}")
        return None, "sin sentimiento"

# Función para procesar por lotes con multiproceso
def procesar_lote(comentarios):
    return [analizar_sentimiento(c) for c in comentarios]

# Procesar los comentarios en paralelo
batch_size = 50  # Puedes ajustar el tamaño del lote
comentarios = df["comment_clean"].dropna().tolist()

# Usar multiproceso para dividir el trabajo
with ThreadPoolExecutor(max_workers=2) as executor:
    resultados = list(executor.map(procesar_lote, [comentarios[i:i + batch_size] for i in range(0, len(comentarios), batch_size)]))

# Aplanar los resultados
resultados_a_planar = [item for sublist in resultados for item in sublist]

# Añadir los resultados al DataFrame
df["sentiment_score"], df["sentiment"] = zip(*resultados_a_planar)

# Guardar los resultados
df.to_csv("../data/comments_analizados_con_sentimiento_cardiffnlp.csv", index=False)
print(f"✅ Análisis completado. Comentarios procesados: {len(df)}")
