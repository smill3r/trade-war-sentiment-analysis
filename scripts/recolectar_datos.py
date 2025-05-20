import praw
import pandas as pd
from datetime import datetime

# --- Configuración de la API de Reddit ---
reddit = praw.Reddit(client_id='fI5xbkQ2URkpPTjh9k4tsw',
                     client_secret='wXr1fwrjPf81vqGZY6L4F9-_25yjeA',
                     user_agent='eirs_project')

# --- Parámetros ---
subreddits = ["politics", "worldnews", "economics"]
keywords = ["trade war", "tariff", "china", "economy"]
limit = 20  # Número de posts por palabra clave

data = []

for subreddit in subreddits:
    for keyword in keywords:
        print(f"🔍 Buscando posts en r/{subreddit} con palabra clave: {keyword}")
        posts = reddit.subreddit(subreddit).search(keyword, limit=limit)

        for post in posts:
            post.comments.replace_more(limit=0)
            for top_level_comment in post.comments:
                data.append({
                    "subreddit": subreddit,
                    "keyword": keyword,
                    "post_title": post.title,
                    "comment": top_level_comment.body,
                    "comment_score": top_level_comment.score,
                    "created_utc": datetime.fromtimestamp(top_level_comment.created_utc),
                    "num_comments": post.num_comments,
                    "url": post.url
                })

# --- Guardar en CSV ---
df = pd.DataFrame(data)
df.to_csv("../data/comments_reddit.csv", index=False)
print(f"✅ Recolectados {len(df)} comentarios principales. Guardado en data/comments_reddit.csv")
