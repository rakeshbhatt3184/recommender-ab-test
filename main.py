from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import pickle
import random
import csv
from datetime import datetime

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and data...")
    ml_models['df'] = pd.read_csv('model_data.csv')
    ml_models['A_tfidf'] = pickle.load(open('model_A_similarity.pkl', 'rb'))
    ml_models['B_sentence'] = pickle.load(open('model_B_similarity.pkl', 'rb'))

    with open('log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0: 
            writer.writerow(['timestamp', 'article_id', 'model_version', 'latency_ms'])
    print("Startup complete.")
    yield
    print("Shutting down...")
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

def get_recommendations(article_idx, sim_matrix, df):
    sim_scores = list(enumerate(sim_matrix[article_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    article_indices = [i[0] for i in sim_scores]
    return df['id'].iloc[article_indices].tolist()

@app.get("/recommend/{article_id}")
def recommend(article_id: int):
    start_time = datetime.now()
    df = ml_models['df']
    
    try:
        article_idx = df[df['id'] == article_id].index[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="Article ID not found")

    if random.random() < 0.5:
        model_version = "A_tfidf"
        sim_matrix = ml_models[model_version]
    else:
        model_version = "B_sentence"
        sim_matrix = ml_models[model_version]

    recommendations = get_recommendations(article_idx, sim_matrix, df)
    
    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
    with open('log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), article_id, model_version, f"{latency_ms:.2f}"])
    
    return {
        "source_article_id": article_id,
        "model_version_used": model_version,
        "recommendations": recommendations
    }