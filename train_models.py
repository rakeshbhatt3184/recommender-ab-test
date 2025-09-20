import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("Loading processed data...")
df = pd.read_csv('processed_articles.csv')

print("Building Model A (TF-IDF)...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['content'])
cosine_sim_A = cosine_similarity(tfidf_matrix, tfidf_matrix)

pickle.dump(cosine_sim_A, open('model_A_similarity.pkl', 'wb'))
df.to_csv('model_data.csv', index=False)
print("Model A saved successfully.")

print("\nBuilding Model B (Sentence-Transformers)...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

print("Encoding articles. This may take a moment...")
embeddings = model.encode(df['content'].tolist(), show_progress_bar=True)
cosine_sim_B = cosine_similarity(embeddings)

pickle.dump(cosine_sim_B, open('model_B_similarity.pkl', 'wb'))
print("Model B saved successfully.")