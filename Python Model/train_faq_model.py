import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Step 1: CSV load karo
df = pd.read_csv('faq.csv')  # Apna CSV file path yahan dena

# Step 2: Model load karo (lightweight model)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Questions ki embeddings banayein
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

print("Generating embeddings...")
question_embeddings = model.encode(questions, convert_to_numpy=True)

# Step 4: FAISS index banayein
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Step 5: Sab data ko ek file mein save karo
faq_data = {
    'questions': questions,
    'answers': answers,
    'embeddings': question_embeddings,
    'faiss_index': index
}

# Note: FAISS index ko pickle nahi kar sakte, isliye index ko alag save karte hain:
faiss.write_index(index, 'faq_index.faiss')

with open('faq_data.pkl', 'wb') as f:
    # Hum questions aur answers save karenge, embeddings already FAISS mein hain
    pickle.dump({'questions': questions, 'answers': answers}, f)

print("Training complete. Files saved: faq_index.faiss and faq_data.pkl")
