from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import pandas as pd

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faq_index.faiss")
with open("faq_data.pkl", "rb") as f:
    faq_data = pickle.load(f)

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'answer': 'No question received'}), 400

    # Convert input to embedding
    question_vec = model.encode([question])
    
    # Search nearest match
    D, I = index.search(np.array(question_vec), k=1)
    best_index = I[0][0]
    best_score = D[0][0]

    # You can define a threshold (optional)
    if best_score > 1.0:  # This threshold is distance-based (lower is better)
        return jsonify({'answer': 'Sorry, I could not find a matching answer.'})

    #return jsonify({'answer': faq_data[best_index]['answer']})
    return jsonify({'answer': faq_data['answers'][best_index]})

if __name__ == '__main__':
    app.run(debug=True)

