import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask_cors import CORS

# Loading model
file = open('./model/nb_model_adv.pkl', 'rb')
model = joblib.load(file)

# Loading dataset
gizi = pd.read_csv('./dataset/gizi.csv')

output_col = ['id_food', 'nama', 'energi', 'protein', 'lemak', 'karbohidrat', 'gambar']
numeric_gizi = gizi[['energi', 'protein', 'lemak', 'karbohidrat']]

app = Flask(__name__)
CORS(app)

@app.route('/')
def message():
    return "Model deployed!"

@app.route('/advpredict', methods=['POST'])
def advpredict():
    energi = request.json['energi']
    protein = request.json['protein']
    lemak = request.json['lemak']
    karbohidrat = request.json['karbohidrat']
    float_features = [energi, protein, lemak, karbohidrat]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    data = gizi[gizi['nama'] == prediction.item()]
    predict_data = data[['energi', 'protein', 'lemak', 'karbohidrat']]

    similarity = cosine_similarity(predict_data, numeric_gizi)
    # Get top 10 similarity
    top10 = np.argsort(similarity, axis=1)[0][::-1][:10] 
    recommended = gizi.iloc[top10][output_col]
    dict_recom = recommended.to_dict('records')

    return jsonify({'data':dict_recom})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    