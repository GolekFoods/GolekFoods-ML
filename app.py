import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from flask_cors import CORS

# Loading model
file = open('./model/nb_model.pkl', 'rb')
file2 = open('./model/nb_model_adv.pkl', 'rb')
model = joblib.load(file)
model2 = joblib.load(file2)

# Loading dataset
gizi = pd.read_csv('./dataset/gizi.csv')

# Membuat Representasi vektor untuk sistem rekomendasi
tfidf_nama = TfidfVectorizer().fit_transform(gizi['Nama Pangan'])
tfidf_energi = gizi['Energi'].to_numpy().reshape(-1, 1)
tfidf_protein = gizi['Protein'].to_numpy().reshape(-1, 1)
tfidf_lemak = gizi['Lemak'].to_numpy().reshape(-1, 1)
tfidf_karbohidrat = gizi['Karbohidrat'].to_numpy().reshape(-1, 1)

# Menggabungkan semua vektor
item_vectors = pd.DataFrame(cosine_similarity(tfidf_nama, tfidf_nama), columns=gizi["Nama Pangan"]).mul(0.1)
item_vectors += pd.DataFrame(cosine_similarity(tfidf_energi, tfidf_energi), columns=gizi["Nama Pangan"]).mul(0.225)
item_vectors += pd.DataFrame(cosine_similarity(tfidf_protein, tfidf_protein), columns=gizi["Nama Pangan"]).mul(0.225)
item_vectors += pd.DataFrame(cosine_similarity(tfidf_lemak, tfidf_lemak), columns=gizi["Nama Pangan"]).mul(0.225)
item_vectors += pd.DataFrame(cosine_similarity(tfidf_karbohidrat, tfidf_karbohidrat), columns=gizi["Nama Pangan"]).mul(0.225)

app = Flask(__name__)
CORS(app)

@app.route('/')
def message():
    return "Model deployed!"


@app.route('/predict', methods=['POST'])
def predict():
    energi = request.json['energi']
    protein = request.json['protein']
    lemak = request.json['lemak']
    karbohidrat = request.json['karbohidrat']
    float_features = [energi, protein, lemak, karbohidrat]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    data = gizi[gizi['Nama Pangan'] == prediction.item()]
    id = data['Id'].values[0]
    nama = data['Nama Pangan'].values[0]
    energi = data['Energi'].values[0]
    protein = data['Protein'].values[0]
    lemak = data['Lemak'].values[0]
    karbohidrat = data['Karbohidrat'].values[0]
    gambar = data['Gambar'].values[0]

    recom_data = recommend(nama, 5, ['Id', 'Nama Pangan', 'Energi', 'Protein', 'Lemak', 'Karbohidrat', 'Gambar']).to_dict(orient='records')
    converted_recom = [{key: str(value) for key, value in item.items()} for item in recom_data]

    return jsonify({'id':str(id), 'nama':nama, 'energi':str(energi), 'protein':str(protein), 'lemak':str(lemak), 'karbohidrat':str(karbohidrat), 'gambar':gambar,
    'recom': converted_recom})

def recommend(nama, n=5, columns=None):
    idx = gizi[gizi["Nama Pangan"] == nama].index[0]
    sim_scores = list(enumerate(item_vectors.iloc[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    food_indices = [i[0] for i in sim_scores]
    if columns is None:
        return gizi.iloc[food_indices].reset_index(drop=True)
    else:
        return gizi[columns].iloc[food_indices].reset_index(drop=True)

@app.route('/advpredict', methods=['POST'])
def advpredict():
    energi = request.json['energi']
    protein = request.json['protein']
    lemak = request.json['lemak']
    karbohidrat = request.json['karbohidrat']
    float_features = [energi, protein, lemak, karbohidrat]
    features = [np.array(float_features)]
    prediction = model2.predict(features)

    data = gizi[gizi['Nama Pangan'] == prediction.item()]
    id = data['Id'].values[0]
    nama = data['Nama Pangan'].values[0]
    energi = data['Energi'].values[0]
    protein = data['Protein'].values[0]
    lemak = data['Lemak'].values[0]
    karbohidrat = data['Karbohidrat'].values[0]
    gambar = data['Gambar'].values[0]

    recom_data = advrecommend(nama, 5, ['Id', 'Nama Pangan', 'Energi', 'Protein', 'Lemak', 'Karbohidrat', 'Gambar']).to_dict(orient='records')
    converted_recom = [{key: str(value) for key, value in item.items()} for item in recom_data]

    return jsonify({'id':str(id),'nama':nama, 'energi':str(energi), 'protein':str(protein), 'lemak':str(lemak), 'karbohidrat':str(karbohidrat), 'gambar':gambar,
    'recom': converted_recom})

def advrecommend(nama, n=5, columns=None):
    idx = gizi[gizi["Nama Pangan"] == nama].index[0]
    sim_scores = list(enumerate(item_vectors.iloc[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    food_indices = [i[0] for i in sim_scores]
    if columns is None:
        return gizi.iloc[food_indices].reset_index(drop=True)
    else:
        return gizi[columns].iloc[food_indices].reset_index(drop=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
