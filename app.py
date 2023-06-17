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

    recom = recommend(nama)
    recom_data = [
        {'id_recom':'1', 'id_food':str(id), 'nama':nama, 'energi':str(energi), 'protein':str(protein), 'lemak':str(lemak), 'karbohidrat':str(karbohidrat), 'gambar':gambar},
        {'id_recom':'2', 'id_food':str(recom['Id'].values[0]), 'nama':recom['Nama Pangan'].values[0], 'energi':str(recom['Energi'].values[0]), 'protein':str(recom['Protein'].values[0]), 'lemak':str(recom['Lemak'].values[0]), 'karbohidrat':str(recom['Karbohidrat'].values[0]), 'gambar':recom['Gambar'].values[0]},
        {'id_recom':'3', 'id_food':str(recom['Id'].values[1]), 'nama':recom['Nama Pangan'].values[1], 'energi':str(recom['Energi'].values[1]), 'protein':str(recom['Protein'].values[1]), 'lemak':str(recom['Lemak'].values[1]), 'karbohidrat':str(recom['Karbohidrat'].values[1]), 'gambar':recom['Gambar'].values[1]},
        {'id_recom':'4', 'id_food':str(recom['Id'].values[2]), 'nama':recom['Nama Pangan'].values[2], 'energi':str(recom['Energi'].values[2]), 'protein':str(recom['Protein'].values[2]), 'lemak':str(recom['Lemak'].values[2]), 'karbohidrat':str(recom['Karbohidrat'].values[2]), 'gambar':recom['Gambar'].values[2]},
        {'id_recom':'5', 'id_food':str(recom['Id'].values[3]), 'nama':recom['Nama Pangan'].values[3], 'energi':str(recom['Energi'].values[3]), 'protein':str(recom['Protein'].values[3]), 'lemak':str(recom['Lemak'].values[3]), 'karbohidrat':str(recom['Karbohidrat'].values[3]), 'gambar':recom['Gambar'].values[3]},
        {'id_recom':'6', 'id_food':str(recom['Id'].values[4]), 'nama':recom['Nama Pangan'].values[4], 'energi':str(recom['Energi'].values[4]), 'protein':str(recom['Protein'].values[4]), 'lemak':str(recom['Lemak'].values[4]), 'karbohidrat':str(recom['Karbohidrat'].values[4]), 'gambar':recom['Gambar'].values[4]}
    ]

    return jsonify({'data':recom_data})

def recommend(nama, n=5, columns=['Id', 'Nama Pangan', 'Energi', 'Protein', 'Lemak', 'Karbohidrat', 'Gambar']):
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

    recom = advrecommend(nama)

    recom_data = [
        {'id_recom':'1', 'id_food':str(id), 'nama':nama, 'energi':str(energi), 'protein':str(protein), 'lemak':str(lemak), 'karbohidrat':str(karbohidrat), 'gambar':gambar},
        {'id_recom':'2', 'id_food':str(recom['Id'].values[0]), 'nama':recom['Nama Pangan'].values[0], 'energi':str(recom['Energi'].values[0]), 'protein':str(recom['Protein'].values[0]), 'lemak':str(recom['Lemak'].values[0]), 'karbohidrat':str(recom['Karbohidrat'].values[0]), 'gambar':recom['Gambar'].values[0]},
        {'id_recom':'3', 'id_food':str(recom['Id'].values[1]), 'nama':recom['Nama Pangan'].values[1], 'energi':str(recom['Energi'].values[1]), 'protein':str(recom['Protein'].values[1]), 'lemak':str(recom['Lemak'].values[1]), 'karbohidrat':str(recom['Karbohidrat'].values[1]), 'gambar':recom['Gambar'].values[1]},
        {'id_recom':'4', 'id_food':str(recom['Id'].values[2]), 'nama':recom['Nama Pangan'].values[2], 'energi':str(recom['Energi'].values[2]), 'protein':str(recom['Protein'].values[2]), 'lemak':str(recom['Lemak'].values[2]), 'karbohidrat':str(recom['Karbohidrat'].values[2]), 'gambar':recom['Gambar'].values[2]},
        {'id_recom':'5', 'id_food':str(recom['Id'].values[3]), 'nama':recom['Nama Pangan'].values[3], 'energi':str(recom['Energi'].values[3]), 'protein':str(recom['Protein'].values[3]), 'lemak':str(recom['Lemak'].values[3]), 'karbohidrat':str(recom['Karbohidrat'].values[3]), 'gambar':recom['Gambar'].values[3]},
        {'id_recom':'6', 'id_food':str(recom['Id'].values[4]), 'nama':recom['Nama Pangan'].values[4], 'energi':str(recom['Energi'].values[4]), 'protein':str(recom['Protein'].values[4]), 'lemak':str(recom['Lemak'].values[4]), 'karbohidrat':str(recom['Karbohidrat'].values[4]), 'gambar':recom['Gambar'].values[4]}
    ]

    return jsonify({'data':recom_data})

def advrecommend(nama, n=5, columns=['Id', 'Nama Pangan', 'Energi', 'Protein', 'Lemak', 'Karbohidrat', 'Gambar']):
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
