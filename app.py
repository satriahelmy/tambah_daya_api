from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import joblib
import pandas as pd

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Membuat aplikasi FastAPI
app = FastAPI()

class InputData(BaseModel):
    pemkwh: float
    rpptl: float
    jam_nyala: float
    daya: float
    keterangan_tarif: str
    kategori_layanan: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API"}

@app.post("/predict")
def predict(input_data: InputData):
    # Mengambil data input dan melakukan prediksi
    pemkwh = input_data.pemkwh
    rpptl = input_data.rpptl
    jam_nyala = input_data.jam_nyala
    daya = input_data.daya
    keterangan_tarif = input_data.keterangan_tarif
    kategori_layanan = input_data.kategori_layanan

    input_num = {
        "pemkwh": pemkwh,
        "jam_nyala": jam_nyala,
        "rpptl": rpptl,
        "daya": daya
    }

    df_numeric = pd.DataFrame([input_num])
    scaled_array = scaler.transform(df_numeric)
    df_scaled = pd.DataFrame(scaled_array, columns=df_numeric.columns)

    input_kat = {
        "keterangan_tarif": keterangan_tarif,
        'kategori_layanan': kategori_layanan
    }

    df_kat = pd.DataFrame([input_kat])
    encoded_array = encoder.transform(df_kat).toarray()

    df_encoded = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(df_kat.columns)
    )

    df_encoded.columns = df_encoded.columns.str.replace("keterangan_", "", regex=False)
    df_encoded.columns = df_encoded.columns.str.replace("kategori_", "", regex=False)

    df_final = pd.concat([df_scaled, df_encoded], axis=1)

    # Lakukan prediksi
    prediction = model.predict(df_final)

    # Mengkonversi hasil prediksi menjadi tipe data yang kompatibel dengan JSON
    prediction_value = int(prediction[0])

    res = ""
    if prediction_value == 0:
        res = "Tidak berpotensi tambah daya"
    else:
        res = "Berpotensi tambah daya"

    # Mengembalikan hasil prediksi
    return {"prediction": prediction_value, "result": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)