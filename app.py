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

# Membuat aplikasi FastAPI
app = FastAPI()

class InputData(BaseModel):
    pemkwh1: float
    pemkwh2: float
    pemkwh3: float
    pemkwh4: float
    pemkwh5: float
    pemkwh6: float
    pemkwh7: float
    pemkwh8: float
    pemkwh9: float
    pemkwh10: float
    pemkwh11: float
    pemkwh12: float
    pemkwh13: float
    pemkwh14: float
    pemkwh15: float
    pemkwh16: float
    pemkwh17: float
    pemkwh18: float
    pemkwh19: float
    pemkwh20: float
    pemkwh21: float
    pemkwh22: float
    pemkwh23: float
    pemkwh24: float




@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API"}

@app.post("/predict")
def predict(input_data: InputData):
    # Mengambil data input dan melakukan prediksi
    pemkwh1 = input_data.pemkwh1
    pemkwh2 = input_data.pemkwh2
    pemkwh3 = input_data.pemkwh3
    pemkwh4 = input_data.pemkwh4
    pemkwh5 = input_data.pemkwh5
    pemkwh6 = input_data.pemkwh6
    pemkwh7 = input_data.pemkwh7
    pemkwh8 = input_data.pemkwh8
    pemkwh9 = input_data.pemkwh9
    pemkwh10 = input_data.pemkwh10
    pemkwh11 = input_data.pemkwh11
    pemkwh12 = input_data.pemkwh12
    pemkwh13 = input_data.pemkwh13
    pemkwh14 = input_data.pemkwh14
    pemkwh15 = input_data.pemkwh15
    pemkwh16 = input_data.pemkwh16
    pemkwh17 = input_data.pemkwh17
    pemkwh18 = input_data.pemkwh18
    pemkwh19 = input_data.pemkwh19
    pemkwh20 = input_data.pemkwh20
    pemkwh21 = input_data.pemkwh21
    pemkwh22 = input_data.pemkwh22
    pemkwh23 = input_data.pemkwh23
    pemkwh24 = input_data.pemkwh24


    input_num = {
        "pemkwh1": pemkwh1,
        "pemkwh2": pemkwh2,
        "pemkwh3": pemkwh3,
        "pemkwh4": pemkwh4,
        "pemkwh5": pemkwh5,
        "pemkwh6": pemkwh6,
        "pemkwh7": pemkwh7,
        "pemkwh8": pemkwh8,
        "pemkwh9": pemkwh9,
        "pemkwh10": pemkwh10,
        "pemkwh11": pemkwh11,
        "pemkwh12": pemkwh12,
        "pemkwh13": pemkwh13,
        "pemkwh14": pemkwh14,
        "pemkwh15": pemkwh15,
        "pemkwh16": pemkwh16,
        "pemkwh17": pemkwh17,
        "pemkwh18": pemkwh18,
        "pemkwh19": pemkwh19,
        "pemkwh20": pemkwh20,
        "pemkwh21": pemkwh21,
        "pemkwh22": pemkwh22,
        "pemkwh23": pemkwh23,
        "pemkwh24": pemkwh24,
    }

    df_numeric = pd.DataFrame([input_num])
    scaled_array = scaler.transform(df_numeric)
    df_scaled = pd.DataFrame(scaled_array, columns=df_numeric.columns)

    df_final = df_scaled

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
    uvicorn.run(app, host="0.0.0.0", port=8000)