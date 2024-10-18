from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import os
import pandas as pd
from data import data_preprocess

app = FastAPI()

Models = []
attacks = ["Backdoor", "Analysis", "Fuzzers", "Shellcode", "Reconnaissance", "Exploits", "DoS", "Generic", "Normal"]
models_dir = 'different models tsyp (biclassification)'

for attack in attacks:
    model_filename = os.path.join(models_dir, f"{attack}_model (1).h5")
    if os.path.exists(model_filename):
        model1 = tf.keras.models.load_model(model_filename)  # Load the entire model (architecture + weights)
        print(f"Model {model_filename} loaded successfully")
        Models.append((model1, attack))  # Store model with attack type for easier reference
    else:
        print(f"Model {model_filename} not found, skipping...")


def process_parquet(file: UploadFile):
    df = pd.read_parquet(file.file)

    X_scaled, y = data_preprocess(df)

    return X_scaled, y


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    X_scaled, y = process_parquet(file)

    all_predictions = {}

    for model, attack in Models:
        if attack == 'Normal':
            continue

        loss, accuracy = model.evaluate(X_scaled, y)
        all_predictions[attack] = {
            "Test Loss": loss,
            "Test Accuracy": accuracy
        }

    return {"attack_predictions": all_predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
