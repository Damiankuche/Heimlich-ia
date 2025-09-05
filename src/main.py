from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
import os, joblib
from PIL import Image
import joblib  # O usa torch, tensorflow, etc. según tu modelo

app = FastAPI()

# Carga tu modelo preentrenado
BASE_DIR = os.path.dirname(__file__)   # carpeta src
model_path = os.path.join(BASE_DIR, "modelo_heimlich.pkl")
model = joblib.load(model_path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lee la imagen
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocesa la imagen según lo que espera tu modelo
    # Por ejemplo, resize, normalización, etc.
    # processed_image = preprocess(image)
    
    # Realiza la predicción
    # prediction = model.predict([processed_image])
    prediction = "heimlich_detected"  # Simulación, reemplaza por tu lógica real
    
    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
