#######################################################################################
#    Nombre: Heimlich - Trainer IA                                                    #
#    Autor: Damian Kuczerawy                                                          #
#    Version: 1.0.0                                                                   #
#    Descripcion:                                                                     #
#        Carga el modelo IA entrenado en MoveNet y clasifica las imagenes en base al  #
#        modelo pre entrenado.                                                        #
#                                                                                     #
#######################################################################################
#    Autor        Fecha            Version        Descripcion                         #
#     KD         11/09/2025        1.0.0         Creación.                            #
#                                                                                     #
#######################################################################################
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import io
import os, joblib
from PIL import Image
from typing import List, Optional
from pydantic import BaseModel
import joblib  # O usa torch, tensorflow, etc. según tu modelo

# Índices
TORSO_IDXS = [5, 6, 11, 12]   # hombros y caderas
WRIST_IDXS = [9, 10]          # muñecas

app = FastAPI()

# Carga tu modelo preentrenado
BASE_DIR = os.path.dirname(__file__)   # carpeta src
model_path = os.path.join(BASE_DIR, "modelo_heimlich.pkl")
model = joblib.load(model_path)

class PredictIn(BaseModel):
    images: List[str]        # lista de imágenes en base64

_movenet = None

def _get_model():
    global MODEL
    if MODEL is None:
        if os.path.exists(LOCAL_MODEL_PATH):
            MODEL = tf.saved_model.load(LOCAL_MODEL_PATH)
        else:
            MODEL = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            # guarda una copia local para próximos arranques
            try:
                os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
                tf.saved_model.save(MODEL, LOCAL_MODEL_PATH)
            except Exception:
                pass
    return MODEL

def _movenet_keypoints(image_tf):
    """Devuelve (17,3) con (y,x,score) en [0,1]."""
    inp = tf.image.resize_with_pad(tf.expand_dims(image_tf, axis=0), 256, 256)
    inp = tf.cast(inp, dtype=tf.int32)
    out = movenet.signatures['serving_default'](inp)
    return out['output_0'][0, 0, :, :].numpy()

def _has_hands_and_torso(kps, min_score=0.30):
    scores = kps[:, 2]
    torso_ok = all(scores[i] >= min_score for i in TORSO_IDXS)
    hand_ok  = any(scores[i] >= min_score for i in WRIST_IDXS)  # al menos una muñeca
    return torso_ok and hand_ok
    
def _load_movenet():
    global _movenet
    _movenet = _get_model()
    return _movenet

def _decode_base64_to_pil(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64, validate=True)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen base64 inválida: {e}")


def _detect_pose(image, modelo, tau=0.40, min_score=0.30, default_class=0):
    """
    image: tensor TF (H,W,3) uint8
    modelo: clasificador scikit-learn con predict_proba
    tau: umbral de confianza del clasificador
    min_score: umbral de confianza de keypoints
    default_class: clase a devolver si falla el filtro o la confianza es baja
    Devuelve (pred, proba, razon)
    """
    # 1) Keypoints (17,3)
    kps = _movenet_keypoints(image)

    # 2) Clasificar (aplanar luego del filtro)
    vector = kps.flatten()
    proba = modelo.predict_proba([vector])[0]   # p(c|x)
    c_hat = int(np.argmax(proba))

    # 3) Filtro previo: requiere torso + muñeca
    if not _has_hands_and_torso(kps, min_score=min_score):
        return default_class, proba, "sin_manos_o_sin_torso"
    # 4) Rechazo por baja confianza
    if float(np.max(proba)) < float(tau):
        return default_class, proba, f"baja_confianza({np.max(proba):.2f}<{tau})"

    return c_hat, proba, "ok"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictIn):
    results = []
    cant = 0 
    cant_ok = 0
    cant_mal = 0
    puntaje_ok = 0
    puntaje_mal = 0

    
    for idx, img_b64 in enumerate(payload.images):
        try:
            # Cuento la cant de imagenes
            cant += 1
            
            # 1) Decodificar imagen
            img = _decode_base64_to_pil(payload.image_b64)

            # 2) Extraer keypoints con MoveNet
            c_hat, pred, res = _detect_pose(img,model)

            # cuento la cantidad de imagenes correctas
            if c_hat == 2:
                cant_ok += 1 
                puntaje_ok += max(res)

            # cuento la cantidad de imagenes incorrectas
            else:
                cant_mal += 1                
                puntaje_mal += max(res)

            # obtengo promedio de puntajes
            if cant_ok > 0:
                puntaje_ok = puntaje_ok/cant_ok
            if cant_mal > 0:       
                puntaje_mal = puntaje_mal/puntaje_mal

            # valido maniobra en base al puntaje obtenido
            if puntaje_ok > puntaje_mal:
                prediction = "correcta"
        
            else:
                prediction = "incorrecta"
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=e)

    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
