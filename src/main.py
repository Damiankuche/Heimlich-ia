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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import io, os, base64, joblib
from PIL import Image
from typing import List
from pydantic import BaseModel

# Índices
TORSO_IDXS = [5, 6, 11, 12]   # hombros y caderas
WRIST_IDXS = [9, 10]          # muñecas

app = FastAPI()


idx_correcta = 2         # categoría correcta
tau_frame = 0.6          # confianza mínima por frame
min_ratio_ok = 0.6       # al menos 60% de frames buenos


# Carga tu modelo preentrenado (clasificador scikit-learn)
BASE_DIR = os.path.dirname(__file__)   # carpeta src
model_path = os.path.join(BASE_DIR, "modelo_heimlich.pkl")
model = joblib.load(model_path)

class PredictIn(BaseModel):
    images: List[str]        # lista de imágenes en base64

# -------- MoveNet (cacheado) --------
MODEL = None
# Si quieres cache local, define una carpeta; si no, déjalo así y carga desde TF Hub
# LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "movenet_saved")

def _get_model():
    global MODEL
    if MODEL is None:
        # if os.path.exists(LOCAL_MODEL_PATH):
        #     MODEL = tf.saved_model.load(LOCAL_MODEL_PATH)
        # else:
        MODEL = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        #     try:
        #         os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        #         tf.saved_model.save(MODEL, LOCAL_MODEL_PATH)
        #     except Exception:
        #         pass
    return MODEL

def _movenet_keypoints(image_tf: tf.Tensor):
    """Devuelve (17,3) con (y,x,score) en [0,1]."""
    model_kp = _get_model()
    inp = tf.image.resize_with_pad(tf.expand_dims(image_tf, axis=0), 256, 256)
    inp = tf.cast(inp, dtype=tf.int32)
    out = model_kp.signatures['serving_default'](inp)
    return out['output_0'][0, 0, :, :].numpy()

def _has_hands_and_torso(kps, min_score=0.30):
    scores = kps[:, 2]
    torso_ok = all(scores[i] >= min_score for i in TORSO_IDXS)
    hand_ok  = any(scores[i] >= min_score for i in WRIST_IDXS)  # al menos una muñeca
    return torso_ok and hand_ok

def _decode_base64_to_pil(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen base64 inválida: {e}")

def _detect_pose(image_tf, modelo, tau=0.40, min_score=0.30, default_class=0):
    """
    image_tf: tensor TF (H,W,3) uint8
    modelo: clasificador scikit-learn con predict_proba
    Devuelve (pred, proba, razon)
    """
    kps = _movenet_keypoints(image_tf)
    vector = kps.flatten()
    proba = modelo.predict_proba([vector])[0]   # p(c|x)
    c_hat = int(np.argmax(proba))

    if not _has_hands_and_torso(kps, min_score=min_score):
        return default_class, proba, "sin_manos_o_sin_torso"
    if float(np.max(proba)) < float(tau):
        return default_class, proba, f"baja_confianza({np.max(proba):.2f}<{tau})"

    return c_hat, proba, "ok"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictIn):
    if not payload.images:
        rraise HTTPException(status_code=400, detail="La lista 'images' está vacía.")

    cant = 0 
    cant_ok = 0
    cant_mal = 0
    suma_ok = 0.0
    suma_mal = 0.0
    ratio_ok = 0
    prediction = "incorrecta"

    try:
        for idx, img_b64 in enumerate(payload.images):
            cant += 1
            # 1) Decodificar imagen
            img_pil = _decode_base64_to_pil(img_b64)
            img_np = np.array(img_pil, dtype=np.uint8)
            img_tf = tf.convert_to_tensor(img_np, dtype=tf.uint8)

            # 2) Extraer keypoints + clasificar
            c_hat, proba, reason = _detect_pose(img_tf, model)

            conf = float(np.max(proba))
            if c_hat == idx_correcta and conf >= tau_frame:
                cant_ok += 1 
                suma_ok += conf
            else:
                cant_mal += 1                
                suma_mal += conf

        total = cant_ok + cant_mal
        ratio_ok = cant_ok / (total) if total > 0 else 0.0

        # Promedios
        prom_ok = (suma_ok / cant_ok) if cant_ok > 0 else 0.0
        prom_mal = (suma_mal / cant_mal) if cant_mal > 0 else 0.0

        prediction = "correcta" if prom_ok > prom_mal and ratio_ok >= min_ratio_ok else "incorrecta"

        return JSONResponse(content={
            "prediction": prediction,
            "cant": cant,
            "ok": {"count": cant_ok, "avg_conf": round(prom_ok, 3)},
            "mal": {"count": cant_mal, "avg_conf": round(prom_mal, 3)}
        })
    except HTTPException:
        raise
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error interno: {repr(e)}")
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
