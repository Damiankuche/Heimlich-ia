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
import cv2
from PIL import Image
from typing import List
from pydantic import BaseModel
import mediapipe as mp
from src.features_hands import concat_bihand

# Índices
TORSO_IDXS = [5, 6, 11, 12]   # hombros y caderas
WRIST_IDXS = [9, 10]          # muñecas

app = FastAPI()

 
idx_correcta = 2         # categoría correcta
tau_frame = 0.5          # confianza mínima por frame
min_ratio_ok = 0.5       # al menos 60% de frames buenos

MOVENET_PATH = None

# Carga tu modelo preentrenado (clasificador scikit-learn)
BASE_DIR = os.path.dirname(__file__)   # carpeta src
MODELS_DIR    = os.path.join(BASE_DIR, "models")
MOVENET_PATH = os.path.join(MODELS_DIR, "movenet_saved")
model_path = os.path.join(MODELS_DIR, "modelo_heimlich.pkl")
MODEL_HANDS_PATH    = os.path.join(MODELS_DIR, "hands_clf.joblib")  # clasificador de manos

# ------------- Cargar modelos -----------
model_hands = joblib.load(MODEL_HANDS_PATH)
model_IA = joblib.load(model_path)

class PredictIn(BaseModel):
    images: List[str]        # lista de imágenes en base64
    
class PredictOne(BaseModel):
    image: str

# -------- MoveNet --------
MODEL = None

_MP_HANDS = None
_MP = None

def _get_model():
    global MODEL
    if MODEL is None:
        if os.path.exists(MOVENET_PATH):
            MODEL = tf.saved_model.load(MOVENET_PATH)
            print("Levantando MoveNet de forma local")
        else:
            MODEL = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            print("Levantando MoveNet de la web")
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

def _get_mp_hands():
    """Devuelve (mp, hands) inicializados una sola vez (seguro con --reload)."""
    global _MP_HANDS
    if _MP_HANDS is None:
        _MP_HANDS = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6
        )
    return _MP_HANDS

def _extract_two_hands(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_hands = _get_mp_hands()
    res = mp_hands.process(rgb)
    left = right = None
    if res.multi_hand_landmarks and res.multi_handedness:
        for handedness, hlms in zip(res.multi_handedness, res.multi_hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hlms.landmark], dtype=np.float32)
            if handedness.classification[0].label == "Left":
                left = pts
            else:
                right = pts
    return left, right

def build_feats_top_bottom(left_pts, right_pts) -> np.ndarray:
    feats = concat_bihand(left_pts, right_pts)   # sin use_z
    return np.asarray(feats, np.float32).reshape(-1)

def _order_top_bottom(left_pts, right_pts):
    """
    Devuelve (top_pts, bottom_pts) usando la coordenada y de la muñeca (landmark 0).
    Menor y => más arriba (MediaPipe da coords normalizadas [0,1]).
    Si falta una mano, duplica la otra para mantener dimensión.
    """
    if left_pts is None and right_pts is None:
        return None, None
    if left_pts is None:
        return right_pts, right_pts
    if right_pts is None:
        return left_pts, left_pts

    # y del “wrist”
    yL = left_pts[0, 1]
    yR = right_pts[0, 1]
    return (left_pts, right_pts) if yL < yR else (right_pts, left_pts)

def _p_hands(img_pil: Image.Image) -> float:
    """Devuelve probabilidad de 'correcta' usando MediaPipe Hands + clasificador."""
    bgr = cv2.cvtColor(np.array(img_pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    left_pts, right_pts = _extract_two_hands(bgr)

    # Orden invariante al lado: top/bottom
    top_pts, bot_pts = _order_top_bottom(left_pts, right_pts)

    if top_pts is None or bot_pts is None:
        # sin manos detectadas, prob mínima o la política que prefieras
        return 0.0

    # Extraer features después de ordenar
    feats = build_feats_top_bottom(top_pts,bot_pts)

    # Clasificador binario: columna 1 = 'correcta'
    proba = model_hands.predict_proba(feats[None, :])[0, 1]
    return float(proba)
 
def _detect_pose(image_tf, modelo, tau=0.30, min_score=0.30, default_class=0):
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
        raise HTTPException(status_code=400, detail="La lista 'images' está vacía.")

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
            c_hat, proba, reason = _detect_pose(img_tf, model_IA)

            conf_Hands = _p_hands(img_pil)

            conf = float(np.max(proba))
            if c_hat == idx_correcta and conf >= tau_frame and conf_Hands >= tau_frame:
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
        
        
@app.post("/predictOne")
def predictOne(payload: PredictOne):
    if not payload.image:
        raise HTTPException(status_code=400, detail="No se recibió ninguna imagen")

    try:
        # 1) Decodificar imagen
        img_pil = _decode_base64_to_pil(payload.image)
        img_np = np.array(img_pil, dtype=np.uint8)
        img_tf = tf.convert_to_tensor(img_np, dtype=tf.uint8)

        # conf_Hands = _p_hands(img_pil)
        conf_Hands = 1
        # 2) Extraer keypoints + clasificar
        c_hat, proba, reason = _detect_pose(img_tf, model_IA)

        # conf = float(np.max(proba))
        conf = float(proba[idx_correcta])
        if c_hat == idx_correcta and conf >= tau_frame and conf_Hands >= tau_frame:
            prediction = "correcta"
        else:
            prediction = "incorrecta"

        print("prediction: ",prediction,"average: ",conf,"average_hands: ",conf_Hands,"c_hat: ",c_hat)

        return JSONResponse(content={
            "prediction": prediction,
            "average": conf,
            "average_hands": conf_Hands,
            "c_hat": c_hat
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {repr(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
