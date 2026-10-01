"""REST API for the nuclei segmentation model.

Run:   uvicorn serve:app --port 8000
Call:  curl -F "file=@examples/fluorescent.png" http://127.0.0.1:8000/segment

Returns the nucleus count and the mask as a base64 PNG (255 = nucleus).
Uses the ONNX model and threshold written by the notebook.
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile

from inference import predict_prob_onnx, split_nuclei

MODELS = Path(__file__).parent / "models"
THRESHOLD = json.loads((MODELS / "config.json").read_text())["threshold"]
session = ort.InferenceSession(str(MODELS / "nuclei_unet.onnx"), providers=["CPUExecutionProvider"])

app = FastAPI(title="Nuclei segmentation")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/segment")
def segment(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read the file as an image.")
    rgb = image[..., ::-1]
    nuclei, count = split_nuclei(predict_prob_onnx(session, rgb) > THRESHOLD)
    _, png = cv2.imencode(".png", (nuclei > 0).astype(np.uint8) * 255)
    return {
        "count": count,
        "height": int(rgb.shape[0]),
        "width": int(rgb.shape[1]),
        "mask_png_base64": base64.b64encode(png.tobytes()).decode(),
    }
