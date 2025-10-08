from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from utils import preprocess_image

app = FastAPI(title="COVID-19 Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models with custom configuration to handle batch_shape
model_resnet = tf.keras.models.load_model('./Models/model_resnet50.h5', compile=False)
model_vgg = tf.keras.models.load_model('./Models/model_vgg16.h5', compile=False)
model_xception = tf.keras.models.load_model('./Models/model_xception.h5', compile=False)

@app.get("/")
def read_root():
    return {"message": "Welcome to COVID-19 Detection API! Use /predict for inference."}

@app.get("/healthz")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            processed_img = preprocess_image(img)
            preds_resnet = model_resnet.predict(processed_img)
            preds_vgg = model_vgg.predict(processed_img)
            preds_xception = model_xception.predict(processed_img)
            ensemble_preds = (preds_resnet + preds_vgg + preds_xception) / 3
            pred_class = np.argmax(ensemble_preds, axis=1)[0]
            confidence = ensemble_preds[0][pred_class]
            result = "COVID" if pred_class == 0 else "non-COVID"
            results.append({
                "image": file.filename,
                "prediction": result,
                "confidence": float(confidence),
                "ensemble_contributions": {
                    "ResNet50": float(preds_resnet[0][pred_class]),
                    "VGG16": float(preds_vgg[0][pred_class]),
                    "Xception": float(preds_xception[0][pred_class]),
                },
                "final_ensemble_confidence": float(confidence)
            })
        return JSONResponse({"predictions": results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)