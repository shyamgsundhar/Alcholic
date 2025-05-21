from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the model
# ✅ Correct path (based on your filename)
model = load_model("alcohol_detection_mobilenetv2_multi.h5")
# Constants
IMG_SIZE = 224
CLASS_NAMES = ['Alcohol', 'Normal', 'Not_An_Eye']
CONFIDENCE_THRESHOLD = 0.7

# Utility to preprocess and predict
def predict(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    max_conf = np.max(prediction)
    predicted_class = np.argmax(prediction)

    return {
        "class": CLASS_NAMES[predicted_class],
        "confidence": float(max_conf),
        "is_valid": max_conf > CONFIDENCE_THRESHOLD
    }

# API Route
@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = predict(img_bytes)
    return JSONResponse(content=result)
