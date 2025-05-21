from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# CORS middleware (optional, helpful for frontend testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = "alcohol_detection_mobilenetv2_multi.h5"
model = load_model(MODEL_PATH)

# Constants
IMG_SIZE = 224
class_names = ['Alcohol', 'Normal', 'Not_An_Eye']
CONFIDENCE_THRESHOLD = 0.7

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)[0]
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))
        label = class_names[predicted_class]

        result = {
            "class": label,
            "confidence": round(confidence, 3),
            "is_valid": bool(confidence > CONFIDENCE_THRESHOLD)
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional: For local testing
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
