from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("alcohol_detection_mobilenetv2_multi.h5")

# Class names - must match training order
class_names = ['Alcohol', 'Normal', 'Not_An_Eye']

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = preprocess_image(contents)
        preds = model.predict(img)[0]
        max_confidence = float(np.max(preds))
        predicted_index = int(np.argmax(preds))
        predicted_label = class_names[predicted_index]

        return JSONResponse(content={
            "predicted_class": predicted_label,
            "confidence": max_confidence
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
