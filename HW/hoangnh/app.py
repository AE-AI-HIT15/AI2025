from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Tạo app FastAPI
app = FastAPI()

# Load model CNN đã train
model = load_model("save_model/lenet_model.h5")

# Hàm tiền xử lý ảnh cho model
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    img = img.resize((150, 150))  # Resize đúng kích thước input model của bạn
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    return img_array

# Route dự đoán
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_array = preprocess_image(img_bytes)
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    return {"prediction": predicted_class}
