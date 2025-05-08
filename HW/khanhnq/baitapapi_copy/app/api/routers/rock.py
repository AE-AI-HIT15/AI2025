import sys
sys.path.append(r'D:\README\baitapapi\app\models')
sys.path.append(r'D:\README\baitapapi\app\shared')
from rock_lenet import lenet_rock_model
from rock_vgg import vgg_rock_model  
import numpy as np
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import io
from config import IMG_SIZE_LENET_ROCK
from PIL import Image
from tensorflow.keras.models import load_model
import json
import asyncio
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

rock_router = APIRouter()

# lenet_rock_model = load_model ('models/weights/lenet_rock.weights.h5') 

def preprocess_image(file: UploadFile, target_size=(32, 32)) -> np.ndarray:
    """
    Tiền xử lý ảnh để đưa vào mô hình.

    Args:
        file (UploadFile): Ảnh đầu vào từ request.
        target_size (tuple): Kích thước ảnh cần resize về, ví dụ (32, 32).

    Returns:
        np.ndarray: Ảnh đã được resize và chuẩn hóa, shape (1, target_size[0], target_size[1], 3).
    """
    try:
        # Đọc dữ liệu từ file và mở bằng PIL
        image = Image.open(io.BytesIO(file.file.read())).convert('RGB')
        # Resize ảnh
        image = image.resize(target_size)
        # Chuyển sang numpy array và scale về [0, 1]
        image_array = np.asarray(image) / 255.0
        # Thêm batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise ValueError(f"Lỗi tiền xử lý ảnh: {str(e)}")


@rock_router.post("/lenet/inference")
async def lenet_inference(file: UploadFile = File(...)):
    """
    Inference ảnh với mô hình LeNet.

    - **file**: Ảnh upload (dạng file, ví dụ: jpg, png).
    - **Trả về**: Nhãn dự đoán (kiểu số nguyên) hoặc thông báo lỗi.
    """
    try:
        img = preprocess_image(file, target_size=(150, 150))
        pred = lenet_rock_model.predict(img)
        result = int(pred[np.argmax(pred, axis=1)][0])
        crack_or_no = ''
        if result:
            crack_or_no = 'not crack'
        else:
            crack_or_no = 'crack'
        return JSONResponse(content={"prediction": crack_or_no})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
    pass


@rock_router.post("/vgg/inference")
async def vgg_inference(file: UploadFile = File(...)):
    """
    Inference ảnh với mô hình VGG.

    - **file**: Ảnh upload (dạng file, ví dụ: jpg, png).
    - **Trả về**: Nhãn dự đoán (kiểu số nguyên) hoặc thông báo lỗi.
    """
    try:
        img = preprocess_image(file, target_size=(150 , 150))
        pred = vgg_rock_model.predict(img)
        result = int(pred[np.argmax(pred, axis=1)][0])
        crack_or_no = ''
        if result:
            crack_or_no = 'not crack'
        else:
            crack_or_no = 'crack'
        return JSONResponse(content={"prediction": crack_or_no})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    pass


