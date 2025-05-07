
from models.leaf_lenet import LeNetLeafModel
#from models.rock_vgg import vgg_rock_model  

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from shared.utils.preprocessing import preprocess_image 
from shared.config import IMG_SIZE_LENET_LEAF 

rock_router = APIRouter()

@rock_router.post("/lenet/inference")
async def lenet_inference(file: UploadFile = File(...)):
    
    try:
        img = preprocess_image(file, IMG_SIZE_LENET_LEAF )
        model = LeNetLeafModel()
        pred = model.predict(img)
        #result = int(np.argmax(pred, axis=1)[0])
        return JSONResponse(content={"Phần trăm đó là leaf ": pred })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    pass

# @leaf_router.post("/vgg/inference")
# async def vgg_inference(file: UploadFile = File(...)):
#     """
#     Inference ảnh với mô hình VGG.

#     - **file**: Ảnh upload (dạng file, ví dụ: jpg, png).
#     - **Trả về**: Nhãn dự đoán (kiểu số nguyên) hoặc thông báo lỗi.
#     """
#     # try:
#     #     img = preprocess_image(file, target_size=(224, 224))
#     #     pred = vgg_rock_model.predict(img)
#     #     result = int(np.argmax(pred, axis=1)[0])
#     #     return JSONResponse(content={"prediction": result})
#     # except Exception as e:
#     #     return JSONResponse(content={"error": str(e)}, status_code=400)
#     pass