from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO
import torch.nn as nn
from torchvision.models import resnet18
import os

# Tạo FastAPI app
app = FastAPI()

# Định nghĩa biến đổi ảnh
IMG_SIZE = 64
img_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Định nghĩa mô hình
class CatDogModel(nn.Module):
    def __init__(self, n_classes):
        super(CatDogModel, self).__init__()
        resnet_model = resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = resnet_model.fc.in_features
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Tải mô hình đã huấn luyện
model = CatDogModel(2)
model.load_state_dict(torch.load("E:\python pr\cat_dog_model.pth"))
model.eval()

# API để nhận ảnh và trả về kết quả phân loại
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ file tải lên
    image = Image.open(BytesIO(await file.read()))
    
    # Biến đổi ảnh
    image = img_transforms(image).unsqueeze(0)
    
    # Đưa ảnh vào thiết bị (CPU hoặc GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = image.to(device)
    model.to(device)
    
    # Dự đoán
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Phân loại kết quả
    label = 'Cat' if predicted.item() == 0 else 'Dog'
    
    return JSONResponse(content={"label": label})

