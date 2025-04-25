import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# 1. Load dữ liệu từ Hugging Face
DATASET_NAME = "cats_vs_dogs"
datasets = load_dataset (DATASET_NAME)
datasets

# 2. Định nghĩa biến đổi ảnh
TEST_SIZE = 0.2
datasets = datasets ['train' ].train_test_split (test_size=TEST_SIZE)

# 3. Custom Dataset class
IMG_SIZE = 64
img_transforms = transforms. Compose ( [
transforms.Resize((IMG_SIZE, IMG_SIZE) ),
transforms.Grayscale(num_output_channels=3),
transforms.ToTensor (),
transforms.Normalize (
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])])


# 4. Mô hình ResNet18 tùy chỉnh
class CatDogDataset (Dataset) :
    def __init__(self, data, transform=None) :
        self.data = data
        self.transform = transform


    def __len__(self) :
        return len(self.data)

    def __getitem__(self, idx) :
        image = self.data[idx] ['image' ]
        label = self.data[idx] ['labels']

        if self.transform:
            image = self.transform(image)

        label = torch.tensor (label, dtype=torch.long)

        return image, label

# Batch size constants
TRAIN_BATCH_SIZE = 512
VAL_BATCH_SIZE = 256
# Dataset initialization
train_dataset = CatDogDataset (datasets['train' ], transform=img_transforms)
test_dataset = CatDogDataset (datasets ['test' ], transform=img_transforms)

# Data loaders
train_loader = DataLoader (
    train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
)
test_loader = DataLoader (
    test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False
)

class CatDogModel (nn.Module) :
    def __init__(self, n_classes) :
        super(CatDogModel, self).__init__()


        # Load pre-trained ResNet-18 modelresnet_model = resnet18(weights='IMAGENET1K_V1')
        resnet_model = resnet18(weights='IMAGENET1K_V1')
        # Use all layers except the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet_model.children () ) [ :- 1])

        # Freeze the backbone parameters
        for param in self.backbone.parameters () :
            param. requires_grad = False

        # Replace the final fully connected layer
        in_features = resnet_model.fc.in_features
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x) :
        x = self.backbone (x) # Extract features using backbone
        x = torch. flatten (x, 1) # Flatten the features
        x = self.fc(x) # Classify using the final layer
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 2
model = CatDogModel (N_CLASSES) .to (device)

test_input = torch.rand(1, 3, 224, 224).to (device)

with torch.no_grad ():
    output = model (test_input)

print (output.shape)

# Hyperparameters
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Optimizer and loss function
optimizer = torch. optim. Adam (model.parameters (), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn. CrossEntropyLoss ()
for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_loss = sum(train_losses) / len(train_losses)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# ✅ Sau khi train xong, lưu mô hình
torch.save(model.state_dict(), "e:/python pr/cat_dog_model.pth")
print("✅ Mô hình đã được lưu vào e:/python pr/cat_dog_model.pth")

