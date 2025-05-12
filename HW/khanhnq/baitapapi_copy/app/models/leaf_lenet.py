import tensorflow as tf
import numpy as np
import sys
sys.path.append(r'D:\README\baitapapi\app\shared')
from config import IMG_SIZE_LENET_ROCK 
from tensorflow.keras.models import load_model 


class LeNetLeafModel:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(IMG_SIZE_LENET_ROCK, IMG_SIZE_LENET_ROCK, 3)),
            tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
            tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=5, activation='softmax')
        ])
        print(self.model.summary())
    # def predict(self, img: np.ndarray): 
        # # weight = keras.lenet_rock_model.load_model ()
        # """
        # Dự đoán nhãn cho ảnh đầu vào.
        # img: numpy array shape (1, IMG_SIZE_LENET_ROCK, IMG_SIZE_LENET_ROCK, 3)
        # """
        
        # self.model.load_weights(r"D:\README\baitapapi\app\models\weights\lenet_cassava.weights.h5")
        # preds = self.model.predict(img)
        # # Nếu là binary classification, trả về 0 hoặc 1
        # return (preds > 0.5).astype(int) 
    def predict(self, img: np.ndarray):
    # Dự đoán nhãn cho ảnh đầu vào.
    # img: numpy array shape (1, IMG_SIZE_LENET_ROCK, IMG_SIZE_LENET_ROCK, 3)
    
    # Trả về:
    #   - class_idx: index của lớp dự đoán (0–4)
    #   - confidence: xác suất tương ứng của lớp đó
    #   - probs: mảng softmax probabilities shape (5,)
    
    # Load trọng số
        self.model.load_weights(
            r"D:\README\baitapapi\app\models\weights\lenet_cassava.weights.h5"
    )

    # preds là output của softmax, shape (1, 5)
        preds = self.model.predict(img)
    
    # Lấy về mảng xác suất của sample đầu tiên
        probs = preds[0]  # ví dụ [0.10, 0.05, 0.70, 0.10, 0.05]
    
    # Lớp dự đoán: index của xác suất lớn nhất
        class_idx = int(np.argmax(probs))
    
    # Độ “tự tin” = xác suất của lớp đó
        confidence = float(probs[class_idx])
    
        return class_idx


lenet_leaf_model = LeNetLeafModel()