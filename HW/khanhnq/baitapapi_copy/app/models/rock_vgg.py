import tensorflow as tf
import numpy as np
import sys
# sys.path.append(r'D:\README\baitapapi\app\shared')
# from config import IMG_SIZE_LENET_ROCK 
from tensorflow.keras.models import load_model 

class VggRockModel :
    def __init__(self):
        self.model = tf.keras.models.Sequential([
    # 1st Conv Block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           input_shape=(150, 150, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    # 2nd Conv Block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    # 3rd Conv Block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    # 4th Conv Block
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    # 5th Conv Block
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    # Fully Connected Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # use softmax + units=N for multi-class
        ])
        print(self.model.summary())

    def predict(self, img: np.ndarray): #TODO: Unknown logic
        # weight = keras.lenet_rock_model.load_model ()
        """
        Dự đoán nhãn cho ảnh đầu vào.
        img: numpy array shape (1, IMG_SIZE_LENET_ROCK, IMG_SIZE_LENET_ROCK, 3)
        """
        
        self.model.load_weights(r"D:\README\baitapapi\app\models\weights\base_vgg16_concrete_crack.weights.h5")
        preds = self.model.predict(img)
        # Nếu là binary classification, trả về 0 hoặc 1
        return (preds > 0.5).astype(int) 

vgg_rock_model = VggRockModel()