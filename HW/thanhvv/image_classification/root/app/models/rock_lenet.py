import tensorflow as tf
import numpy as np
from shared.config import IMG_SIZE_LENET_ROCK

class LeNetRockModel:
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(IMG_SIZE_LENET_ROCK, IMG_SIZE_LENET_ROCK, 3)),
            tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
            tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=84, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
        #print(self.model.summary())
        weights_path = "models/weights/lenet_rock.weights.h5"
        self.model.load_weights(weights_path)
        
  
    def predict(self, img: np.ndarray): 
        
        preds = self.model.predict(img)
        return preds 
        #$return (preds > 0.5).astype(int)




# # Ví dụ sử dụng
# if __name__ == "__main__":
#     model = LeNetRockModel()
#     model.load_weights("models/weights/lenet_rock.weights.h5")

#     img_path = "test_rock.jpg"  # đường dẫn ảnh cần dự đoán
#     img_array = load_and_preprocess_image(img_path, IMG_SIZE_LENET_ROCK)

#     label, confidence = model.predict(img_array)
#     print(f"Dự đoán: {'Rock' if label == 1 else 'Not Rock'}, Độ tin cậy: {confidence:.2f}")
