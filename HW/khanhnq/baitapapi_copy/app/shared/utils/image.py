import os
import cv2
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from git import Repo

# filepath = 'temp_concrete_crack'
# Repo.clone_from('https://github.com/bimewok/Concrete-Crack-Image-Classifier', filepath)

base_dir = 'app/temp_concrete_crack/data/concrete_images'

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')