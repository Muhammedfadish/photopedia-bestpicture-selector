import os
import shutil
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

class NIMAScorer:
    def __init__(self, weights_path='D:/COMPANY PROJECTS/PHOTOPEDIA/NEURAL-IMAGE-ASSESSMENT/weights/mobilenet_weights.h5'):
        # Load pretrained MobileNet model as base
        base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=x)
        self.model.load_weights(weights_path)

        self.face_detector = MTCNN(keep_all=True)

    def predict_aesthetic(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)[0]
        mean_score = sum([i * p for i, p in enumerate(preds, 1)])
        return round(mean_score, 2)

    def calculate_sharpness(self, img_cv):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def detect_faces(self, img_rgb):
        boxes, _ = self.face_detector.detect(img_rgb)
        return len(boxes) if boxes is not None else 0

    def score_image(self, img_path):
        # Load image
        img_cv = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        aesthetic = self.predict_aesthetic(img_path)
        sharpness = self.calculate_sharpness(img_cv)
        face_count = self.detect_faces(img_rgb)

        norm_sharpness = min(sharpness / 1000.0, 1.0)

        final_score = (aesthetic * 0.6) + (face_count * 0.2) + (norm_sharpness * 0.2)

        return round(final_score, 2), aesthetic, sharpness, face_count
