import numpy as np
import json
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
import os
import yolo
from yolo.frontend import create_yolo



class HouseNumberDetector(object):
    def __init__(self):
        self.yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)
        DEFAULT_WEIGHT_FILE = 'C:\\Users\\Simas\\Desktop\\Insight\\Data Challenges\\House-Number-Detection\\HND\\Yolo\\Yolo-digit-detector\\weights.h5'
        self.yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)

    def predict_numbers(self, img, threshold=0.3):
        boxes, probs = self.yolo_detector.predict(img, threshold)
        out = ''
        for i in probs:
            out += str(np.argmax(i))
        return int(out)