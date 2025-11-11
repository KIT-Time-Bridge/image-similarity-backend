# TimBridge_Project/ImageSimilarityServer/app/model/onnx_model.py

import cv2
import numpy as np
import onnxruntime as ort
from utils.face_utils import face_alignment

class ONNXFaceEngine:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_mean, self.in2put_std = 127.5, 127.5
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_size = tuple(input_cfg.shape[:4][::-1])
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image: np.ndarray):
        return cv2.dnn.blobFromImage(
            image, 1.0 / self.input_std, self.input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray):
        aligned_face = face_alignment(image, landmarks)
        blob = self.preprocess(aligned_face)
        embedding = self.session.run(self.output_names, {self.input_name: blob})[0]
        norm = np.linalg.norm(embedding)
        return embedding / norm