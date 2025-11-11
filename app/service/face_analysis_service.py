# TimBridge_Project/ImageSimilarityServer/app/service/face_analysis_service.py

import cv2
import numpy as np
import uniface
from app.model.arcface_onnx import ArcFaceONNX
from .. import config
from types import SimpleNamespace

class FaceAnalysisService:
    def __init__(self):
        self.detector = uniface.RetinaFace(model_name=config.DETECTOR_MODEL_NAME)
        # 이미지의 bounding box와 landmarks를 찾는 역할 수행 → 랜드마크 정보를 이용해 얼굴을 반듯하게 정렬
        self.recognizer = ArcFaceONNX(model_file=config.ONNX_MODEL_PATH)
        # 얼굴 인식 모델 로드 → 얼굴의 고유한 특징을 512차원의 벡터(embedding)로 추출
        print("AI 모델 불러오기 완료")

    def get_embedding(self, image_bytes: bytes) -> np.ndarray | None:
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        # 이미지 파일 데이터(바이트 데이터)를 OpenCV가 처리할 수 있는 이미지 형식(NumPy 배열)으로 변환
        _, landmarks = self.detector.detect(image_np)
        if landmarks is None or len(landmarks) == 0:
            print("WARNING: no face detected in image.")
            return None
        face_landmarks = landmarks[0]
        face_object = SimpleNamespace(kps=face_landmarks)
        embedding = self.recognizer.get(image_np, face_object)
        return embedding

    def compute_similarity(self, feat1, feat2):
        feat1 = np.array(feat1)
        feat2 = np.array(feat2)
        similarity = self.recognizer.compute_similarity(feat1, feat2)
        return similarity