# TimBridge_Project/ImageSimilarityServer/app/utils.face_utils.py

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

'''
고정된 기준 좌표
112x112 크기의 이미지에서, 가장 완벽하게 정면을 바라보는 얼굴의 5개 주요 지점(양쪽 눈, 코 끝, 양쪽 입꼬리)에 대한 표준 좌표값을 미리 정의해 둔 것
'''
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)

# 입력된 얼굴의 랜드마크를 reference_alignment에 맞출려면 어떻게 변화해야 하는 지 계산
# Estimate the normalization transformation matrix for facial landmarks.
def estimate_norm(landmark: np.ndarray, image_size:int = 112) -> np.ndarray:
    assert landmark.shape == (5, 2), "Landmark array must have shape (5, 2)."
    assert image_size % 112 == 0, "Image size must be a multiple of 112."
    ratio = float(image_size) / 112.0
    alignment = reference_alignment * ratio
    transform = SimilarityTransform() # 실제 이미지에서 찾아낸 랜드마크가 이상적인 기준점과 일치하도록 만드는 최적읜 transformation matrix를 찾아낸다.
    transform.estimate(landmark, alignment)
    matrix = transform.params[0:2, :]
    return matrix

# 얼굴 정렬
# Align the face in the input image based on the given facial landmarks.
def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size:int = 112) -> np.ndarray:
    # image (np.ndarray): Input image as a NumPy array.
    # landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
    transformation_matrix = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(image, transformation_matrix, (image_size, image_size), borderValue=0.0)
    # 행렬 M의 지시대로 원본 이미지의 픽셀들을 재배치하여, 기울어짐과 크기가 보정된 새로운 112x112 크기의 얼굴 이미지를 생성
    # 정렬된 이미지가 임베딩 모델의 최종 입력값이 됨
    return warped

# 유사도 점수 계산 - 코사인 유사도
# Computing Similarity between two faces.
def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32:
    feat1 = np.array(feat1)
    feat2 = np.array(feat2)

    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    similarity =  np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity
# 두 벡터가 가리키는 방향이 비슷할수록(두 얼굴이 닮을수록) 1에 가까운 값이 나오고, 방향이 다를수록 0에 가까운 값이 나옴