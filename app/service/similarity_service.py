# TimBridge_Project/ImageSimilarityServer/app/service/similarity_service.py

from fastapi import HTTPException
import cv2
import numpy as np
import uniface
from app.repository.imageVector_repository import VectorRepository
# from app.utils.face_utils import compute_similarity
from app.service.face_analysis_service import FaceAnalysisService
# 얼굴 검출, 임베딩 추출, 유사도 비교 등 얼굴 분석의 모든 과정을 총괄하는 서비스.
class SimilarityService:
    # 서비스 초기화 시 필요한 모델들을 미리 로드
    def __init__(self):
        self.inference = FaceAnalysisService()
        self.repository = VectorRepository()  # Repository 인스턴스
        print("Inference 및 Repository 준비 완료.")

    # 등록
    async def register_face(self, missingId: str, gender:int, type:int, image_bytes: bytes) -> bool:
        if self.repository.is_exist(missingId):
            raise HTTPException(status_code=409, detail="이미 존재하는 ID입니다.")
        embedding_vector = self.inference.get_embedding(image_bytes=image_bytes) # np.narray
        # if embedding_vector is None:
        #     raise HTTPException(status_code=400, detail="이미지에서 얼굴을 찾지 못해 등록에 실패했습니다.")
        payload = {"missingId": missingId, "gender_id": gender, "type": type}
        await self.repository.save(vector=embedding_vector,payload=payload)
        return True

    async def update_face_image(self, missingId: str, type: int, gender: int, image_bytes: bytes) -> bool:
        if not self.repository.is_exist(missingId):
            raise HTTPException(status_code=404, detail="존재하지 않는 ID입니다.")
        new_embedding_vector = await self.inference.get_embedding(image_bytes=image_bytes)
        if new_embedding_vector is None:
            raise HTTPException(status_code=400, detail="새 이미지에서 얼굴을 찾을 수 없습니다.")
        payload = {"missingId": missingId, "gender_id": gender, "type": type}
        await self.repository.save(new_vector=new_embedding_vector, payload=payload) # repository에서 새로 추가한 update 호출
        return True

    async def delete_face(self, missingId: str) -> bool:
        if not self.repository.is_exist(missingId):
            raise HTTPException(status_code=404, detail="존재하지 않는 ID입니다.")
        return await self.repository.delete(missingId=missingId)

    async def get_similarities_by_id(self, missingId: str, type: int, gender: int) -> dict:
        # ID를 입력받아 Repository에서 데이터를 가져온 후, 후보들과의 유사도를 모두 계산하여 최종 결과를 반환한다.
        payload = {"missingId": missingId, "gender_id": gender, "type": type}
        comparison_data = await self.repository.get_data_for_comparison(payload=payload)
        target = comparison_data["target"]
        candidates = comparison_data["candidates"]
        target_vector = target["vector"] # list

        similarity_results=[]
        for candidate in candidates:
            candidate_vector = candidate["vector"] # list
            score = self.inference.compute_similarity(target_vector, candidate_vector)
            similarity_results.append({
                "missingId": candidate["missingId"],
                "score": score
            })
        return similarity_results