# TimBridge_Project/ImageSimilarityServer/app/repository/imageVector_repository.py

import logging
from qdrant_client import QdrantClient, models
from fastapi import HTTPException
import numpy as np
import uuid

logger = logging.getLogger(__name__)

class VectorRepository:
    def __init__(self):
        logger.info("Initializing VectorRepository and connecting to Qdrant...")
        # 클래스 인스턴스가 생성될 때 Qdrant DB에 연결
        self.client = QdrantClient(host="localhost", port=6333)
        # Docker 서비스 이름 사용 (docker-compose.yml에 정의되 이름)
        # Docker의 내부 네트워크에서는 해당 서비스 이름을 호스트 주소처럼 사용하여 다른 컨테이너에 접근 가능
        self.collection_name = "image_embeddings"
        logger.info(f"Connected to Qdrant. Using collection: {self.collection_name}")

    # ID의 존재 여부를 확인
    def is_exist(self, missingId: str) -> bool:
        logger.info(f"Checking if missingID '{missingId}' exists in the collection.")
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="missingId",
                    match=models.MatchValue(value=missingId)
                )
            ]
        )
        logger.info(f"Search filter: {search_filter}")
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
                filters=search_filter
            )
            is_found = count_result.count > 0
            # 0보다 큰 경우 True
            if is_found:
                logger.info(f"missingID '{missingId}' was found.")
            else:
                logger.info(f"missingID '{missingId}' was not found")
            return is_found
        except Exception as e:
            logger.error(f"Error checking existence for ID {missingId}: {e}.", exc_info=True)
            return False

    async def save(self, vector: np.ndarray, payload: dict) -> bool:
        logger.info(f"Attempting to save point for missingID")
        logger.debug(f"Payload : {payload}")
        point_id = str(uuid.uuid4())
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)
        '''
        models.PointStruct : Qdrant에 저장될 단일 데이터 포인트
        id 고유 식벽자 vector 임베딩 벡터 payload 필터링에 사용할 추가 정보

        하나의 포인트(Point)는 다음과 같이 구성됨
        - ID: person_123(고유 식별자)
        - Vector: [0.1, 0.2, 0.3, ...](512차원의 얼굴 특징벡터)
        - Payload: {"missingID": "person_123", "type": 0, "gender": 1}(추가 메타데이터)
        3가지 정보가 합쳐져서 하나의 완전한 데이터 레코드, 즉 '포인트'를 이룬다.
        '''
        try:
            logger.info(f"Calling client.upsert for missingID")
            self.client.upsert(collection_name=self.collection_name, points=[point], wait=True)
            # client.upsert 해당 id가 없는 경우 새로 추가하고 있으면 덮어쓰기
            logger.info(f"Successfully saved image for ID")
            print("{} 이미지 저장 완료")
            return True
        except Exception as e:
            logger.error(f"Failed to save image for ID: {e}.", exc_info=True)
            return False

    # async def delete(self, missingId: str) -> bool:
    #     self.client.delete(
    #         collection_name=self.collection_name,
    #         points_selector=models.PointIdsList(points=[missingId]),
    #         wait = True
    #     )
    #     print("{} 이미지 벡터 삭제 완료".format(missingId))
    #     return True


    async def delete(self, missingId: str) -> bool:
        try:        
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[missingId]),
                wait = True
            )
            print("{} 이미지 벡터 삭제 완료".format(missingId))
            return True
        except Exception as e:
            logger.error(f"Failed to save image for ID: {e}.", exc_info=True)
            return False

    async def get_data_for_comparison(self, payload: dict) -> dict:
        '''
        유사도 비교에 필요한 모든 데이터를 Qdrant에서 가져옵니다.
        1. 입력 ID의 정보 (target)
        2. 비교 대상이 될 후보 목록 (candidates)
        '''

        target_missingID = payload["missingId"]
        try:
            target_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="missingId",
                        match=models.MatchValue(value=target_missingID)
                    )
                ]
            )

            target_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=target_filter,
                limit=1,
                with_payload=True,
                with_vectors=True
            ) # 리스트의 첫 번째 요소가 포인터에 해당함

            if not target_points:
                raise IndexError("Target point not foun in Qdrant.")
            target_point = target_points[0]  # list

        except IndexError:
            raise HTTPException(status_code=404, detail="찾고자 하는 {}를 DB에서 찾을 수 없습니다.".format(target_missingID))

        target_data = {
            "missingId": target_point.payload.get("missingId"),
            "vector": target_point.vector,
            # "payload": target_point.payload,
        }

        target_type = payload["type"]
        target_gender = payload["gender_id"]

        if target_type == 1:
            opposite_type = 2
        else:
            opposite_type = 1

        search_filter = models.Filter(
            must=[
                models.FieldCondition(key="gender_id", match=models.MatchValue(value=target_gender)),
                models.FieldCondition(key="type", match=models.MatchValue(value=opposite_type)),
            ]
        )

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=target_data["vector"],
            query_filter=search_filter,
            with_payload=True,
            with_vectors=True
        )

        candidates_data=[
            {
                "missingId": hit.payload.get("missingId"),
                "vector": hit.vector, # list
            }
            for hit in hits
        ]

        return {
            "target": target_data, # 비교 주체
            "candidates": candidates_data # 비교 대상 목록
        }