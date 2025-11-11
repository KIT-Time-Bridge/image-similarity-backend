# TimBridge_Project/ImageSimilarityServer/main.py

from fastapi import FastAPI
from app.router import imageSimilarity_route
from qdrant_client import QdrantClient, models
from app.service.similarity_service import SimilarityService
from app import config

app = FastAPI(title="Image Similarity Server")

similarity_service_instance = SimilarityService()
app.dependency_overrides[imageSimilarity_route.get_similarity_service] = lambda: similarity_service_instance

@app.on_event("startup")
def startup_event():
    # 앱이 시작될 때 Qdrant 컬렉션을 확인하고 없으면 생성한다.
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "image_embeddings"
    try:
        collections_response = client.get_collections()
        existing_collections = [collection.name for collection in collections_response.collections]
        if collection_name not in existing_collections:
            print("컬렉션 'image_embeddings'를 새로 생성합니다.")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=512,
                                                   distance=models.Distance.COSINE),
            )
        else:
            print("컬렉션 'image_embeddings'가 이미 존재합니다.")
    except Exception as e:
        print(f"Qdrant 서버에 연결하거나 컬렉션을 확인하는 중 에러가 발생했습니다: {e}")

app.include_router(imageSimilarity_route.router)