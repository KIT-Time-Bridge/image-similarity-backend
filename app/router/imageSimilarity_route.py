# TimBridge_Project/ImageSimilarityServer/app/router/imageSimilarity_route.py

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Query
from .. import config
import logging
from app.service.similarity_service import SimilarityService
from typing import Annotated

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_similarity_service() -> SimilarityService:
    return similarity_service_instance

@router.post("/insert")
async def register_image(
        missing_id: Annotated[str, Form(...)],
        img: UploadFile = File(...),
        type: int = Form(...),
        gender_id: int = Form(...),
    service: SimilarityService = Depends(get_similarity_service)
):
  # 얼굴 이미지를 임베딩하여 DB에 새로 등록
    logger.info(f"Received request to register image for missing_id: {missing_id}")
    logger.info(f"Form data: type={type}, gender_id={gender_id}")
    image_bytes = await img.read()
    logger.info(f"Image received. Filename:{img.filename}, Size:{len(image_bytes)} bytes")

    try:
        logger.info("Calling service.register_face...")
        await service.register_face(
            missingId=missing_id,
            type=type,
            gender=gender_id,
            image_bytes=image_bytes
        )
        logger.info(f"Successfully registered image for missing_id: {missing_id}")
        return {"success": True, "message": f"ID '{missing_id}'가 성공적으로 등록되었습니다."}
    except HTTPException as e:
        logger.error(f"Failed to register ID {missing_id}. An error occurred: {e}.", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@router.put("/update")
async def update_image(missing_id: str,
                       img: UploadFile = File(...),
                       type: int = Form(...),
                       gender_id: int = Form(...),
                       service: SimilarityService = Depends(get_similarity_service)):
    await service.update_face_image(
        missingId=missing_id,
        type=type,
        gender=gender_id,
        image_bytes=await img.read()
    )
    return {"success": True, "message": f"ID '{missing_id}'의 이미지가 성공적으로 업데이트되었습니다."}

@router.delete("/delete")
async def delete_image(post_id: str,
                       service: SimilarityService = Depends(get_similarity_service)):
    await service.delete_face(missingId=post_id)
    return {"success": True, "message": f"ID '{post_id}'가 성공적으로 삭제되었습니다."}

@router.get("/similarity")
async def get_similarity(missingId: str,
                         type: int = Query(...),
                         gender: int = Query(...),
                         service: SimilarityService = Depends(get_similarity_service)):
    result = await service.get_similarities_by_id(
        missingId=missingId,
        type=type,
        gender=gender)
    return {"result": result}

# return {
#             "targetMissingID": target["missingID"],
#             "result": similarity_results
#         }