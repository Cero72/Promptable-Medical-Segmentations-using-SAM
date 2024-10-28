from fastapi import APIRouter, UploadFile, File, Body
from typing import List
from app.services.segmentation_service import SegmentationService

router = APIRouter()
segmentation_service = SegmentationService()

@router.post("/segment/automatic")
async def segment_image(file: UploadFile = File(...)):
    """Endpoint for automatic segmentation"""
    result = await segmentation_service.process_image(file)
    return result

@router.post("/segment/prompted")
async def segment_with_prompts(
    file: UploadFile = File(...),
    prompts: List[dict] = Body(...)
):
    """Endpoint for prompted segmentation"""
    result = await segmentation_service.process_with_prompts(file, prompts)
    return result
