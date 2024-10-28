from fastapi import UploadFile, HTTPException
import numpy as np
from PIL import Image
import io
from app.models.medical_segmentation import MedicalImageSegmentor

class SegmentationService:
    def __init__(self):
        self.segmentor = MedicalImageSegmentor()
    
    async def process_image(self, file: UploadFile):
        """Process uploaded medical image"""
        try:
            # Read image file
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Preprocess image
            processed_image = self.segmentor.preprocess_medical_image(image_array)
            
            # Generate automatic masks
            masks, scores = await self.segmentor.generate_automatic_masks(processed_image)
            
            # Post-process masks
            processed_masks = [
                self.segmentor.post_process_mask(mask)
                for mask in masks
            ]
            
            return {
                "masks": processed_masks,
                "confidence_scores": scores.tolist()
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def process_with_prompts(self, image_array: np.ndarray, prompts: list):
        """Process image with user-provided prompts"""
        try:
            processed_image = self.segmentor.preprocess_medical_image(image_array)
            
            # Extract prompt points and labels
            points = [[p['x'], p['y']] for p in prompts]
            labels = [1 for _ in prompts]  # 1 for foreground points
            
            # Generate mask
            mask, score = await self.segmentor.generate_prompted_mask(
                processed_image, points, labels
            )
            
            processed_mask = self.segmentor.post_process_mask(mask)
            
            return {
                "mask": processed_mask,
                "confidence_score": float(score)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
