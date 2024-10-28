import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    Resize,
    SpatialPad,
)
from skimage.transform import resize
import os
import requests
from tqdm import tqdm

class MedicalImageSegmentor:
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default checkpoint path if none provided
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "sam_vit_h_4b8939.pth")
        
        # Download checkpoint if it doesn't exist
        if not os.path.exists(checkpoint_path):
            self.download_sam_checkpoint(checkpoint_path)
        
        # Initialize SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        self.confidence_threshold = 0.5  # Lower threshold to get more masks
    
    def download_sam_checkpoint(self, checkpoint_path):
        """Download SAM checkpoint if it doesn't exist"""
        print("Downloading SAM checkpoint...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(checkpoint_path, 'wb') as file, tqdm(
            desc=checkpoint_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        print("Checkpoint downloaded successfully!")

    def preprocess_medical_image(self, image_array):
        """Preprocess medical image"""
        try:
            # Ensure image is in the right format
            if len(image_array.shape) == 2:
                # Convert grayscale to RGB
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
                image_array = np.repeat(image_array, 3, axis=2)
            
            # Ensure uint8 format
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Resize if needed
            if image_array.shape[:2] != (512, 512):
                image_array = resize(image_array, (512, 512, 3), 
                                  preserve_range=True,
                                  anti_aliasing=True).astype(np.uint8)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    async def generate_automatic_masks(self, image, confidence_threshold=0.5):  # Lower default threshold
        """Generate automatic masks without prompts"""
        try:
            self.predictor.set_image(image)
            
            # Generate masks
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=True,
                return_logits=True
            )
            
            # Filter masks by confidence
            valid_indices = scores > confidence_threshold
            
            if not any(valid_indices):  # If no masks pass the threshold
                # Take the top 3 masks regardless of threshold
                top_k = min(3, len(scores))
                top_indices = np.argsort(scores)[-top_k:]
                valid_masks = masks[top_indices]
                valid_scores = scores[top_indices]
            else:
                valid_masks = masks[valid_indices]
                valid_scores = scores[valid_indices]
            
            return valid_masks, valid_scores
            
        except Exception as e:
            raise RuntimeError(f"Error generating masks: {str(e)}")

    async def generate_prompted_mask(self, image, prompt_points, prompt_labels):
        """Generate mask based on user prompts"""
        try:
            self.predictor.set_image(image)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(prompt_points),
                point_labels=np.array(prompt_labels),
                multimask_output=True
            )
            
            # Return the highest confidence mask
            best_mask_idx = np.argmax(scores)
            return masks[best_mask_idx], scores[best_mask_idx]
        except Exception as e:
            raise RuntimeError(f"Error generating prompted mask: {str(e)}")

    def post_process_mask(self, mask):
        """Apply post-processing to improve mask quality"""
        from scipy import ndimage
        
        # Remove small objects
        mask = ndimage.binary_opening(mask)
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        return mask




