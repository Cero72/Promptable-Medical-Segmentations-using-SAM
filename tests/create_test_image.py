import numpy as np
from PIL import Image
import os

def create_medical_test_image():
    # Create a grayscale image (like an X-ray)
    width, height = 512, 512
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Add a bright circular region
    center_x, center_y = width // 2, height // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create main circular structure
    image[dist_from_center < 100] = 200
    
    # Add a smaller bright spot (simulated nodule)
    nodule_center_x = center_x + 30
    nodule_center_y = center_y - 30
    nodule_dist = np.sqrt((X - nodule_center_x)**2 + (Y - nodule_center_y)**2)
    image[nodule_dist < 20] = 255
    
    # Save as PNG to preserve quality
    save_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    Image.fromarray(image).save(save_path)
    print(f"Created test image at: {save_path}")
    return save_path

if __name__ == "__main__":
    create_medical_test_image()
