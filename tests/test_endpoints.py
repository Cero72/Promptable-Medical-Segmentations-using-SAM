import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import requests
import time
from requests.exceptions import ConnectionError
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize  # Add this import

class SegmentationTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def check_server(self):
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/docs")
            return response.status_code == 200
        except ConnectionError:
            return False

    def test_automatic_segmentation(self, image_path: str):
        """Test automatic segmentation endpoint"""
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return

        # Check if server is running
        if not self.check_server():
            print("Error: Server is not running. Please start the server first.")
            print("Run 'python run_server.py' in a separate terminal")
            return

        try:
            url = f"{self.base_url}/api/v1/segment/automatic"
            
            # Prepare the file
            with open(image_path, 'rb') as f:
                files = {'file': ('image.jpg', f, 'image/jpeg')}
                
                # Make request
                print("Sending request to server...")
                response = requests.post(url, files=files)
                
                if response.status_code == 200:
                    print("Success!")
                    result = response.json()
                    print("Segmentation Results:")
                    print(f"Number of masks: {len(result['masks'])}")
                    print(f"Confidence scores: {result['confidence_scores']}")
                    
                    # Visualize results if available
                    self.visualize_results(image_path, result['masks'])
                else:
                    print(f"Error: Server returned status code {response.status_code}")
                    print(response.json())
                    
        except ConnectionError:
            print("Error: Could not connect to the server.")
            print("Make sure the server is running (python run_server.py)")
        except Exception as e:
            print(f"Error: {str(e)}")

    def visualize_results(self, image_path: str, masks: list):
        """Visualize the segmentation results"""
        try:
            # Load original image
            image = Image.open(image_path)
            plt.figure(figsize=(15, 5))
            
            # Plot original image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # Plot image with mask overlay
            plt.subplot(1, 2, 2)
            plt.imshow(image)
            for mask in masks:
                plt.imshow(mask, alpha=0.5, cmap='jet')
            plt.title('Segmentation Result')
            plt.axis('off')
            
            plt.show()
        except Exception as e:
            print(f"Error visualizing results: {str(e)}")

def visualize_segmentation_results(image_path, response_data):
    """Visualize original image and segmentation masks"""
    # Load original image
    original_image = np.array(Image.open(image_path))
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot masks
    plt.subplot(132)
    masks = np.array(response_data['masks'])
    
    # Overlay all masks with different colors
    overlay = np.zeros((*original_image.shape[:2], 3))
    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1)]  # Different colors for masks
    
    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        for channel in range(3):
            overlay[:,:,channel] += mask * color[channel]
    
    plt.imshow(original_image, cmap='gray')
    plt.imshow(overlay, alpha=0.5)
    plt.title('Segmentation Masks')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(133)
    plt.imshow(original_image, cmap='gray')
    plt.imshow(np.any(masks, axis=0), alpha=0.5, cmap='jet')
    plt.title('Combined Overlay')
    plt.axis('off')
    
    # Add confidence scores as text
    scores = response_data['confidence_scores']
    plt.figtext(0.02, 0.02, f'Confidence Scores: {[f"{s:.3f}" for s in scores]}')
    
    plt.tight_layout()
    plt.show()

def test_segmentation():
    # Create test image
    image_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    
    if not os.path.exists(image_path):
        print("Test image doesn't exist. Creating one...")
        create_medical_test_image()
    
    print(f"Using image: {image_path}")
    
    # Send to server
    url = "http://localhost:8000/api/v1/segment/automatic"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.png', f, 'image/png')}
            print("Sending request to server...")
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            visualize_results(image_path, result)
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")

def visualize_results(image_path, result):
    """Visualize the original image and segmentation results"""
    # Load original image
    original = np.array(Image.open(image_path))
    
    # Convert masks from list to numpy array if there are any
    masks = np.array(result.get('masks', []))
    scores = result.get('confidence_scores', [])
    
    # Create figure with equal size subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Segmentation Results\nFound {len(masks)} masks', y=1.05)
    
    # Plot original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot segmentation masks
    axes[1].imshow(original, cmap='gray')
    if len(masks) > 0:
        # Create a colormap for multiple masks
        colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
        for mask, color in zip(masks, colors):
            # Resize mask to match original image size
            mask_resized = resize(mask, original.shape[:2], preserve_range=True)
            axes[1].imshow(mask_resized, alpha=0.3, cmap=plt.cm.colors.ListedColormap([color]))
        axes[1].set_title(f'All Masks ({len(masks)} found)')
    else:
        axes[1].set_title('No Masks Found')
    axes[1].axis('off')
    
    # Plot highest confidence mask or message
    axes[2].imshow(original, cmap='gray')
    if len(masks) > 0:
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        # Resize best mask to match original image size
        best_mask_resized = resize(best_mask, original.shape[:2], preserve_range=True)
        axes[2].imshow(best_mask_resized, alpha=0.5, cmap='jet')
        axes[2].set_title(f'Best Mask\nConfidence: {best_score:.2f}')
    else:
        axes[2].set_title('No Masks Found')
    axes[2].axis('off')
    
    # Ensure all subplots have the same size
    plt.tight_layout()
    plt.show()

def create_medical_test_image():
    """Create a more distinctive medical-like test image"""
    # Create a 512x512 grayscale image
    size = 512
    image = np.ones((size, size), dtype=np.uint8) * 50  # Gray background
    
    # Add multiple structures
    Y, X = np.ogrid[:size, :size]
    
    # Add two large circular structures (like lungs)
    for x_offset in [-100, 100]:
        center = (size//2 + x_offset, size//2)
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        image[dist_from_center < 80] = 200
    
    # Add some smaller nodules
    nodules = [(size//2 - 80, size//2 - 30), (size//2 + 120, size//2 + 20)]
    for center in nodules:
        nodule_dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        image[nodule_dist < 15] = 255
    
    # Add some texture
    noise = np.random.normal(0, 5, (size, size))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Save image
    save_path = os.path.join(os.path.dirname(__file__), "test_image.png")
    Image.fromarray(image).save(save_path)
    print(f"Created test image at: {save_path}")
    return save_path

if __name__ == "__main__":
    test_segmentation()
