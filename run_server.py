import uvicorn
import os
from dotenv import load_dotenv

def set_torch_settings():
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

def main():
    # Load environment variables
    load_dotenv()
    
    # Set torch settings
    set_torch_settings()
    
    # Configure server
    config = {
        "app": "app.main:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": True,
        "workers": 1
    }
    
    # Run server
    uvicorn.run(**config)

if __name__ == "__main__":
    main()
